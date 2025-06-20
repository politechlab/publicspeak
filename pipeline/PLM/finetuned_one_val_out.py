import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4,7"
from transformers import AutoTokenizer
import torch
import json
import pandas as pd
from transformers import DataCollatorWithPadding
import random
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prfs
import datasets
from datasets import Dataset, DatasetDict
import functools 
import evaluate
import argparse
from config import Paths, Settings
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path


class PLMProcessor:
    def __init__(self, 
                 model_name: str = Settings.MODEL_NAME,
                 device: str = "cuda",
                 seed: int = Settings.SEED):
        """
        初始化PLM处理器
        
        Args:
            model_name: 模型名称
            device: 设备
            seed: 随机种子
        """
        self.model_name = model_name
        self.device = device
        self.seed = seed
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # 设置随机种子
        self._seed_everything(seed)
        
    def _seed_everything(self, seed_value: int) -> None:
        """设置随机种子"""
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        os.environ['PYTHONHASHSEED'] = str(seed_value)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
            
    def _load_model(self) -> None:
        """加载模型和tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=3
        )
        self.model.to(self.device)
        
    def _prepare_data(self, data: List[Dict[str, Any]]) -> DatasetDict:
        """准备数据"""
        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True)
        
        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(df)
        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        
        return DatasetDict({"predict": tokenized_dataset})
        
    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """计算评估指标"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = evaluate.load("accuracy")
        return accuracy.compute(predictions=predictions, references=labels)

    def train(self, 
              train: List[pd.DataFrame],
              val: Optional[List[pd.DataFrame]] = None,
              lr: float = Settings.LEARNING_RATE,
              epoch: int = Settings.EPOCHS,
              bs: int = Settings.PLM_BATCH_SIZE) -> None:
        """
        训练模型，只做训练，不返回任何内容
        
        Args:
            train: 训练数据
            val: 验证数据，如果为None则使用train作为验证集
            lr: 学习率
            epoch: 训练轮数
            bs: batch size
        """
        self._load_model()
        training_set = train
        val_set = val if val is not None else train  # 如果没有val，就用train
        
        # 合并数据集
        if training_set:
            train_df = pd.concat(training_set, ignore_index=True)
            tds = Dataset.from_pandas(train_df)
        else:
            tds = Dataset.from_pandas(pd.DataFrame(columns=["text", "label"]))
            
        if val_set:
            val_df = pd.concat(val_set, ignore_index=True)
            vds = Dataset.from_pandas(val_df)
        else:
            vds = Dataset.from_pandas(pd.DataFrame(columns=["text", "label"]))

        ds = DatasetDict()
        ds['train'] = tds
        ds['val'] = vds

        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True)

        tokenized_ds = ds.map(preprocess_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        model_output_dir = Paths.PLM_DIR / "models"
        model_output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(model_output_dir),
            learning_rate=lr,
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs,
            num_train_epochs=epoch,
            weight_decay=0.01,
            evaluation_strategy="epoch",  # 每个epoch进行评估
            save_strategy="epoch",  # 不保存检查点
            load_best_model_at_end=True,  # 加载最佳模型
        )

        # 创建trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_ds["train"],
            eval_dataset=tokenized_ds["val"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )

        # 训练模型
        self.trainer.train()

    def predict(self, test_set: List[pd.DataFrame]) -> List[int]:
        """
        对test_set进行推理，返回预测结果
        
        Args:
            test_set: 测试数据列表，每个元素是一个DataFrame，包含text列
            
        Returns:
            List[int]: 预测结果列表
        """
        if not test_set:
            return []
        test_df = pd.concat(test_set, ignore_index=True)
        ttd = Dataset.from_pandas(test_df)
        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True)
        tokenized_ds = ttd.map(preprocess_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        predictions = trainer.predict(tokenized_ds)
        pred = np.argmax(predictions.predictions, axis=1).tolist()
        return pred

    def get_metrics(self, y_true: List[int], y_pred: List[int]) -> dict:
        """
        输入真实标签和预测标签，返回metrics字典
        """
        pre0, pre1, rec0, rec1, f10, f11 = precision_recall(y_pred, y_true)
        pre0_b, pre1_b, rec0_b, rec1_b, f10_b, f11_b = precision_recall_bad(y_pred, y_true)
        metrics = {
            "precision_comment": pre0,
            "precision_hearing": pre1,
            "recall_comment": rec0,
            "recall_hearing": rec1,
            "f1_comment": f10,
            "f1_hearing": f11,
            "precision_comment_pess": pre0_b,
            "precision_hearing_pess": pre1_b,
            "recall_comment_pess": rec0_b,
            "recall_hearing_pess": rec1_b,
            "f1_comment_pess": f10_b,
            "f1_hearing_pess": f11_b
        }
        return metrics

    def load_trained_model(self, model_path: Optional[str] = None) -> None:
        """
        加载训练好的模型
        
        Args:
            model_path: 模型路径，如果为None则使用默认路径
        """
        if model_path is None:
            model_path = str(Paths.PLM_DIR / "models")
            
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # 加载训练好的模型
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        
        print(f"Model loaded from: {model_path}")

    def process_transcript(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理转录文本
        
        Args:
            transcript_data: 转录文本数据
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Please call load_trained_model first.")
            
        # 准备数据
        data = []
        for i, item in enumerate(transcript_data):
            data.append({
                'text': item['text']
            })
        
        # 准备数据集
        dataset = self._prepare_data(data)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # 创建trainer
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # 进行预测
        predictions = trainer.predict(dataset["predict"])
        pred = np.argmax(predictions.predictions, axis=1).tolist()
        
        # 整理结果
        result = pred
        
        return result

    # TODO: Check if this function is needed.
    def process_test_file(self, test_file: str) -> Dict[str, Any]:
        """
        处理测试文件，与main函数中的逻辑一致
        
        Args:
            test_file: 测试文件路径
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Please call load_trained_model first.")
            
        # 加载测试数据
        with open(test_file) as f:
            test = json.load(f)
        
        # 准备测试集（不需要label）
        test_data = []
        for k in test:
            df = pd.DataFrame([[str(v["text"])] for v in test[k]], columns=["text"])
            test_data.append(df)
        
        # 进行预测
        test_pred = self.predict(test_data)
        
        # 整理结果
        result = {
            "pred": test_pred
        }
        
        return result


def compute_precision_recall(y_test, y_pred_encode):
    t = prfs(y_test, y_pred_encode, average=None)
    y_s = set(y_test)
    y_pred_s = set(y_pred_encode)
    if 2 not in y_s and 2 not in y_pred_s:
        return t[0][1], 1, t[1][1], 1, t[2][1], 1
    if 1 not in y_s and 1 not in y_pred_s:
        return 1, t[0][1], 1, t[1][1], 1, t[2][1]
    return t[0][1], t[0][2], t[1][1], t[1][2], t[2][1], t[2][2]


def compute_precision_recall_bad(y_test, y_pred_encode):
    t = prfs(y_test, y_pred_encode, average=None)
    y_s = set(y_test)
    y_pred_s = set(y_pred_encode)
    if 2 not in y_s and 2 not in y_pred_s:
        return t[0][1], None, t[1][1], None, t[2][1], None
    if 1 not in y_s and 1 not in y_pred_s:
        return None, t[0][1], None, t[1][1], None, t[2][1]
    return t[0][1], t[0][2], t[1][1], t[1][2], t[2][1], t[2][2]


def precision_recall(y, y_pred):
    pre0, pre1, rec0, rec1, f10, f11 = compute_precision_recall(y, y_pred)
    return pre0, pre1, rec0, rec1, f10, f11


def precision_recall_bad(y, y_pred):
    pre0, pre1, rec0, rec1, f10, f11 = compute_precision_recall_bad(y, y_pred)
    return pre0, pre1, rec0, rec1, f10, f11


def main(args):
    
    # 创建处理器实例
    print(args)
    processor = PLMProcessor(
        model_name=args.plm_model_name,
        seed=args.seed
    )
    
    # 加载数据
    train_file = Paths.RAW_TRAIN_DIR / Settings.TRAIN_FILE.format(city=args.city)
    test_file = Paths.RAW_TEST_DIR / Settings.TEST_FILE.format(city=args.city)
    
    with open(train_file) as f:
        train = json.load(f)
    with open(test_file) as f:
        test = json.load(f)
        
    # 检查验证集文件是否存在
    val = None
    if hasattr(Settings, 'VAL_FILE'):
        val_file = Paths.RAW_DIR / Settings.VAL_FILE.format(city=args.city)
        if val_file.exists():
            with open(val_file) as f:
                val = json.load(f)
    
    # 准备数据
    def assign_label(val):
        try:
            if val['is_public_comment']:
                return 1
            elif val['is_public_hearing']:
                return 2
            return 0
        except:
            return 0
            
    # 准备训练集
    train_data = []
    for k in train:
        df = pd.DataFrame([[str(v["text"]), assign_label(v)] for v in train[k]], columns=["text", "label"])
        train_data.append(df)
    
    # 准备验证集
    val_data = None
    if val is not None:
        val_data = []
        for k in val:
            df = pd.DataFrame([[str(v["text"]), assign_label(v)] for v in val[k]], columns=["text", "label"])
            val_data.append(df)
    
    # 准备测试集
    test_data = []
    for k in test:
        df = pd.DataFrame([[str(v["text"]), assign_label(v)] for v in test[k]], columns=["text", "label"])
        test_data.append(df)
    
    # 训练模型
    processor.train(
        train=train_data,
        val=val_data,  # 如果没有验证集，val_data为None，train方法会使用训练集作为验证集
        lr=args.lr,
        epoch=args.epoch,
        bs=args.plm_batch_size
    )
    
    # 预测并评估
    test_pred = processor.predict(test_data)
    test_labels = [label for df in test_data for label in df['label']]
    metrics = processor.get_metrics(test_labels, test_pred)
    
    # 获取训练集和验证集的预测结果
    train_pred = processor.predict(train_data)
    val_pred = processor.predict(val_data) if val_data is not None else []
    
    # 保存结果
    result = {
        "pred": test_pred,
        "train_pred": train_pred,
        "val_pred": val_pred,
        "metrics": metrics
    }
    
    output_file = Paths.PLM_DIR / Settings.PLM_PRED_FILE.format(city=args.city)
    with open(output_file, "w") as f:
        json.dump(result, f)
    
    # 保存模型
    model_output_dir = Paths.PLM_DIR / "models"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存模型和tokenizer
    processor.model.save_pretrained(str(model_output_dir))
    processor.tokenizer.save_pretrained(str(model_output_dir))
    
    print(f"Model saved to: {model_output_dir}")


if __name__ == "__main__":
    main()