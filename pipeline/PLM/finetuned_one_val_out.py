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
        Initialize PLM processor.

        Args:
            model_name: Model name.
            device: Device (e.g. cuda/cpu).
            seed: Random seed.
        """
        self.model_name = model_name
        self.device = device
        self.seed = seed
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Set random seed
        self._seed_everything(seed)
        
    def _seed_everything(self, seed_value: int) -> None:
        """Set random seed."""
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
        """Load model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=3
        )
        self.model.to(self.device)
        
    def _prepare_data(self, data: List[Dict[str, Any]]) -> DatasetDict:
        """Prepare data."""
        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True)
        
        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(df)
        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        
        return DatasetDict({"predict": tokenized_dataset})
        
    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute evaluation metrics."""
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
        Train the model (training only, no return value).

        Args:
            train: Training data.
            val: Validation data; if None, train is used as validation set.
            lr: Learning rate.
            epoch: Number of epochs.
            bs: Batch size.
        """
        self._load_model()
        training_set = train
        val_set = val if val is not None else train  # Use train as val if val is None

        # Merge datasets
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
            evaluation_strategy="epoch",  # Evaluate every epoch
            save_strategy="epoch",  # Save every epoch
            load_best_model_at_end=True,  # Load best model at end
        )

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_ds["train"],
            eval_dataset=tokenized_ds["val"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )

        # Train model
        self.trainer.train()

    def predict(self, test_set: List[pd.DataFrame]) -> List[int]:
        """
        Run inference on test_set and return predictions.

        Args:
            test_set: List of test DataFrames, each with a text column.

        Returns:
            List[int]: List of predicted labels.
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
        Compute metrics from ground-truth and predicted labels.
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
        Load trained model.

        Args:
            model_path: Model path; if None, use default path.
        """
        if model_path is None:
            model_path = str(Paths.PLM_DIR / "models")
            
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load trained model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        
        print(f"Model loaded from: {model_path}")

    def process_transcript(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process transcript text.

        Args:
            transcript_data: Transcript data.

        Returns:
            Dict[str, Any]: Processing result.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Please call load_trained_model first.")
            
        # Prepare data
        data = []
        for i, item in enumerate(transcript_data):
            data.append({
                'text': item['text']
            })
        
        # Prepare dataset
        dataset = self._prepare_data(data)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # Run prediction
        predictions = trainer.predict(dataset["predict"])
        pred = np.argmax(predictions.predictions, axis=1).tolist()

        # Format result
        result = pred
        
        return result

    # TODO: Check if this function is needed.
    def process_test_file(self, test_file: str) -> Dict[str, Any]:
        """
        Process test file (logic aligned with main).

        Args:
            test_file: Path to test file.

        Returns:
            Dict[str, Any]: Processing result.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Please call load_trained_model first.")
            
        # Load test data
        with open(test_file) as f:
            test = json.load(f)
        
        # Prepare test set (no labels needed)
        test_data = []
        for k in test:
            df = pd.DataFrame([[str(v["text"])] for v in test[k]], columns=["text"])
            test_data.append(df)
        
        # Run prediction
        test_pred = self.predict(test_data)

        # Format result
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
    
    # Create processor instance
    print(args)
    processor = PLMProcessor(
        model_name=args.plm_model_name,
        seed=args.seed
    )
    
    # Load data

    train_file = args.raw_train_dir / args.train_file
    test_file = args.raw_test_dir / args.test_file
    
    with open(train_file) as f:
        train = json.load(f)
    with open(test_file) as f:
        test = json.load(f)
        
    # Check if validation file exists
    val = None
    if args.raw_eval_dir and args.eval_file:
        val_file = args.raw_eval_dir / args.eval_file
        if os.path.exists(val_file):
            with open(val_file) as f:
                val = json.load(f)
    
    # Prepare data
    def assign_label(val):
        try:
            if val['is_public_comment']:
                return 1
            elif val['is_public_hearing']:
                return 2
            return 0
        except:
            return 0
            
    # Prepare training set
    train_data = []
    for k in train:
        df = pd.DataFrame([[str(v["text"]), assign_label(v)] for v in train[k]], columns=["text", "label"])
        train_data.append(df)
    
    # Prepare validation set
    val_data = None
    if val is not None:
        val_data = []
        for k in val:
            df = pd.DataFrame([[str(v["text"]), assign_label(v)] for v in val[k]], columns=["text", "label"])
            val_data.append(df)
    
    # Prepare test set
    test_data = []
    for k in test:
        df = pd.DataFrame([[str(v["text"]), assign_label(v)] for v in test[k]], columns=["text", "label"])
        test_data.append(df)
    
    # Train model
    processor.train(
        train=train_data,
        val=val_data,  # If no validation set, train() uses training set as validation
        lr=args.lr,
        epoch=args.epoch,
        bs=args.plm_batch_size
    )
    
    # Predict and evaluate
    test_pred = processor.predict(test_data)
    test_labels = [label for df in test_data for label in df['label']]
    metrics = processor.get_metrics(test_labels, test_pred)
    
    # Get predictions for train and validation sets
    train_pred = processor.predict(train_data)
    val_pred = processor.predict(val_data) if val_data is not None else []
    
    # Save results
    result = {
        "pred": test_pred,
        "train_pred": train_pred,
        "val_pred": val_pred,
        "metrics": metrics
    }
    
    output_file = Paths.PLM_DIR / Settings.PLM_PRED_FILE
    with open(output_file, "w") as f:
        json.dump(result, f)
    
    # Save model
    if args.save_plm_model:
        model_output_dir = Paths.PLM_DIR / "models"
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        processor.model.save_pretrained(str(model_output_dir))
        processor.tokenizer.save_pretrained(str(model_output_dir))

        print(f"Model saved to: {model_output_dir}")


if __name__ == "__main__":
    main()