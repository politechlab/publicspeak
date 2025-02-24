#!/usr/bin/env python
# coding: utf-8

import os
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


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def main(model_name="roberta-large", city="AA", lr=2e-5, epoch=7, seed=42):
    model_name = args.model_name
    city = args.city
    lr = args.lr
    epoch = args.epoch
    seed = args.seed
    
    
    seed_everything(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    with open(current_dir + f"/../../data/raw_train/{city}_train.json") as f:
        train = json.load(f)
        
    with open(current_dir + f"/../../data/raw_val/{city}_val.json") as f:
        val = json.load(f)
        
    with open(current_dir + f"/../../data/raw_test/{city}_test.json") as f:
        test = json.load(f)
        
    merged = []
    train_limit = len(train.keys())
    test_limit = len(train.keys()) + len(val.keys())
    print(train_limit)
    print(test_limit)
    for k in train:
        merged.append(train[k])
    for k in val:
        merged.append(val[k])
    for k in test:
        merged.append(test[k])
    
    def assign_label(val):
        try:
            if val['is_public_comment']:
                return 1
            elif val['is_public_hearing']:
                return 2
            return 0
        except:
            return 0

    data = []
    for val in merged:
        df = pd.DataFrame([[str(v["text"]), assign_label(v)] for v in val], columns=["text", "label"])
        data.append(df)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_precision_recall(y_test, y_pred_encode):

        t = prfs(y_test, y_pred_encode, average=None)
        y_s = set(y_test)
        y_pred_s = set(y_pred_encode)
        if 2 not in y_s and 2 not in y_pred_s:
            return t[0][1], 1, t[1][1], 1, t[2][1], 1
        return t[0][1], t[0][2], t[1][1], t[1][2], t[2][1], t[2][2]
    def compute_precision_recall(y_test, y_pred_encode):

        t = prfs(y_test, y_pred_encode, average=None)
        y_s = set(y_test)
        y_pred_s = set(y_pred_encode)
        if 2 not in y_s and 2 not in y_pred_s:
            # TODO: assign N/A
            return t[0][1], 1, t[1][1], 1, t[2][1], 1
        if 1 not in y_s and 1 not in y_pred_s:
            # TODO: assign N/A
            return 1, t[0][1], 1, t[1][1], 1, t[2][1]
        return t[0][1], t[0][2], t[1][1], t[1][2], t[2][1], t[2][2]

    def compute_precision_recall_bad(y_test, y_pred_encode):

        t = prfs(y_test, y_pred_encode, average=None)
        y_s = set(y_test)
        y_pred_s = set(y_pred_encode)
        if 2 not in y_s and 2 not in y_pred_s:
            # TODO: assign N/A
            return t[0][1], None, t[1][1], None, t[2][1], None
        if 1 not in y_s and 1 not in y_pred_s:
            # TODO: assign N/A
            return None, t[0][1], None, t[1][1], None, t[2][1]
        return t[0][1], t[0][2], t[1][1], t[1][2], t[2][1], t[2][2]

    def precision_recall(y, y_pred):
        pre0, pre1, rec0, rec1, f10, f11 = compute_precision_recall(y, y_pred)
        return  pre0, pre1, rec0, rec1, f10, f11

    def precision_recall_bad(y, y_pred):
        pre0, pre1, rec0, rec1, f10, f11 = compute_precision_recall_bad(y, y_pred)
        return  pre0, pre1, rec0, rec1, f10, f11

    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    )

    def get_pred(data, train_limit = 0, test_limit=0):
        
        # The ith group is used as the test set
        val_set = data[train_limit: test_limit]
        for i in val_set:
            print(len(i))
        # All other groups are combined into the training set
        training_set = data[:train_limit]
        
        test_set = data[test_limit:]
        
        for ii in training_set:
            for i, item in ii.iterrows():
                for k in item:
                    if type(item["text"]) != str:
                        print(i, item["text"], type(item["text"]))
        tds = Dataset.from_pandas(functools.reduce(lambda a, b: a.append(b, ignore_index=True), training_set))
        vds = Dataset.from_pandas(functools.reduce(lambda a, b: a.append(b, ignore_index=True), val_set))
        ttd = Dataset.from_pandas(functools.reduce(lambda a, b: a.append(b, ignore_index=True), test_set))

        ds = DatasetDict()

        ds['train'] = tds
        ds['val'] = vds
        ds['test'] = ttd

        tokenized_ds = ds.map(preprocess_function, batched=True)

        training_args = TrainingArguments(
            output_dir="plm_model",
            learning_rate=lr,
            per_device_train_batch_size=10,
            per_device_eval_batch_size=10,
            num_train_epochs=epoch,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds["train"],
            eval_dataset=tokenized_ds["val"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        predictions = trainer.predict(tokenized_ds["test"])
        pred = np.argmax(predictions.predictions, axis=1).tolist()
        val_predictions = trainer.predict(tokenized_ds["val"])
        val_pred = np.argmax(val_predictions.predictions, axis=1).tolist()
        label = predictions.label_ids.tolist()
        val_label = val_predictions.label_ids.tolist()

        predictions_train = trainer.predict(tokenized_ds["train"])
        pred_train = np.argmax(predictions_train.predictions, axis=1).tolist()

        pre0, pre1, rec0, rec1, f10, f11 = precision_recall(pred, label)
        pre0_b, pre1_b, rec0_b, rec1_b, f10_b, f11_b = precision_recall_bad(pred, label)
        
        pre0_val, pre1_val, rec0_val, rec1_val, f10_val, f11_val = precision_recall_bad(val_pred, val_label)

        return  pre0_val, pre1_val, rec0_val, rec1_val, f10_val, f11_val, pre0, pre1, rec0, rec1, f10, f11, pre0_b, pre1_b, rec0_b, rec1_b, f10_b, f11_b, pred, label, pred_train, val_pred

    pre0_val, pre1_val, rec0_val, rec1_val, f10_val, f11_val, pre0, pre1, rec0, rec1, f10, f11, pre0_b, pre1_b, rec0_b, rec1_b, f10_b, f11_b, pred, label, pred_train, val_pred = get_pred(data, train_limit, test_limit)

    if "roberta" in model_name:
        name = "roberta"
    else:
        name = "distilbert"

    metrics = {"precision_comment_val": pre0_val, 
               "precision_hearing_val": pre1_val, 
               "recall_comment_val": rec0_val, 
               "recall_hearing_val": rec1_val, 
               "f1_comment_val": f10_val, 
               "f1_hearing_val": f11_val,
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
               "f1_hearing_pess": f11_b, 
              }
    
    
    out_dir = os.path.abspath(os.path.join(current_dir, "../../data/PLM_indicators"))
    with open(out_dir + f"/{city}_pred_LOO_roberta.json", "w") as f:
        json.dump({"pred": pred, "train_pred": pred_train, "val_pred": val_pred, "metrics": metrics}, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("--model_name", type=str, default="roberta-large")
    parser.add_argument("--city", type=str, default="AA")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epoch", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    main(args)