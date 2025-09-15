import os
from typing import Dict

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk, load_dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from transformers.data.data_collator import default_data_collator

from src.model.qa_feezer import GradualUnfreezingCallback
from src.model.qa_model import QAModel
from src.model.qa_trainer import QATrainer
from src.model.qa_utils import postprocess_qa_predictions, compute_additional_metrics, get_optimizer_params

class QAModelTrainer:
    def __init__(self, model_name: str = "bert-base-uncased",
                 data_dir: str = "/kaggle/input/squad-2-test/processed_squad_v2/train_valid",
                 save_dir: str = "./qa_checkpoints",
                 batch_size_train: int = 16, batch_size_eval: int = 32, num_epochs: int = 3):
        self.model_name = model_name
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.batch_size_train = batch_size_train
        self.batch_size_eval = batch_size_eval
        self.num_epochs = num_epochs
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "logs"), exist_ok=True)

        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.train_dataset = None
        self.val_dataset = None
        self.original_val = None
        self.squad_metric = evaluate.load("squad_v2")

    def _load_data(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.train_dataset = load_from_disk(os.path.join(self.data_dir, "train"))
        self.val_dataset = load_from_disk(os.path.join(self.data_dir, "validation"))
        self.original_val = load_dataset("squad_v2")["validation"]

        required_columns = [
            "input_ids", "attention_mask", "token_type_ids", "start_positions",
            "end_positions", "is_impossible", "offset_mapping", "example_id"
        ]
        columns_to_remove = [col for col in self.train_dataset.column_names if col not in required_columns]
        self.train_dataset = self.train_dataset.remove_columns(columns_to_remove)
        self.val_dataset = self.val_dataset.remove_columns(columns_to_remove)

    def _compute_metrics(self, p) -> Dict:
        start_logits, end_logits, no_answer_logits = p.predictions
        predictions = postprocess_qa_predictions(
            examples=self.original_val,
            features=self.val_dataset,
            raw_predictions=(start_logits, end_logits, no_answer_logits),
            tokenizer=self.tokenizer,
        )
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in self.original_val]

        squad_scores = self.squad_metric.compute(predictions=predictions, references=references)
        no_answer_acc = np.mean([
            (pred["prediction_text"] == "") == (len(ref["answers"]["text"]) == 0)
            for pred, ref in zip(predictions, references)
        ])
        extra = compute_additional_metrics(predictions, references)

        return {
            "eval_exact": squad_scores["exact"],
            "eval_f1": squad_scores["f1"],
            "eval_no_answer_accuracy": no_answer_acc,
            **extra
        }

    def _setup_trainer(self):
        self.model = QAModel(self.model_name)
        optimizer = AdamW(get_optimizer_params(self.model), betas=(0.9, 0.999), eps=1e-6)

        args = TrainingArguments(
            output_dir=self.save_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size_train,
            per_device_eval_batch_size=self.batch_size_eval,
            num_train_epochs=self.num_epochs,
            weight_decay=0.01,
            gradient_accumulation_steps=2,
            logging_dir=os.path.join(self.save_dir, "logs"),
            logging_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            report_to=["tensorboard"],
            fp16=torch.cuda.is_available(),
            max_grad_norm=1.0,
            warmup_ratio=0.1,
        )

        self.trainer = QATrainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=default_data_collator,
            compute_metrics=self._compute_metrics,
            optimizers=(optimizer, None),
        )

        self.trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=1))
        self.trainer.add_callback(GradualUnfreezingCallback(self.model))

    def _save_results(self):
        self.trainer.save_model(os.path.join(self.save_dir, "final"))
        df_logs = pd.DataFrame(self.trainer.state.log_history)
        df_logs.to_csv(os.path.join(self.save_dir, "training_logs.csv"), index=False)
        print(f"Training finished. Model and logs saved to {self.save_dir}")

    def train_model(self):
        self._load_data()
        self._setup_trainer()
        self.trainer.train()
        self._save_results()


if __name__ == "__main__":
    trainer = QAModelTrainer()
    trainer.train_model()
