import json
import os
from typing import Dict, List, Tuple

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, TrainingArguments
from transformers.data.data_collator import default_data_collator

from src.model.qa_model import QAModel
from src.model.qa_trainer import QATrainer
from src.model.qa_utils import postprocess_qa_predictions, compute_additional_metrics


class QAModelTester:
    def __init__(self, model_path: str = "./qa_checkpoints/final", test_data_dir: str = "./data/test",
                 results_dir: str = "./test_results", batch_size: int = 32):
        self.model_path = model_path
        self.test_data_dir = test_data_dir
        self.results_dir = results_dir
        self.batch_size = batch_size
        os.makedirs(self.results_dir, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.test_dataset = None
        self.original_test = None

    def _load_model_and_data(self):
        self.model = QAModel.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.test_dataset = load_from_disk(self.test_data_dir)["test"]
        self.original_test = load_dataset("squad_v2")["validation"]

        required_columns = [
            "input_ids", "attention_mask", "token_type_ids",
            "offset_mapping", "example_id"
        ]
        columns_to_remove = [col for col in self.test_dataset.column_names if col not in required_columns]
        self.test_dataset = self.test_dataset.remove_columns(columns_to_remove)

    def _setup_trainer(self):
        self.trainer = QATrainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=self.results_dir,
                per_device_eval_batch_size=self.batch_size,
                fp16=torch.cuda.is_available(),
            ),
            data_collator=default_data_collator,
        )

    def _get_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        predictions = self.trainer.predict(self.test_dataset)
        return predictions.predictions  # (start_logits, end_logits, no_answer_logits)

    def _postprocess_predictions(self, start_logits: np.ndarray, end_logits: np.ndarray,
                                 no_answer_logits: np.ndarray) -> List[Dict]:
        return postprocess_qa_predictions(
            examples=self.original_test,
            features=self.test_dataset,
            raw_predictions=(start_logits, end_logits, no_answer_logits),
            tokenizer=self.tokenizer,
        )

    def _compute_metrics(self, predictions: List[Dict]):
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in self.original_test]

        squad_metric = evaluate.load("squad_v2")
        squad_scores = squad_metric.compute(predictions=predictions, references=references)

        no_answer_acc = np.mean([
            (pred["prediction_text"] == "") == (len(ref["answers"]["text"]) == 0)
            for pred, ref in zip(predictions, references)
        ])

        extra_metrics = compute_additional_metrics(predictions, references)
        return squad_scores, extra_metrics, no_answer_acc

    def _print_results(self, squad_scores: Dict, extra_metrics: Dict, no_answer_acc: float):
        print("TEST RESULTS")
        print(f"Exact Match (EM): {squad_scores['exact']:.4f}")
        print(f"F1 Score: {squad_scores['f1']:.4f}")
        print(f"No-Answer Accuracy: {no_answer_acc:.4f}")
        print(f"Answerable Precision: {extra_metrics['precision_answerable']:.4f}")
        print(f"Answerable Recall: {extra_metrics['recall_answerable']:.4f}")

        confusion = extra_metrics['no_answer_confusion']
        print(f"\nNo-Answer Confusion Matrix:")
        print(f"True Positives (TP): {confusion['TP']}")
        print(f"True Negatives (TN): {confusion['TN']}")
        print(f"False Positives (FP): {confusion['FP']}")
        print(f"False Negatives (FN): {confusion['FN']}")

    def _save_results(self, squad_scores: Dict, extra_metrics: Dict, no_answer_acc: float,
                      predictions: List[Dict]):
        results = {
            "exact_match": squad_scores["exact"],
            "f1_score": squad_scores["f1"],
            "no_answer_accuracy": no_answer_acc,
            "precision_answerable": extra_metrics["precision_answerable"],
            "recall_answerable": extra_metrics["recall_answerable"],
            "confusion_matrix": extra_metrics["no_answer_confusion"]
        }

        with open(os.path.join(self.results_dir, "test_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(os.path.join(self.results_dir, "test_predictions.csv"), index=False)

    def test_model(self):
        self._load_model_and_data()
        self._setup_trainer()
        start_logits, end_logits, no_answer_logits = self._get_predictions()
        predictions = self._postprocess_predictions(start_logits, end_logits, no_answer_logits)
        squad_scores, extra_metrics, no_answer_acc = self._compute_metrics(predictions)
        self._print_results(squad_scores, extra_metrics, no_answer_acc)
        self._save_results(squad_scores, extra_metrics, no_answer_acc, predictions)


if __name__ == "__main__":
    tester = QAModelTester()
    tester.test_model()
