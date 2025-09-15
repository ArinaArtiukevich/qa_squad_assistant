import collections
import os

import evaluate
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt


def freeze_embeddings(model):
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False


def unfreeze_embeddings(model):
    for param in model.bert.embeddings.parameters():
        param.requires_grad = True


def unfreeze_layer(model, layer_idx):
    for param in model.bert.encoder.layer[layer_idx].parameters():
        param.requires_grad = True


def get_optimizer_params(model, base_lr=2e-5, lr_decay=0.95):
    layers = [model.bert.embeddings] + list(model.bert.encoder.layer)
    optimizer_grouped_parameters = []

    for i, layer in enumerate(layers):
        lr = base_lr * (lr_decay ** (len(layers) - i - 1))
        optimizer_grouped_parameters.append({
            "params": layer.parameters(),
            "lr": lr,
            "weight_decay": 0.01,
        })

    optimizer_grouped_parameters.append({
        "params": list(model.qa_outputs.parameters()) + list(model.no_answer_classifier.parameters()),
        "lr": base_lr,
        "weight_decay": 0.01,
    })

    return optimizer_grouped_parameters


def postprocess_qa_predictions(examples, features, raw_predictions, tokenizer, n_best_size=20, max_answer_length=30, null_score_diff_threshold=0.0,):
    all_start_logits, all_end_logits, all_no_answer_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        if "example_id" not in feature:
            continue
        example_id = feature["example_id"]
        if example_id not in example_id_to_index:
            continue
        features_per_example[example_id_to_index[example_id]].append(i)

    predictions = []

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        if not feature_indices:
            predictions.append({"id": example["id"], "prediction_text": "", "no_answer_probability": 1.0})
            continue

        min_null_score = None
        valid_answers = []
        context = example["context"]

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            no_answer_logit = all_no_answer_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index] + no_answer_logit
            if min_null_score is None or feature_null_score < min_null_score:
                min_null_score = float(feature_null_score)

            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                            start_index >= len(offsets)
                            or end_index >= len(offsets)
                            or offsets[start_index] is None
                            or offsets[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    start_char = offsets[start_index][0]
                    end_char = offsets[end_index][1]
                    valid_answers.append({
                        "score": float(start_logits[start_index] + end_logits[end_index]),
                        "text": context[start_char:end_char],
                    })

        if valid_answers:
            best_answer = max(valid_answers, key=lambda x: x["score"])
            answer = best_answer["text"]
            score = best_answer["score"]
        else:
            answer = ""
            score = 0.0

        no_answer_prob = 1.0 if not answer else 0.0
        if min_null_score is not None and min_null_score > score - null_score_diff_threshold:
            answer = ""
            no_answer_prob = float(torch.sigmoid(torch.tensor(min_null_score)))

        predictions.append({"id": example["id"], "prediction_text": answer, "no_answer_probability": no_answer_prob})

    return predictions


def compute_metrics(p, original_val, val_dataset, tokenizer):
    squad_metric = evaluate.load("squad_v2")
    start_logits, end_logits, no_answer_logits = p.predictions
    predictions = postprocess_qa_predictions(
        examples=original_val,
        features=val_dataset,
        raw_predictions=(start_logits, end_logits, no_answer_logits),
        tokenizer=tokenizer,
    )
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in original_val]

    squad_scores = squad_metric.compute(predictions=predictions, references=references)
    no_answer_acc = np.mean([
        (pred["prediction_text"] == "") == (len(ref["answers"]["text"]) == 0)
        for pred, ref in zip(predictions, references)
    ])

    return {"eval_exact": squad_scores["exact"], "eval_f1": squad_scores["f1"],
            "eval_no_answer_accuracy": no_answer_acc}


def compute_additional_metrics(predictions, references):
    y_true = [len(r["answers"]["text"]) > 0 for r in references]
    y_pred = [p["prediction_text"] != "" for p in predictions]

    tp = sum(yt and yp for yt, yp in zip(y_true, y_pred))
    tn = sum((not yt) and (not yp) for yt, yp in zip(y_true, y_pred))
    fp = sum((not yt) and yp for yt, yp in zip(y_true, y_pred))
    fn = sum(yt and (not yp) for yt, yp in zip(y_true, y_pred))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    return {
        "precision_answerable": precision,
        "recall_answerable": recall,
        "no_answer_confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
    }




