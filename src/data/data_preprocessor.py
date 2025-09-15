import os
from typing import Optional, Tuple

from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


class QAPreprocessor:

    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 384, stride: int = 128,
                 data_dir: str = "./data"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length
        self.stride = stride
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def _create_answerable_mask(self, dataset: Dataset) -> list[bool]:
        return [bool(example["answers"]["text"] and len(example["answers"]["text"][0]) > 0)
                for example in dataset]

    def _split_dataset(self, dataset: Dataset, val_size: float, answerable_mask: list[bool]) -> Tuple[
        Dataset, Optional[Dataset]]:
        if val_size <= 0:
            return dataset, None

        indices = range(len(dataset))
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_size,
            random_state=42,
            stratify=answerable_mask
        )
        return dataset.select(train_indices), dataset.select(val_indices)

    def _tokenize_examples(self, examples) -> dict:
        tokenized = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized["overflow_to_sample_mapping"]
        offset_mapping = tokenized["offset_mapping"]
        cls_index = self.tokenizer.cls_token_id

        start_positions = []
        end_positions = []
        is_impossible = []
        example_ids = []

        for i, offsets in enumerate(offset_mapping):
            sample_idx = sample_mapping[i]
            answer = examples["answers"][sample_idx]
            input_ids = tokenized["input_ids"][i]
            cls_pos = input_ids.index(cls_index)

            if not answer["text"]:
                start_positions.append(cls_pos)
                end_positions.append(cls_pos)
                is_impossible.append(1.0)
            else:
                start_char = answer["answer_start"][0]
                end_char = start_char + len(answer["text"][0])
                token_start_index = None
                token_end_index = None

                for idx, (start, end) in enumerate(offsets):
                    if start <= start_char < end:
                        token_start_index = idx
                    if start < end_char <= end:
                        token_end_index = idx

                if token_start_index is not None and token_end_index is not None:
                    start_positions.append(token_start_index)
                    end_positions.append(token_end_index)
                    is_impossible.append(0.0)
                else:
                    start_positions.append(cls_pos)
                    end_positions.append(cls_pos)
                    is_impossible.append(1.0)

            example_ids.append(f"{sample_idx}_{i}")

        tokenized.update({
            "start_positions": start_positions,
            "end_positions": end_positions,
            "is_impossible": is_impossible,
            "example_id": example_ids
        })
        return tokenized

    def _tokenize_dataset(self, dataset: Dataset) -> Dataset:
        return dataset.map(self._tokenize_examples, batched=True, remove_columns=dataset.column_names)

    def prepare_datasets(self, dataset_name: str = "squad_v2", val_size: float = 0.1) -> Tuple[
        DatasetDict, DatasetDict]:
        dataset = load_dataset(dataset_name)
        train_data = dataset["train"]
        test_data = dataset["validation"]

        answerable_mask = self._create_answerable_mask(train_data)
        train_split, val_split = self._split_dataset(train_data, val_size, answerable_mask)

        train_tokenized = self._tokenize_dataset(train_split)
        val_tokenized = self._tokenize_dataset(val_split) if val_split else None
        test_tokenized = self._tokenize_dataset(test_data)

        train_valid = DatasetDict({
            "train": train_tokenized,
            "validation": val_tokenized if val_tokenized else train_tokenized
        })
        test = DatasetDict({"test": test_tokenized})

        train_valid.save_to_disk(os.path.join(self.data_dir, "train_valid"))
        test.save_to_disk(os.path.join(self.data_dir, "test"))

        val_size = len(val_tokenized) if val_tokenized else 0
        print(f"Train ({len(train_tokenized)}), Validation ({val_size}), Test ({len(test_tokenized)})")

        return train_valid, test

    def load_datasets(self) -> Tuple[DatasetDict, DatasetDict]:
        return (
            load_from_disk(os.path.join(self.data_dir, "train_valid")),
            load_from_disk(os.path.join(self.data_dir, "test"))
        )


if __name__ == "__main__":
    preprocessor = QAPreprocessor(data_dir="/Users/arynaartsiukevich/PycharmProjects/qa_assistant/src/input/data")
    train_valid, test = preprocessor.prepare_datasets(val_size=0.1)
