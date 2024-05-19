from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, DataCollatorWithPadding, DistilBertForSequenceClassification, DistilBertConfig
from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np


class ModelSentiment:
    """
    A class for data preparation and fine-tuning selected pre-trained model
    """

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        configuration = DistilBertConfig.from_pretrained(
            self.checkpoint,
            num_labels=2,
            id2label={0: "NEGATIVE", 1: "POSITIVE"},
            label2id={"NEGATIVE": 0, "POSITIVE": 1}
        )

        self.model = DistilBertForSequenceClassification.from_pretrained(self.checkpoint, config=configuration)

    def _tokenize_function(self, example):
        return self.tokenizer(example['text'], max_length=512, truncation=True)

    def tokenize_dataset(self, dataset):
        tokenized_dataset = dataset.map(self._tokenize_function, batched=True)
        return tokenized_dataset

    def load_and_split_dataset(self, dataset_name):
        # Split ratios
        train_split = 0.6
        test_split = 0.2

        # Load the dataset
        dataset_train = load_dataset(dataset_name, split="train")
        dataset_test = load_dataset(dataset_name, split="test")

        # Merge them and shuffle
        dataset_full = concatenate_datasets([dataset_train, dataset_test])

        # Shuffle the data with fixed seed to ensure the reproducibility of the dataset
        dataset_full = dataset_full.shuffle(seed=42).flatten_indices()

        # Calculate the number of samples for train, validate, and test
        total_samples = len(dataset_full)
        train_size = int(total_samples * train_split)
        test_size = int(total_samples * test_split)

        # Split the dataset
        dataset_train = dataset_full.select(range(train_size))
        dataset_test = dataset_full.select(range(train_size, train_size + test_size))
        dataset_validation = dataset_full.select(range(train_size + test_size, total_samples))

        return dataset_train, dataset_validation, dataset_test

    def fine_tune_model(self, tokenized_dataset_train, tokenized_dataset_validate):
        training_args = TrainingArguments(
            output_dir="distilbert-base-finetuned-sentiment",
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=3,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=True
        )

        trainer = Trainer(
            self.model,
            training_args,
            train_dataset=tokenized_dataset_train,
            eval_dataset=tokenized_dataset_validate,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        trainer.train()

        return trainer

    def save_model(self, trainer):
        trainer.push_to_hub()

    def compute_metrics(self, eval_preds):
        metric = evaluate.load("accuracy")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=1)
        return metric.compute(predictions=predictions, references=labels)
