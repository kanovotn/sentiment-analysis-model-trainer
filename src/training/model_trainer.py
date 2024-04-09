
from transformers import (AutoTokenizer, DataCollatorWithPadding, TFAutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

from datasets import load_dataset
import evaluate
import numpy as np


class ModelTrainer:
    def __init__(self, checkpoint, num_labels):
        id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        label2id = {"NEGATIVE": 0, "POSITIVE": 1}

        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(self.checkpoint, num_labels=num_labels,
                                                                          id2label=id2label, label2id=label2id)

    def _tokenize_function(self, example):
        return self.tokenizer(example['text'], truncation=True)

    def _compute_metrics(eval_preds):
        metric = evaluate.load("accuracy")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    def load_and_preprocess_dataset(self, dataset_name):
        raw_dataset_train = load_dataset(dataset_name, split="train")
        raw_dataset_validate = load_dataset(dataset_name, split="test")
        tokenized_dataset_train = raw_dataset_train.map(self._tokenize_function, batched=True)
        tokenized_dataset_validate = raw_dataset_validate.map(self._tokenize_function, batched=True)

        return tokenized_dataset_train, tokenized_dataset_validate


    def fine_tune_model(self, tokenized_dataset_train, tokenized_dataset_validate, dataset_name):
        training_args = TrainingArguments(
            output_dir=self.checkpoint + "-finetuned-" + dataset_name,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=True,
        )

        trainer = Trainer(
            self.model,
            training_args,
            train_dataset=tokenized_dataset_train,
            eval_dataset=tokenized_dataset_validate,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
        )

        trainer.train()
        trainer.push_to_hub()

        return trainer

    def save_model(self, trainer):
        trainer.push_to_hub()