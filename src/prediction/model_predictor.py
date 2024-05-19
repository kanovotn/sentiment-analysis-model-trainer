from datasets import load_dataset
import evaluate
from evaluate import evaluator
import pandas as pd


class ModelPredictor:
    """
        Class evaluating selected list of models. It returns accuracy metric for each model on a given dataset.
    """
    def __init__(self, models_ids, dataset_source):
        self.models = models_ids
        self.task_evaluator = evaluator("sentiment-analysis")
        self.data = load_dataset(dataset_source, split="test")

    def evaluate_models(self):
        results = []
        for model in self.models:
            model_results = self.task_evaluator.compute(
                model_or_pipeline=model,
                data=self.data,
                label_mapping={"NEGATIVE": 0, "POSITIVE": 1}
            )
            results.append(model_results)
        return pd.DataFrame(results, index=self.models)


