"""Load a model bundle saved by ``ml.classical.training`` and run predictions."""
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd

Records = Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]


class ModelPredictor:
    """Thin wrapper around a saved pipeline that validates inputs and decodes labels."""

    def __init__(self, model_path: str):
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.metadata = bundle["metadata"]
        self.task = self.metadata["task"]
        self.classes = self.metadata.get("classes")
        self.feature_cols = self.metadata["numeric_cols"] + self.metadata["categorical_cols"]

    def _to_frame(self, records: Records) -> pd.DataFrame:
        if isinstance(records, dict):
            records = [records]
        df = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)

        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        return df[self.feature_cols]

    def _decode(self, preds: np.ndarray) -> list:
        if self.classes is None:
            return preds.tolist()
        return [self.classes[int(p)] for p in preds]

    def predict(self, records: Records) -> list:
        """Predict labels (classification) or values (regression)."""
        return self._decode(self.pipeline.predict(self._to_frame(records)))

    def predict_proba(self, records: Records) -> List[Dict[str, float]]:
        """Per-class probabilities, keyed by the original class label."""
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification models")

        proba = self.pipeline.predict_proba(self._to_frame(records))
        labels = self.classes or self.pipeline.classes_.tolist()
        return [
            {str(label): round(float(p), 4) for label, p in zip(labels, row)}
            for row in proba
        ]

    def predict_batch(self, df: pd.DataFrame, batch_size: int = 10_000) -> pd.DataFrame:
        """Score a large DataFrame in chunks and append a ``prediction`` column."""
        preds = []
        for start in range(0, len(df), batch_size):
            preds.extend(self.predict(df.iloc[start:start + batch_size]))
        return df.assign(prediction=preds)


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    predictor = ModelPredictor("ml/artifacts/breast_cancer_lgbm.joblib")
    sample = load_breast_cancer(as_frame=True).frame.drop(columns="target").head(3)

    print(predictor.predict(sample))
    print(predictor.predict_proba(sample))
