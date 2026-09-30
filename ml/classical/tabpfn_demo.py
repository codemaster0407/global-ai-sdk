"""Compare TabPFN with LightGBM on the same train/test split.

    python -m ml.classical.tabpfn_demo            # uses TABPFN_MODEL_VERSION, else v3.5 (needs TABPFN_TOKEN)
    TABPFN_MODEL_VERSION=v2 python -m ml.classical.tabpfn_demo   # ungated v2 weights
"""
import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from ml.classical.training import train_model


def compare_models(
    df: pd.DataFrame,
    target: str,
    model_names: List[str] = ("tabpfn", "lightgbm"),
    task: str = "classification",
    params: Optional[Dict[str, Dict[str, Any]]] = None,
    **train_kwargs,
) -> pd.DataFrame:
    """Train each model on the same split and return one row of test metrics + seconds per model."""
    rows = []
    for name in model_names:
        start = time.perf_counter()
        result = train_model(df, target, model_name=name, task=task,
                             params=(params or {}).get(name), **train_kwargs)
        rows.append({"model": name, **result.metrics, "seconds": round(time.perf_counter() - start, 1)})
    return pd.DataFrame(rows).set_index("model")


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True).frame
    version = os.getenv("TABPFN_MODEL_VERSION")
    tabpfn_params = {"model_version": version} if version else {}
    print(compare_models(data, "target", params={"tabpfn": tabpfn_params}))
