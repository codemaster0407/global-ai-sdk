from typing import List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Column type inference
# ---------------------------------------------------------------------------
def infer_column_types(
    df: pd.DataFrame,
    target: Optional[str] = None,
    exclude: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Split the feature columns of ``df`` into numeric and categorical lists.

    ``target`` and any ``exclude`` columns (ids, timestamps, ...) are dropped
    before inference.  Booleans are treated as categorical.
    """
    drop = set(exclude or [])
    if target:
        drop.add(target)
    features = df.drop(columns=[c for c in drop if c in df.columns])

    numeric_cols = features.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in features.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------
def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    scale: bool = True,
) -> ColumnTransformer:
    """Build a ``ColumnTransformer`` that imputes, scales and one-hot encodes.

    Tree models (XGBoost / LightGBM) don't need scaling, so pass
    ``scale=False`` for them to keep the raw feature values.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    categorical_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", Pipeline(categorical_steps), categorical_cols))

    # DataFrame output keeps column names, so tree models see real feature names
    return ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=False
    ).set_output(transform="pandas")


def build_passthrough_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """Select the feature columns unchanged: numeric first, then categorical.

    For models that impute, encode and scale internally (TabPFN). The fixed
    column order lets the caller point the model at the categorical columns
    by position: indices ``len(numeric_cols)`` onwards.
    """
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", "passthrough", numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", "passthrough", categorical_cols))
    return ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=False
    ).set_output(transform="pandas")
