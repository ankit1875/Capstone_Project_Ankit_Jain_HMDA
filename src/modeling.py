"""
Model training utilities (Logistic Regression + CatBoost).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostClassifier

@dataclass
class TrainedModel:
    model: object
    feature_columns: List[str]
    cat_columns: Optional[List[str]] = None
    threshold: float = 0.5

def train_logistic(df_train: pd.DataFrame, y_train: pd.Series, numeric_cols: List[str], cat_cols: List[str]) -> TrainedModel:
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(max_iter=5000)
    pipe = Pipeline([("preprocess", pre), ("clf", clf)])
    pipe.fit(df_train[numeric_cols + cat_cols], y_train)
    return TrainedModel(model=pipe, feature_columns=numeric_cols + cat_cols, cat_columns=cat_cols)

def train_catboost(X_train: pd.DataFrame, y_train: pd.Series, cat_features: Optional[List[str]] = None,
                   params: Optional[Dict] = None, random_state: int = 42, eval_metric: str = "AUC") -> TrainedModel:
    params = dict(params or {})
    # avoid duplicate kwargs
    params.pop("eval_metric", None)
    params.pop("random_seed", None)

    model = CatBoostClassifier(**params, eval_metric=eval_metric, random_seed=random_state, verbose=0)
    model.fit(X_train, y_train, cat_features=cat_features)
    return TrainedModel(model=model, feature_columns=list(X_train.columns), cat_columns=cat_features)

def predict_proba(trained: TrainedModel, X: pd.DataFrame) -> np.ndarray:
    return trained.model.predict_proba(X[trained.feature_columns])[:, 1]

def predict_label(y_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (y_prob >= threshold).astype(int)
