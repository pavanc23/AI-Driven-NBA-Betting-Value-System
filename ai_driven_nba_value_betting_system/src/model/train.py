from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


FEATURE_COLS: List[str] = [
    "home_win_rolling",
    "away_win_rolling",
    "home_pd_rolling",
    "away_pd_rolling",
    "home_rest_days",
    "away_rest_days",
]


def load_features(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def train_model(
    features_path: Path,
    model_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[LogisticRegression, dict]:
    df = load_features(features_path)
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].mean())
    y = df["target_home_win"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    if len(pd.unique(y_test)) < 2:
        # Tiny sample test split with a single class; avoid metric errors.
        metrics = {
            "note": "test set contains a single class; add more games for full metrics",
            "brier": brier_score_loss(y_test, y_pred_proba),
        }
    else:
        metrics = {
            "brier": brier_score_loss(y_test, y_pred_proba),
            "logloss": log_loss(y_test, y_pred_proba),
            "auc": roc_auc_score(y_test, y_pred_proba),
        }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model, metrics


def load_model(model_path: Path) -> LogisticRegression:
    return joblib.load(model_path)

