"""
Performance metrics utilities for RQ1.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p")
    pos = df["y"].sum()
    neg = (1 - df["y"]).sum()
    df["cdf_pos"] = df["y"].cumsum() / max(pos, 1)
    df["cdf_neg"] = (1 - df["y"]).cumsum() / max(neg, 1)
    return float((df["cdf_pos"] - df["cdf_neg"]).abs().max())

def lift_at_top_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.1) -> float:
    n = len(y_true)
    top_n = max(int(np.ceil(k * n)), 1)
    idx = np.argsort(y_prob)[::-1][:top_n]
    top_rate = y_true[idx].mean()
    base_rate = y_true.mean()
    return float(top_rate / base_rate) if base_rate > 0 else np.nan

def calibration_by_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5) -> Tuple[List[float], List[float]]:
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    agg = df.groupby("bin").agg(avg_actual=("y", "mean"), avg_pred=("p", "mean"))
    return agg["avg_actual"].tolist(), agg["avg_pred"].tolist()

def compute_rq1_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    auc = roc_auc_score(y_true, y_prob)
    ks = ks_statistic(y_true, y_prob) * 100.0
    return {
        "ROC_AUC": float(auc),
        "Gini": float(2 * auc - 1),
        "KS": float(ks),
        "Lift": float(lift_at_top_k(y_true, y_prob, k=0.1)),
        "Brier_Score_Loss": float(brier_score_loss(y_true, y_prob)),
        "Log_Loss": float(log_loss(y_true, y_prob, eps=1e-15)),
    }
