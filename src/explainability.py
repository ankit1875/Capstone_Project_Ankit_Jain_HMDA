"""
Explainability utilities for RQ4 (SHAP group-wise proxy detection).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional

def compute_groupwise_shap_table(df: pd.DataFrame, X_all: pd.DataFrame, shap_values: np.ndarray, group_col: str,
                                feat_cols: List[str], prob: Optional[np.ndarray] = None,
                                drop_values: Optional[List[str]] = None, min_n: int = 1000) -> pd.DataFrame:
    tmp = df[[group_col]].copy()
    tmp[group_col] = tmp[group_col].astype(str)
    if drop_values:
        tmp = tmp[~tmp[group_col].isin(set(map(str, drop_values)))]
    tmp = tmp.dropna()

    rows = []
    for grp, idx in tmp.groupby(group_col).groups.items():
        idx = list(idx)
        if len(idx) < min_n:
            continue
        pos = X_all.index.get_indexer(idx)  # aligns index -> row positions
        Sg = shap_values[pos, :]

        row = {"group": str(grp), "n": len(idx)}
        if prob is not None:
            row["avg_pred_prob"] = float(np.mean(prob[pos]))

        for j, f in enumerate(feat_cols):
            row[f"imp_{f}"] = float(np.mean(np.abs(Sg[:, j])))
            row[f"dir_{f}"] = float(np.mean(Sg[:, j]))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)

def top_proxy_candidates(groupwise_tbl: pd.DataFrame, feat_cols: List[str], top_k: int = 5) -> pd.Series:
    ranges = {}
    for f in feat_cols:
        ranges[f] = float(groupwise_tbl[f"imp_{f}"].max() - groupwise_tbl[f"imp_{f}"].min())
    return pd.Series(ranges).sort_values(ascending=False).head(top_k)
