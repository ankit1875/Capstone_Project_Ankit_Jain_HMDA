"""
Fairness auditing utilities for RQ2 and RQ3.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from scipy.stats import chi2_contingency

def _drop_groups(df: pd.DataFrame, group_col: str, drop_values: Optional[List[str]]) -> pd.DataFrame:
    out = df.copy()
    out[group_col] = out[group_col].astype(str)
    if drop_values:
        out = out[~out[group_col].isin(set(map(str, drop_values)))]
    return out

def rq2_outcome_audit(df: pd.DataFrame, group_col: str, y_true_col: str = "target", y_prob_col: str = "Prob",
                     threshold: float = 0.5, drop_values: Optional[List[str]] = None, min_n: int = 1000) -> Tuple[pd.DataFrame, float]:
    d = df[[group_col, y_true_col, y_prob_col]].dropna().copy()
    d = _drop_groups(d, group_col, drop_values)
    d["pred"] = (d[y_prob_col] >= threshold).astype(int)

    g = d.groupby(group_col).agg(
        n=(y_true_col, "size"),
        actual_rate=(y_true_col, "mean"),
        avg_pred_prob=(y_prob_col, "mean"),
        pred_rate=("pred", "mean"),
    ).reset_index().rename(columns={group_col: "group"})
    g = g[g["n"] >= min_n].copy()

    best = g["pred_rate"].max() if len(g) else np.nan
    g["DIR_vs_best"] = g["pred_rate"] / best if best and best > 0 else np.nan
    g["gap_prob_minus_actual"] = g["avg_pred_prob"] - g["actual_rate"]

    ct = pd.crosstab(d[group_col], d["pred"])
    _, p, _, _ = chi2_contingency(ct)

    return g.sort_values("pred_rate", ascending=False).reset_index(drop=True), float(p)

def rq3_error_rate_audit(df: pd.DataFrame, group_col: str, y_true_col: str = "target", y_prob_col: str = "Prob",
                        threshold: float = 0.5, drop_values: Optional[List[str]] = None, min_n: int = 1000) -> Tuple[pd.DataFrame, float, str]:
    d = df[[group_col, y_true_col, y_prob_col]].dropna().copy()
    d = _drop_groups(d, group_col, drop_values)
    d["pred"] = (d[y_prob_col] >= threshold).astype(int)

    rows = []
    for grp, sub in d.groupby(group_col):
        if len(sub) < min_n:
            continue
        y = sub[y_true_col].astype(int).values
        p = sub["pred"].astype(int).values

        TP = int(((p == 1) & (y == 1)).sum())
        FN = int(((p == 0) & (y == 1)).sum())
        FP = int(((p == 1) & (y == 0)).sum())
        TN = int(((p == 0) & (y == 0)).sum())

        TPR = TP / (TP + FN) if (TP + FN) else np.nan
        FNR = FN / (TP + FN) if (TP + FN) else np.nan
        FPR = FP / (FP + TN) if (FP + TN) else np.nan
        TNR = TN / (FP + TN) if (FP + TN) else np.nan

        rows.append({"group": str(grp), "n": len(sub), "TP": TP, "FN": FN, "FP": FP, "TN": TN,
                     "TPR": TPR, "FNR": FNR, "FPR": FPR, "TNR": TNR})

    out = pd.DataFrame(rows)
    ref = out.loc[out["TPR"].idxmax()]
    ref_group = str(ref["group"])
    out["EOD_vs_ref"] = out["TPR"] - float(ref["TPR"])
    out["FNR_gap_vs_ref"] = out["FNR"] - float(ref["FNR"])

    ct = out.set_index("group")[["TP", "FN"]]
    _, p, _, _ = chi2_contingency(ct)

    cols = ["group","n","TPR","FNR","FPR","TNR","EOD_vs_ref","FNR_gap_vs_ref"]
    return out.sort_values("TPR", ascending=False)[cols].reset_index(drop=True), float(p), ref_group
