"""
Data preparation utilities for HMDA project.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable

def add_missing_flag_and_impute_median(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[f"{col}_missing"] = out[col].isna().astype(int)
    out[col] = pd.to_numeric(out[col], errors="coerce")
    out[col] = out[col].fillna(out[col].median())
    return out

def impute_many_with_flags(df: pd.DataFrame, numeric_cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in numeric_cols:
        out = add_missing_flag_and_impute_median(out, c)
    return out

def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["applicant_income_000s"] = pd.to_numeric(out["applicant_income_000s"], errors="coerce")
    out["loan_amount_000s"] = pd.to_numeric(out["loan_amount_000s"], errors="coerce")

    out["log_income"] = np.log1p(out["applicant_income_000s"])
    out["log_loan_amount"] = np.log1p(out["loan_amount_000s"])
    out["log_loan_to_income"] = out["log_loan_amount"] - out["log_income"]
    return out

def ensure_bool_dummies_are_int(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "bool":
            out[c] = out[c].astype(int)
    return out
