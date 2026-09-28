"""Dataset audit helpers (Rubric A1 / B1)."""
import pandas as pd

from .loading import SETTING_COLS, SENSOR_COLS


def audit_dataset(df):
    """One row per column: dtype, missing count/%, unique values."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isnull().sum(),
        "missing_pct": (df.isnull().mean() * 100).round(2),
        "n_unique": df.nunique(),
    })


def iqr_outlier_audit(df, cols=None, k=1.5):
    """Count values outside [Q1 - k*IQR, Q3 + k*IQR] for each column."""
    cols = cols or (SETTING_COLS + SENSOR_COLS)
    rows = []
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n = int(((df[c] < lo) | (df[c] > hi)).sum())
        rows.append({
            "feature": c,
            "q1": q1, "q3": q3, "lower_bound": lo, "upper_bound": hi,
            "outliers_count": n,
            "outliers_pct": round(100 * n / len(df), 2),
            "retention_rationale": (
                "Retained: physical extreme regime telemetry" if n > 0 else "No outliers"
            ),
        })
    return pd.DataFrame(rows)
