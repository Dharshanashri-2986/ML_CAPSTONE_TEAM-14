"""Data loading, auditing, feature engineering and leakage-free split/scale
for the NASA C-MAPSS FD004 turbofan dataset.

All notebooks import from here so that every track trains on exactly the
same preprocessed data.
"""
from .loading import (
    DATA_DIR, SETTING_COLS, SENSOR_COLS, ALL_RAW_COLS, SENSOR_DESCRIPTIONS,
    load_raw, compute_rul,
)
from .audit import audit_dataset, iqr_outlier_audit
from .features import engineer_features, ENGINEERED_SENSORS, ROLLING_WINDOW
from .split_scale import engine_train_val_split, scale_features
from .dataset import get_dataset, RISK_THRESHOLD

__all__ = [
    "DATA_DIR", "SETTING_COLS", "SENSOR_COLS", "ALL_RAW_COLS", "SENSOR_DESCRIPTIONS",
    "load_raw", "compute_rul", "audit_dataset", "iqr_outlier_audit",
    "engineer_features", "ENGINEERED_SENSORS", "ROLLING_WINDOW",
    "engine_train_val_split", "scale_features", "get_dataset", "RISK_THRESHOLD",
]
