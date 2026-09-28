"""One-call access to the fully preprocessed, scaled dataset for every track."""
from .loading import load_raw, compute_rul
from .features import engineer_features
from .split_scale import engine_train_val_split, scale_features

RISK_THRESHOLD = 30  # cycles; RUL <= 30 -> "Critical Risk" (class 1)
_EXCLUDE = {"unit_id", "cycle", "RUL", "Risk_Class"}


def get_dataset(task="regression", dataset_id="FD004", val_size=0.2, random_state=42):
    """Return a dict with scaled X/y arrays for train, val and the NASA test set.

    task : "regression"     -> y = RUL (cycles)
           "classification" -> y = 1 if RUL <= RISK_THRESHOLD else 0
           "clustering"     -> same X; y = risk label, for post-hoc interpretation only
    """
    train_raw, test_raw, rul_raw = load_raw(dataset_id)
    train_df, test_df = compute_rul(train_raw, test_raw, rul_raw)
    train_feat = engineer_features(train_df)
    test_feat = engineer_features(test_df)
    feature_names = [c for c in train_feat.columns if c not in _EXCLUDE]

    tr, va = engine_train_val_split(train_feat, val_size=val_size, random_state=random_state)
    scaler, X_train, X_val, X_test = scale_features(
        tr[feature_names].values, va[feature_names].values, test_feat[feature_names].values
    )

    def target(frame):
        if task == "regression":
            return frame["RUL"].values
        return (frame["RUL"] <= RISK_THRESHOLD).astype(int).values

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": target(tr), "y_val": target(va), "y_test": target(test_feat),
        "feature_names": feature_names, "scaler": scaler,
        "train_df": tr, "val_df": va, "test_df": test_feat,
    }
