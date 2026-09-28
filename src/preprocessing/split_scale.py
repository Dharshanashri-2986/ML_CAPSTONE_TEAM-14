"""Leakage-free engine-level splitting and train-only scaling (Rubric B2)."""
import numpy as np
from sklearn.preprocessing import StandardScaler


def engine_train_val_split(df, val_size=0.2, random_state=42):
    """Split whole engines (unit_id) into train / validation.

    An engine's trajectory is never divided, so no cycles from a validation
    engine are seen during training.
    """
    units = np.sort(df["unit_id"].unique())
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(units)
    n_val = int(len(units) * val_size)
    val_units = set(shuffled[:n_val])
    val_mask = df["unit_id"].isin(val_units)
    return df[~val_mask].reset_index(drop=True), df[val_mask].reset_index(drop=True)


def scale_features(X_train, *others):
    """Fit StandardScaler on X_train ONLY, then transform every other array."""
    scaler = StandardScaler()
    Xs = [scaler.fit_transform(X_train)] + [scaler.transform(X) for X in others]
    return (scaler, *Xs)
