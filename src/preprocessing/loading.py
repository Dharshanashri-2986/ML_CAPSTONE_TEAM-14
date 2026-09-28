"""Raw file loading and ground-truth RUL construction."""
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root = two levels above this file (src/preprocessing/loading.py)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "raw"

SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_RAW_COLS = ["unit_id", "cycle"] + SETTING_COLS + SENSOR_COLS

# Sensor meanings from the C-MAPSS documentation (Saxena et al., 2008)
SENSOR_DESCRIPTIONS = {
    "sensor_1": "Fan inlet temperature (T2)",
    "sensor_2": "LPC outlet temperature (T24)",
    "sensor_3": "HPC outlet temperature (T30)",
    "sensor_4": "LPT outlet temperature (T50)",
    "sensor_5": "Fan inlet pressure (P2)",
    "sensor_6": "Bypass-duct pressure (P15)",
    "sensor_7": "HPC outlet pressure (P30)",
    "sensor_8": "Physical fan speed (Nf)",
    "sensor_9": "Physical core speed (Nc)",
    "sensor_10": "Engine pressure ratio (epr)",
    "sensor_11": "HPC outlet static pressure (Ps30)",
    "sensor_12": "Fuel flow / Ps30 ratio (phi)",
    "sensor_13": "Corrected fan speed (NRf)",
    "sensor_14": "Corrected core speed (NRc)",
    "sensor_15": "Bypass ratio (BPR)",
    "sensor_16": "Burner fuel-air ratio (farB)",
    "sensor_17": "Bleed enthalpy (htBleed)",
    "sensor_18": "Demanded fan speed (Nf_dmd)",
    "sensor_19": "Demanded corrected fan speed (PCNfR_dmd)",
    "sensor_20": "HPT coolant bleed (W31)",
    "sensor_21": "LPT coolant bleed (W32)",
}


def _read_space_separated(path, names=None):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, engine="python")


def load_raw(dataset_id="FD004", data_dir=None):
    """Load the train, test and RUL ground-truth files for one C-MAPSS subset.

    Returns
    -------
    train_raw, test_raw : DataFrame with columns ALL_RAW_COLS
    rul_raw : DataFrame with a single column 'RUL' (one row per test engine)
    """
    folder = Path(data_dir) if data_dir else DATA_DIR / f"NASA_{dataset_id}"
    train_raw = _read_space_separated(folder / f"train_{dataset_id}.txt", ALL_RAW_COLS)
    test_raw = _read_space_separated(folder / f"test_{dataset_id}.txt", ALL_RAW_COLS)
    rul_raw = _read_space_separated(folder / f"RUL_{dataset_id}.txt", ["RUL"])
    return train_raw, test_raw, rul_raw


def compute_rul(train_raw, test_raw, rul_raw):
    """Attach a per-cycle Remaining Useful Life target.

    Train engines run to failure, so RUL = last_cycle - cycle.
    Test engines are truncated; NASA provides the true RUL at the last
    observed cycle, so RUL = true_RUL_at_end + (last_cycle - cycle).
    """
    train_df = train_raw.copy()
    last = train_df.groupby("unit_id")["cycle"].transform("max")
    train_df["RUL"] = (last - train_df["cycle"]).astype(int)

    test_df = test_raw.copy()
    end_rul = pd.Series(rul_raw["RUL"].values, index=np.sort(test_df["unit_id"].unique()))
    last_t = test_df.groupby("unit_id")["cycle"].transform("max")
    test_df["RUL"] = (test_df["unit_id"].map(end_rul) + (last_t - test_df["cycle"])).astype(int)
    return train_df, test_df
