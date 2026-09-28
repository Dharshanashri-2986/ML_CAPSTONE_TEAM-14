"""Temporal feature engineering (Rubric B3).

For each key sensor, per engine (never across engines):
  <sensor>_rolling_mean : mean of the last ROLLING_WINDOW cycles (min_periods=1)
  <sensor>_change       : s_t - s_{t-1}  (0 on an engine's first cycle)
Both only look backwards in time, so no future information leaks in.
"""
ENGINEERED_SENSORS = ["sensor_2", "sensor_11", "sensor_14"]
ROLLING_WINDOW = 5


def engineer_features(df, sensors=None, window=ROLLING_WINDOW):
    sensors = sensors or ENGINEERED_SENSORS
    out = df.sort_values(["unit_id", "cycle"]).copy()
    g = out.groupby("unit_id")
    for s in sensors:
        out[f"{s}_rolling_mean"] = (
            g[s].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        out[f"{s}_change"] = g[s].diff().fillna(0.0)
    return out.reset_index(drop=True)
