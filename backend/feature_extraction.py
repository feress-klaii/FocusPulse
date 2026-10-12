"""
FocusPulse — Sprint 2: Feature extraction.

Reads sensor_log.csv (raw ESP32 readings) and labels.csv (your manual
focused/distracted tags), groups sensor readings into rolling windows,
computes engineered features per window, and attaches the label that
was active during that window.

Output: features.csv — one row per window, ready for model training.

Usage: python feature_extraction.py
"""

import argparse
import numpy as np
import pandas as pd

SENSOR_LOG = "sensor_log.csv"
LABELS_LOG = "labels.csv"
OUTPUT_FILE = "features.csv"

WINDOW_SECONDS = 15  # size of each rolling feature window


def load_data(sensor_path, labels_path):
    sensors = pd.read_csv(sensor_path)
    labels = pd.read_csv(labels_path)
    # tolerate mixed timestamp formats (with / without microseconds)
    sensors["received_at"] = pd.to_datetime(sensors["received_at"], utc=True, format="ISO8601")
    labels["timestamp_utc"] = pd.to_datetime(labels["timestamp_utc"], utc=True, format="ISO8601")
    sensors = sensors.sort_values("received_at").reset_index(drop=True)
    labels = labels.sort_values("timestamp_utc").reset_index(drop=True)
    return sensors, labels


def label_at(labels: pd.DataFrame, t) -> str:
    """Return whichever label was most recently set before timestamp t."""
    past = labels[labels["timestamp_utc"] <= t]
    if past.empty:
        return "unlabeled"
    return past.iloc[-1]["label"]


def compute_window_features(window: pd.DataFrame) -> dict:
    accel_mag = window["accel_magnitude"].values
    light = window["light_raw"].values
    air = window["air_quality_raw"].values
    sound = window["sound_raw"].values

    # Motion features
    motion_variance = float(np.var(accel_mag)) if len(accel_mag) > 1 else 0.0
    jerk = np.diff(accel_mag) if len(accel_mag) > 1 else np.array([0.0])
    jerk_mean_abs = float(np.mean(np.abs(jerk)))
    stillness_ratio = float(np.mean(np.abs(accel_mag - np.mean(accel_mag)) < 0.3)) if len(accel_mag) > 1 else 1.0

    # Air quality drift (slope over the window — positive = rising)
    if len(air) > 1:
        x = np.arange(len(air))
        air_slope = float(np.polyfit(x, air, 1)[0])
    else:
        air_slope = 0.0

    # Light stability (lower std = more stable)
    light_stability = float(np.std(light)) if len(light) > 1 else 0.0

    # Noise level (mean raw sound reading)
    noise_level = float(np.mean(sound))

    return {
        "motion_variance": motion_variance,
        "jerk_mean_abs": jerk_mean_abs,
        "stillness_ratio": stillness_ratio,
        "air_quality_slope": air_slope,
        "light_stability": light_stability,
        "noise_level": noise_level,
    }


def extract_features(sensors: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if sensors.empty:
        return pd.DataFrame()

    start = sensors["received_at"].min()
    end = sensors["received_at"].max()
    window_delta = pd.Timedelta(seconds=WINDOW_SECONDS)

    rows = []
    t = start
    while t < end:
        window = sensors[(sensors["received_at"] >= t) & (sensors["received_at"] < t + window_delta)]
        if len(window) > 0:
            features = compute_window_features(window)
            features["window_start"] = t
            features["label"] = label_at(labels, t) if not labels.empty else "unlabeled"
            rows.append(features)
        t += window_delta

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensors", default=SENSOR_LOG, help="raw sensor log CSV (default sensor_log.csv)")
    parser.add_argument("--labels", default=LABELS_LOG, help="labels CSV (default labels.csv)")
    parser.add_argument("--out", default=OUTPUT_FILE, help="output features CSV (default features.csv)")
    args = parser.parse_args()

    sensors, labels = load_data(args.sensors, args.labels)
    if sensors.empty:
        print(f"{args.sensors} is empty — collect some data first.")
        return

    out = extract_features(sensors, labels)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} feature windows to {args.out}")
    if not labels.empty:
        print(out["label"].value_counts())


if __name__ == "__main__":
    main()