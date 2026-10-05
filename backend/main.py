"""
FocusPulse backend — Sprint 1 receiver.
Accepts sensor payloads from the ESP32 and logs them to a CSV file.
Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import csv
import os
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="FocusPulse Ingest API")

# Allow the dashboard (opened as a local file or a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG_FILE = "sensor_log.csv"
MODEL_FILE = "focus_model.joblib"
PREDICT_WINDOW_SECONDS = 15  # must match feature_extraction.py's WINDOW_SECONDS

FIELDNAMES = [
    "received_at",
    "timestamp_ms",
    "accel_x", "accel_y", "accel_z", "accel_magnitude",
    "gyro_x", "gyro_y", "gyro_z",
    "light_raw", "air_quality_raw", "sound_raw",
]

_model = None  # lazy-loaded so the server still starts if no model exists yet


def get_model():
    global _model
    if _model is None and os.path.exists(MODEL_FILE):
        import joblib
        _model = joblib.load(MODEL_FILE)
    return _model


def compute_window_features(window: pd.DataFrame) -> dict:
    """Same feature logic as feature_extraction.py — keep these two in sync."""
    accel_mag = window["accel_magnitude"].values
    light = window["light_raw"].values
    air = window["air_quality_raw"].values
    sound = window["sound_raw"].values

    motion_variance = float(np.var(accel_mag)) if len(accel_mag) > 1 else 0.0
    jerk = np.diff(accel_mag) if len(accel_mag) > 1 else np.array([0.0])
    jerk_mean_abs = float(np.mean(np.abs(jerk)))
    stillness_ratio = float(np.mean(np.abs(accel_mag - np.mean(accel_mag)) < 0.3)) if len(accel_mag) > 1 else 1.0

    if len(air) > 1:
        x = np.arange(len(air))
        air_slope = float(np.polyfit(x, air, 1)[0])
    else:
        air_slope = 0.0

    light_stability = float(np.std(light)) if len(light) > 1 else 0.0
    noise_level = float(np.mean(sound))

    return {
        "motion_variance": motion_variance,
        "jerk_mean_abs": jerk_mean_abs,
        "stillness_ratio": stillness_ratio,
        "air_quality_slope": air_slope,
        "light_stability": light_stability,
        "noise_level": noise_level,
    }


class SensorPayload(BaseModel):
    timestamp_ms: int
    accel_x: float
    accel_y: float
    accel_z: float
    accel_magnitude: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    light_raw: int
    air_quality_raw: int
    sound_raw: int


def ensure_log_file():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


@app.on_event("startup")
def startup():
    ensure_log_file()


@app.post("/ingest")
def ingest(payload: SensorPayload):
    row = payload.dict()
    row["received_at"] = datetime.now(timezone.utc).isoformat()

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)

    return {"status": "ok", "received_at": row["received_at"]}


@app.get("/health")
def health():
    return {"status": "running"}


@app.get("/latest")
def latest(n: int = 10):
    """Return the last n logged readings — handy for sanity-checking during Sprint 1."""
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r") as f:
        rows = list(csv.DictReader(f))
    return rows[-n:]


@app.get("/predict")
def predict():
    """
    Compute features over the most recent window of sensor data and
    return a focus prediction. Returns a clear status if the model
    isn't trained yet or there isn't enough recent data.
    """
    model = get_model()
    if model is None:
        return {"status": "no_model", "message": "focus_model.joblib not found — run train_model.py first."}

    if not os.path.exists(LOG_FILE):
        return {"status": "no_data", "message": "No sensor data logged yet."}

    df = pd.read_csv(LOG_FILE, parse_dates=["received_at"])
    if df.empty:
        return {"status": "no_data", "message": "No sensor data logged yet."}

    cutoff = df["received_at"].max() - timedelta(seconds=PREDICT_WINDOW_SECONDS)
    window = df[df["received_at"] >= cutoff]

    if len(window) < 2:
        return {"status": "insufficient_data", "message": "Not enough recent readings to predict."}

    features = compute_window_features(window)
    feature_order = [
        "motion_variance", "jerk_mean_abs", "stillness_ratio",
        "air_quality_slope", "light_stability", "noise_level",
    ]
    X = [[features[f] for f in feature_order]]

    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    confidence = float(max(proba))

    return {
        "status": "ok",
        "label": prediction,
        "confidence": confidence,
        "probabilities": dict(zip(classes, [float(p) for p in proba])),
        "window_seconds": PREDICT_WINDOW_SECONDS,
        "as_of": df["received_at"].max().isoformat(),
    }
