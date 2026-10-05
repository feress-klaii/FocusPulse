"""
FocusPulse backend — Sprint 1 receiver.
Accepts sensor payloads from the ESP32 and logs them to a CSV file.
Run: python -m uvicorn mainiot:app --host 0.0.0.0 --port 8000 --reload
"""

import csv
import os
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="FocusPulse Ingest API")

LOG_FILE = "sensor_log.csv"
FIELDNAMES = [
    "received_at",
    "timestamp_ms",
    "accel_x", "accel_y", "accel_z", "accel_magnitude",
    "gyro_x", "gyro_y", "gyro_z",
    "light_raw", "air_quality_raw", "sound_raw",
]


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
