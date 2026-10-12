"""
FocusPulse — Synthetic RAW sensor log generator.

Simulates what the ESP32 would have logged over several work sessions,
and writes it in exactly the same format as the real files:

    synthetic_sensor_log.csv   (same columns as sensor_log.csv)
    synthetic_labels.csv       (same columns as labels.csv)

Run them through the normal pipeline so features are computed by the
SAME code the live /predict endpoint uses (no scale mismatch):

    python generate_synthetic_sensor_log.py
    python train_model.py --sensors synthetic_sensor_log.csv --labels synthetic_labels.csv

Your real sensor_log.csv / labels.csv are never touched.

Behaviour is calibrated from the self-reported survey:
  - MOTION is inverted vs the generic assumption: more movement when
    genuinely focused (writing/typing), very still when distracted
    (phone scrolling). Each state also has an "off-pattern" mode
    (focused-but-reading = still, distracted-but-restless = moving),
    which creates realistic overlap between the classes.
  - AIR QUALITY drifts upward faster when distracted/drowsy (stuffy room):
    strong signal.
  - LIGHT stability: only a weak difference (not confirmed by survey).
  - NOISE: identical distribution in both states (survey: noise level
    doesn't predict focus), so it should carry ~no signal.

Honesty note for the report: this is simulated data encoding the
hypotheses above. It validates the pipeline and the hypothesis, not
real-world behaviour.
"""

import argparse
import csv
from datetime import datetime, timedelta, timezone

import numpy as np

SENSOR_OUT = "synthetic_sensor_log.csv"
LABELS_OUT = "synthetic_labels.csv"
G = 9.80665

FIELDNAMES = [
    "received_at", "timestamp_ms",
    "accel_x", "accel_y", "accel_z", "accel_magnitude",
    "gyro_x", "gyro_y", "gyro_z",
    "light_raw", "air_quality_raw", "sound_raw",
]

# ---- CONFIG ----------------------------------------------------------
# Probability that a block behaves "off-pattern" for its label.
OFF_PATTERN_P = {"focused": 0.25, "distracted": 0.20}

# accel   = median std of per-axis motion noise (m/s^2)
# drift   = mean air-quality ADC change per sample (5 s), drift_sd = its spread
# light   = median per-sample LDR noise (ADC units)
MODES = {
    ("focused", "typical"):    dict(accel=0.35, drift=1.0, drift_sd=2.0, light=8.0),   # actively working
    ("focused", "off"):        dict(accel=0.08, drift=1.5, drift_sd=2.0, light=8.0),   # reading / thinking
    ("distracted", "typical"): dict(accel=0.04, drift=8.0, drift_sd=4.0, light=12.0),  # scrolling, drowsy air
    ("distracted", "off"):     dict(accel=0.25, drift=5.0, drift_sd=4.0, light=12.0),  # restless
}
# ----------------------------------------------------------------------


def pick_mode(rng, label, overlap):
    off_p = min(0.5, OFF_PATTERN_P[label] * overlap)
    return "off" if rng.random() < off_p else "typical"


def simulate_session(rng, start, minutes, interval, overlap, sensor_writer, label_rows):
    n_samples = int(minutes * 60 / interval)

    # per-session environment
    tilt_x, tilt_y = rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)
    air = rng.uniform(1200, 2400)
    air_center = air
    light = rng.uniform(900, 2800)
    sound_base = rng.uniform(300, 1500)

    t = start
    i = 0
    label = rng.choice(["focused", "distracted"])  # random start; blocks then alternate
    first_block = True
    while i < n_samples:
        # new block (alternate states so classes stay balanced)
        if not first_block:
            label = "distracted" if label == "focused" else "focused"
        first_block = False
        mode = MODES[(label, pick_mode(rng, label, overlap))]
        sigma = rng.lognormal(np.log(mode["accel"]), 0.35)
        drift = rng.normal(mode["drift"], mode["drift_sd"])
        light_noise = rng.lognormal(np.log(mode["light"]), 0.3)
        block_len = int(rng.uniform(4, 12) * 60 / interval)

        label_rows.append({"timestamp_utc": t.isoformat(timespec="microseconds"), "label": label})

        for _ in range(min(block_len, n_samples - i)):
            # motion
            ax = tilt_x + rng.normal(0, sigma)
            ay = tilt_y + rng.normal(0, sigma)
            az = np.sqrt(max(G**2 - tilt_x**2 - tilt_y**2, 0.0)) + rng.normal(0, sigma)
            if rng.random() < 0.025:  # occasional gesture / pick-up burst, any state
                az += rng.uniform(1.0, 3.0) * rng.choice([-1, 1])
            mag = float(np.sqrt(ax**2 + ay**2 + az**2))
            gx, gy, gz = (rng.normal(0, sigma * 0.6) for _ in range(3))

            # air quality: continuous drift with gentle pull back to the session baseline
            air += drift + rng.normal(0, 4) + 0.01 * (air_center - air)
            air = float(np.clip(air, 0, 4095))

            # light: slow level with rare lamp/cloud shifts
            if rng.random() < 0.003:
                light += rng.choice([-1, 1]) * rng.uniform(60, 200)
            light = float(np.clip(light, 0, 4095))
            light_sample = int(np.clip(light + rng.normal(0, light_noise), 0, 4095))

            # sound: label-independent, occasional events
            sound = sound_base + rng.normal(0, 50)
            if rng.random() < 0.02:
                sound += rng.uniform(300, 1200)
            sound = int(np.clip(sound, 0, 4095))

            sensor_writer.writerow({
                "received_at": t.isoformat(timespec="microseconds"),
                "timestamp_ms": (i + 1) * int(interval * 1000) + 4,
                "accel_x": round(float(ax), 5), "accel_y": round(float(ay), 5),
                "accel_z": round(float(az), 5), "accel_magnitude": round(mag, 5),
                "gyro_x": round(float(gx), 5), "gyro_y": round(float(gy), 5),
                "gyro_z": round(float(gz), 5),
                "light_raw": light_sample, "air_quality_raw": int(air), "sound_raw": sound,
            })

            t += timedelta(seconds=interval + rng.uniform(-0.1, 0.3))  # real logs jitter slightly
            i += 1

    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", type=int, default=8, help="number of work sessions (default 8)")
    p.add_argument("--minutes", type=float, default=30, help="minutes per session (default 30)")
    p.add_argument("--interval", type=float, default=5, help="seconds between samples (match firmware, default 5)")
    p.add_argument("--overlap", type=float, default=1.0,
                   help="multiplier on off-pattern block probability; higher = more class overlap (default 1.0)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    start = datetime(2026, 9, 1, 8, 0, 0, tzinfo=timezone.utc)
    label_rows = []

    with open(SENSOR_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        t = start
        for _ in range(args.sessions):
            t = simulate_session(rng, t, args.minutes, args.interval, args.overlap, writer, label_rows)
            t += timedelta(minutes=45)  # gap between sessions

    with open(LABELS_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_utc", "label"])
        writer.writeheader()
        writer.writerows(label_rows)

    total_min = args.sessions * args.minutes
    print(f"Wrote {SENSOR_OUT} ({args.sessions} sessions x {args.minutes:g} min = {total_min:g} min of readings)")
    print(f"Wrote {LABELS_OUT} ({len(label_rows)} label changes)")


if __name__ == "__main__":
    main()
