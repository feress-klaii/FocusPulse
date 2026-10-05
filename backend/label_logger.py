"""
FocusPulse — Sprint 2: Manual labeling tool.

Run this alongside your normal work. Press a key to log your current
state; it writes a timestamped row to labels.csv. Later, feature
extraction joins this against sensor_log.csv by timestamp.

Controls:
  f  -> mark FOCUSED  (starts a "focused" interval from now)
  d  -> mark DISTRACTED (starts a "distracted" interval from now)
  q  -> quit

Usage: python label_logger.py
"""

import csv
import os
from datetime import datetime, timezone

LABELS_FILE = "labels.csv"
FIELDNAMES = ["timestamp_utc", "label"]


def ensure_file():
    if not os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def log_label(label: str):
    now = datetime.now(timezone.utc).isoformat()
    with open(LABELS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow({"timestamp_utc": now, "label": label})
    print(f"[{now}] -> {label}")


def main():
    ensure_file()
    print("FocusPulse labeling tool. Press 'f' = focused, 'd' = distracted, 'q' = quit, then Enter.")
    while True:
        key = input("> ").strip().lower()
        if key == "f":
            log_label("focused")
        elif key == "d":
            log_label("distracted")
        elif key == "q":
            print("Session ended.")
            break
        else:
            print("Unrecognized input. Use f / d / q.")


if __name__ == "__main__":
    main()
