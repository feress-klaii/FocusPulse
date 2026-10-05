"""
FocusPulse — Synthetic data generator.

Generates feature rows (same shape as feature_extraction.py's output)
with configurable per-class distributions, so you can bootstrap model
training without waiting on long manual labeling sessions.

Honesty note for your report: a model trained purely on synthetic
data demonstrates the pipeline works end-to-end, but doesn't prove
it generalizes to real focus patterns. Mixing in real labeled data
(even a small amount) makes results more credible. State this as a
known limitation if you use synthetic data in your final results.

Usage:
    python generate_synthetic_data.py                  # generate fresh synthetic set
    python generate_synthetic_data.py --append          # append to existing features.csv
    python generate_synthetic_data.py --n 150 --noise 0.3
"""

import argparse
import numpy as np
import pandas as pd

OUTPUT_FILE = "features.csv"

# ---- CONFIG: calibrated from self-reported focus patterns (survey, Oct 2026) ----
# Each feature: (focused_mean, distracted_mean, shared_std)
# A bigger gap between means + smaller std = easier/cleaner separation.
# A smaller gap + bigger std = more realistic overlap, harder classification.
#
# Personalization notes:
# - Motion is INVERTED from the generic assumption: this user reports being
#   MORE still when distracted (e.g. phone scrolling) and MORE active when
#   genuinely focused (writing, typing, engaged movement).
# - Noise level is a WEAK signal for this user (reported comfortable with
#   silence, low background noise, AND moderate noise) — small mean gap,
#   large std, so it won't dominate feature importance the way it did
#   with the earlier manually-collected (Wokwi-interaction-biased) data.
# - Air quality / CO2 drift is a STRONG, confirmed signal — this user
#   reports noticeable drowsiness in stuffy/poorly ventilated rooms.
FEATURE_CONFIG = {
    "motion_variance":   (0.5, 0.08, 0.15),   # focused = MORE active, distracted = MORE still
    "jerk_mean_abs":     (0.35, 0.05, 0.12),  # same inversion
    "stillness_ratio":   (0.35, 0.85, 0.15),  # distracted = high stillness ratio (e.g. scrolling)
    "air_quality_slope": (0.4, 3.2, 1.0),     # strong signal — confirmed by self-report
    "light_stability":   (4.0, 10.0, 5.0),    # kept moderate — not confirmed by survey, weaker gap
    "noise_level":       (700, 950, 500),     # weak signal — small gap, large overlap
}


def generate_rows(n_per_class: int, noise_scale: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for label, mean_idx in [("focused", 0), ("distracted", 1)]:
        for _ in range(n_per_class):
            row = {}
            for feat, (focused_mean, distracted_mean, std) in FEATURE_CONFIG.items():
                mean = focused_mean if mean_idx == 0 else distracted_mean
                value = rng.normal(mean, std * noise_scale)
                # clip to realistic non-negative ranges for the ones that must be >= 0
                if feat in ("motion_variance", "jerk_mean_abs", "light_stability", "noise_level"):
                    value = max(0.0, value)
                if feat == "stillness_ratio":
                    value = min(1.0, max(0.0, value))
                row[feat] = round(float(value), 4)
            row["label"] = label
            row["window_start"] = "synthetic"
            rows.append(row)

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="samples per class (default 100)")
    parser.add_argument("--noise", type=float, default=1.0, help="noise multiplier — higher = more overlap/realism (default 1.0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--append", action="store_true", help="append to existing features.csv instead of overwriting")
    args = parser.parse_args()

    synthetic = generate_rows(args.n, args.noise, args.seed)

    if args.append:
        try:
            existing = pd.read_csv(OUTPUT_FILE)
            combined = pd.concat([existing, synthetic], ignore_index=True)
        except FileNotFoundError:
            combined = synthetic
    else:
        combined = synthetic

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {len(combined)} total rows to {OUTPUT_FILE} ({len(synthetic)} synthetic rows this run)")
    print(combined["label"].value_counts())


if __name__ == "__main__":
    main()
