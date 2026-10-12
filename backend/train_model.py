"""
FocusPulse — Sprint 2: Model training.

Reads features.csv (output of feature_extraction.py), trains a
classifier to predict focused vs distracted, validates it, and
saves the trained model to disk for later use by the dashboard/API.

Usage: python train_model.py
       python train_model.py --sensors synthetic_sensor_log.csv --labels synthetic_labels.csv
"""

import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from feature_extraction import extract_features, load_data

FEATURES_FILE = "features.csv"
MODEL_FILE = "focus_model.joblib"

FEATURE_COLUMNS = [
    "motion_variance",
    "jerk_mean_abs",
    "stillness_ratio",
    "air_quality_slope",
    "light_stability",
    "noise_level",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+",
                        help="one or more features CSVs to train on (e.g. synthetic + real)")
    parser.add_argument("--sensors", help="raw sensor log CSV; use with --labels to extract features directly")
    parser.add_argument("--labels", help="labels CSV to pair with --sensors")
    parser.add_argument("--model", default=MODEL_FILE, help=f"output model file (default {MODEL_FILE})")
    args = parser.parse_args()

    if bool(args.sensors) != bool(args.labels):
        parser.error("--sensors and --labels must be provided together")
    if args.sensors and args.features:
        parser.error("use either --features or --sensors/--labels, not both")

    if args.sensors:
        sensors, labels = load_data(args.sensors, args.labels)
        if sensors.empty:
            parser.error(f"{args.sensors} is empty")
        df = extract_features(sensors, labels)
    else:
        feature_files = args.features or [FEATURES_FILE]
        df = pd.concat([pd.read_csv(f) for f in feature_files], ignore_index=True)

    df = df[df["label"].isin(["focused", "distracted"])]  # drop unlabeled windows

    if len(df) < 10:
        print(f"Only {len(df)} labeled windows found — collect more labeled data before training.")
        print("Aim for at least a few dozen windows per class for a meaningful first model.")
        return

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Feature importance — useful for your report's "why" section
    importances = sorted(
        zip(FEATURE_COLUMNS, clf.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\nFeature importance:")
    for name, score in importances:
        print(f"  {name}: {score:.3f}")

    joblib.dump(clf, args.model)
    print(f"\nModel saved to {args.model}")


if __name__ == "__main__":
    main()