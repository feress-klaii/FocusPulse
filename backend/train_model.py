"""
FocusPulse — Sprint 2: Model training.

Reads features.csv (output of feature_extraction.py), trains a
classifier to predict focused vs distracted, validates it, and
saves the trained model to disk for later use by the dashboard/API.

Usage: python train_model.py
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

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
    df = pd.read_csv(FEATURES_FILE)
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
    print(classification_report(y_test, y_pred))

    # Feature importance — useful for your report's "why" section
    importances = sorted(
        zip(FEATURE_COLUMNS, clf.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\nFeature importance:")
    for name, score in importances:
        print(f"  {name}: {score:.3f}")

    joblib.dump(clf, MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()
