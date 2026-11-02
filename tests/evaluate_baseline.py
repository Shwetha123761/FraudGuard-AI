from pathlib import Path
import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train_data.csv"
TEST_PATH = DATA_DIR / "test_data.csv"


def main():
    print("=" * 60)
    print("FraudGuard AI - Logistic Regression Baseline")
    print("=" * 60)

    # Load the chronological split created by data/prepare_data.py
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Training or testing data not found. "
            "Run: python data/prepare_data.py"
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    target = "Class"

    if target not in train_df.columns or target not in test_df.columns:
        raise ValueError("Both datasets must contain the 'Class' target column.")

    # Exclude the label and raw time from the initial baseline.
    # V1-V28 and Amount are used as predictors.
    feature_columns = [
        col for col in train_df.columns
        if col.startswith("V") or col == "Amount"
    ]

    if not feature_columns:
        raise ValueError("No V1-V28 or Amount feature columns were found.")

    X_train = train_df[feature_columns]
    y_train = train_df[target].astype(int)

    X_test = test_df[feature_columns]
    y_test = test_df[target].astype(int)

    print(f"Training transactions: {len(X_train):,}")
    print(f"Testing transactions:  {len(X_test):,}")
    print(f"Training fraud cases:  {int(y_train.sum()):,}")
    print(f"Testing fraud cases:   {int(y_test.sum()):,}")
    print(f"Features: {len(feature_columns)}")

    # StandardScaler is fitted only on training data.
    # This avoids fitting the scaler using test-set statistics.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    )

    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)

    # Evaluate using continuous scores and the default 0.5 threshold.
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= 0.5).astype(int)

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    roc_auc = float(roc_auc_score(y_test, y_scores))
    pr_auc = float(average_precision_score(y_test, y_scores))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=true, columns=predicted):")
    print(cm)
    print(f"\nROC-AUC: {roc_auc:.6f}")
    print(f"PR-AUC (Average Precision): {pr_auc:.6f}")

    results = {
        "model": "Logistic Regression",
        "features": feature_columns,
        "threshold": 0.5,
        "train_transactions": int(len(X_train)),
        "test_transactions": int(len(X_test)),
        "train_fraud_cases": int(y_train.sum()),
        "test_fraud_cases": int(y_test.sum()),
        "roc_auc": roc_auc,
        "pr_auc_average_precision": pr_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    output_path = RESULTS_DIR / "logistic_regression_baseline.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nResults saved to: {output_path}")
    print("\nNote: These results describe this baseline and split,")
    print("not the performance of the complete FraudGuard hybrid system.")


if __name__ == "__main__":
    main()


