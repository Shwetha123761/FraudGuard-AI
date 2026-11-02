
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train_data.csv"
TEST_PATH = DATA_DIR / "test_data.csv"


def metrics_at_threshold(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_true, predictions, zero_division=0
    )
    recall = recall_score(
        y_true, predictions, zero_division=0
    )
    f1 = f1_score(y_true, predictions, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    # Harmonic mean of fraud recall and legitimate-transaction specificity.
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (recall + specificity) / 2

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "balanced_accuracy": float(balanced_accuracy),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Run python data/prepare_data.py first."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    feature_columns = [
        col for col in train_df.columns
        if col.startswith("V") or col == "Amount"
    ]

    # Chronological validation split: earlier rows train the model;
    # later rows select the operating threshold.
    split_index = int(len(train_df) * 0.8)
    fit_df = train_df.iloc[:split_index]
    validation_df = train_df.iloc[split_index:]

    X_fit = fit_df[feature_columns]
    y_fit = fit_df["Class"].astype(int)

    X_val = validation_df[feature_columns]
    y_val = validation_df["Class"].astype(int)

    X_test = test_df[feature_columns]
    y_test = test_df["Class"].astype(int)

    print("Training cases:", len(X_fit))
    print("Validation cases:", len(X_val))
    print("Final test cases:", len(X_test))
    print("Training fraud:", int(y_fit.sum()))
    print("Validation fraud:", int(y_val.sum()))
    print("Final test fraud:", int(y_test.sum()))

    if y_fit.nunique() < 2 or y_val.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "Training, validation, and test partitions must each contain "
            "both classes. Revisit the chronological split."
        )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    )

    print("\nTraining Logistic Regression...")
    model.fit(X_fit, y_fit)

    validation_scores = model.predict_proba(X_val)[:, 1]

    # Search validation thresholds. Candidate values come from
    # the observed validation probabilities.
    candidates = np.unique(
        np.quantile(validation_scores, np.linspace(0, 1, 501))
    )
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], candidates)))

    validation_results = [
        metrics_at_threshold(y_val.to_numpy(), validation_scores, t)
        for t in candidates
    ]

    # Choose the threshold with the highest balanced accuracy.
    # Break ties by preferring higher precision, then higher threshold.
    best = max(
        validation_results,
        key=lambda row: (
            row["balanced_accuracy"],
            row["precision"],
            row["threshold"],
        ),
    )
    selected_threshold = best["threshold"]

    print("\nValidation-selected threshold:")
    print(json.dumps(best, indent=2))

    print("\nOther validation operating points:")
    for threshold in [0.5, selected_threshold]:
        row = metrics_at_threshold(
            y_val.to_numpy(), validation_scores, threshold
        )
        print(json.dumps(row, indent=2))

    # The test set is used only after selecting the threshold.
    test_scores = model.predict_proba(X_test)[:, 1]
    test_metrics = metrics_at_threshold(
        y_test.to_numpy(), test_scores, selected_threshold
    )

    test_metrics["roc_auc"] = float(
        roc_auc_score(y_test, test_scores)
    )
    test_metrics["pr_auc_average_precision"] = float(
        average_precision_score(y_test, test_scores)
    )

    print("\nFinal held-out test results:")
    print(json.dumps(test_metrics, indent=2))

    output = {
        "model": "Logistic Regression",
        "features": feature_columns,
        "threshold_selection": "maximum validation balanced accuracy",
        "selected_threshold": selected_threshold,
        "validation_metrics_at_selected_threshold": best,
        "test_metrics_at_selected_threshold": test_metrics,
        "note": (
            "Threshold selected using validation data. "
            "Final test results are reported separately."
        ),
    }

    output_path = RESULTS_DIR / "threshold_tuning_results.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()

