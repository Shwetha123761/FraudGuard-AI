
import os

# Limit numerical-library threads before importing NumPy or scikit-learn.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True, parents=True)

TARGET_RECALL = 0.80
MAX_FPR = 0.005


def evaluate(y, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    recall = recall_score(y, pred, zero_division=0)
    precision = precision_score(y, pred, zero_division=0)
    specificity = tn / (tn + fp) if tn + fp else 0.0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(fp / (tn + fp)),
        "balanced_accuracy": float((recall + specificity) / 2),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    train = pd.read_csv(DATA / "train_data.csv")
    test = pd.read_csv(DATA / "test_data.csv")

    features = [
        c for c in train.columns
        if c.startswith("V") or c == "Amount"
    ]

    # Preserve chronology: train on the earlier portion and validate later.
    split = int(len(train) * 0.8)
    fit = train.iloc[:split]
    val = train.iloc[split:]

    X_fit = fit[features]
    y_fit = fit["Class"].astype(int)
    X_val = val[features]
    y_val = val["Class"].astype(int)
    X_test = test[features]
    y_test = test["Class"].astype(int)

    print("Training rows:", len(fit), "| Fraud:", int(y_fit.sum()))
    print("Validation rows:", len(val), "| Fraud:", int(y_val.sum()))
    print("Test rows:", len(test), "| Fraud:", int(y_test.sum()))

    output = {
        "target_recall": TARGET_RECALL,
        "maximum_validation_fpr": MAX_FPR,
        "models": {},
    }

    for name, weight in [
        ("balanced", "balanced"),
        ("unweighted", None),
    ]:
        print(f"\nTraining {name} Logistic Regression...", flush=True)

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight=weight,
                max_iter=1000,
                solver="liblinear",
                random_state=42,
            ),
        )
        model.fit(X_fit, y_fit)

        val_scores = model.predict_proba(X_val)[:, 1]

        # A small candidate set based on validation score quantiles.
        thresholds = np.unique(
            np.quantile(val_scores, np.linspace(0, 1, 301))
        )

        val_results = [
            evaluate(y_val.to_numpy(), val_scores, t)
            for t in thresholds
        ]

        eligible = [
            r for r in val_results
            if r["recall"] >= TARGET_RECALL
            and r["false_positive_rate"] <= MAX_FPR
        ]

        if eligible:
            chosen = max(
                eligible,
                key=lambda r: (
                    r["precision"], r["recall"], r["threshold"]
                ),
            )
            status = "both_validation_targets_met"
        else:
            eligible_fpr = [
                r for r in val_results
                if r["false_positive_rate"] <= MAX_FPR
            ]

            if eligible_fpr:
                chosen = max(
                    eligible_fpr,
                    key=lambda r: (
                        r["recall"], r["precision"], r["threshold"]
                    ),
                )
                status = "recall_target_not_met"
            else:
                chosen = min(
                    val_results,
                    key=lambda r: (
                        r["false_positive_rate"], -r["recall"]
                    ),
                )
                status = "fpr_target_not_met"

        print("Validation status:", status, flush=True)
        print("Validation metrics:", json.dumps(chosen), flush=True)

        test_scores = model.predict_proba(X_test)[:, 1]
        test_metrics = evaluate(
            y_test.to_numpy(), test_scores, chosen["threshold"]
        )
        test_metrics["roc_auc"] = float(
            roc_auc_score(y_test, test_scores)
        )
        test_metrics["pr_auc_average_precision"] = float(
            average_precision_score(y_test, test_scores)
        )

        print("Held-out test metrics:", json.dumps(test_metrics), flush=True)

        output["models"][name] = {
            "class_weight": weight,
            "validation_status": status,
            "validation_metrics": chosen,
            "test_metrics": test_metrics,
        }

    output_path = RESULTS / "false_positive_experiment.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\nSaved results to:", output_path)


if __name__ == "__main__":
    main()

