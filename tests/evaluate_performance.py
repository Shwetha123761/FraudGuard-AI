
# tests/evaluate_performance.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from sklearn.model_selection import train_test_split


def evaluate_system():
    print("[*] Running FraudGuard AI diagnostic evaluation...")

    # Temporary synthetic labels.
    # Replace these with actual test labels later.
    rng = np.random.default_rng(42)

    y_true = np.array([0] * 950 + [1] * 50)

    # Split the data to demonstrate reproducible test setup.
    # These are still synthetic labels, not a real dataset.
    _, y_test = train_test_split(
        y_true,
        test_size=0.30,
        random_state=42,
        stratify=y_true
    )

    # Generate random scores independently of the labels.
    # These scores do NOT represent a trained model.
    y_scores = rng.uniform(0, 1, size=len(y_test))

    # Example threshold only.
    threshold = 0.5
    y_pred = (y_scores >= threshold).astype(int)

    print("\n" + "=" * 50)
    print("DIAGNOSTIC CLASSIFICATION REPORT")
    print("=" * 50)

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            zero_division=0
        )
    )

    print(f"ROC-AUC: {roc_auc_score(y_test, y_scores):.4f}")
    print(
        "PR-AUC (Average Precision): "
        f"{average_precision_score(y_test, y_scores):.4f}"
    )

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    output_dir = (
        Path(__file__).resolve().parent.parent
        / "database"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"]
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("FraudGuard AI - Diagnostic Confusion Matrix")
    plt.tight_layout()

    output_path = output_dir / "confusion_matrix.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nConfusion matrix saved to: {output_path}")
    print("\nNOTE: These are synthetic diagnostic results.")
    print("They must NOT be reported as model performance.")


if __name__ == "__main__":
    evaluate_system()

