import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
WEIGHTS_DIR = ROOT / "models" / "weights"
RESULTS_DIR = ROOT / "results"

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = WEIGHTS_DIR / "random_forest_proper.joblib"
RESULT_PATH = RESULTS_DIR / "random_forest_proper_results.json"

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

FPR_LIMIT = 0.005
RECALL_TARGET = 0.80
RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATASETS
# ============================================================

print("\nLoading datasets...", flush=True)

train_df = pd.read_csv(DATA_DIR / "train_data.csv")
validation_df = pd.read_csv(DATA_DIR / "validation_data.csv")
test_df = pd.read_csv(DATA_DIR / "test_data.csv")

for name, df in [
    ("Training", train_df),
    ("Validation", validation_df),
    ("Test", test_df),
]:
    required = FEATURES + ["Class"]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"{name} dataset is missing columns: {sorted(missing)}")

    if df["Class"].isna().any():
        raise ValueError(f"{name} dataset contains missing labels.")

    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError(f"{name} dataset has unexpected labels.")

    print(
        f"{name}: {len(df):,} transactions | "
        f"Fraud: {int(df['Class'].sum()):,} | "
        f"Legitimate: {int((df['Class'] == 0).sum()):,}",
        flush=True,
    )

X_train = train_df[FEATURES]
y_train = train_df["Class"].astype(int)

X_val = validation_df[FEATURES]
y_val = validation_df["Class"].astype(int)

X_test = test_df[FEATURES]
y_test = test_df["Class"].astype(int)

if any(y.nunique() != 2 for y in [y_train, y_val, y_test]):
    raise ValueError("All datasets must contain both classes.")


# ============================================================
# 3. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...", flush=True)

model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

start_time = time.perf_counter()
model.fit(X_train, y_train)
training_seconds = time.perf_counter() - start_time

print(f"Training completed in {training_seconds:.2f} seconds.", flush=True)


# ============================================================
# 4. VALIDATION-BASED THRESHOLD SELECTION
#    Optimized using sorting and cumulative counts
# ============================================================

print("\nSelecting threshold using validation data...", flush=True)

val_probabilities = model.predict_proba(X_val)[:, 1]
val_labels = y_val.to_numpy()

order = np.argsort(-val_probabilities, kind="stable")

sorted_probabilities = val_probabilities[order]
sorted_labels = val_labels[order]

total_fraud = int(np.sum(sorted_labels == 1))
total_legitimate = int(np.sum(sorted_labels == 0))

# Last index of each group of identical probabilities.
change_positions = np.r_[
    np.flatnonzero(
        sorted_probabilities[:-1] != sorted_probabilities[1:]
    ),
    len(sorted_probabilities) - 1,
]

cumulative_tp = np.cumsum(sorted_labels == 1)
cumulative_fp = np.cumsum(sorted_labels == 0)

tp = np.r_[0, cumulative_tp[change_positions]]
fp = np.r_[0, cumulative_fp[change_positions]]

# First threshold flags no transactions.
thresholds = np.r_[
    np.nextafter(sorted_probabilities[0], np.inf),
    sorted_probabilities[change_positions],
]

fn = total_fraud - tp
tn = total_legitimate - fp

precision = np.divide(
    tp,
    tp + fp,
    out=np.zeros(len(tp), dtype=float),
    where=(tp + fp) > 0,
)

recall = tp / total_fraud

f1 = np.divide(
    2 * precision * recall,
    precision + recall,
    out=np.zeros(len(tp), dtype=float),
    where=(precision + recall) > 0,
)

fpr = fp / total_legitimate

eligible = np.flatnonzero(fpr <= FPR_LIMIT)

if len(eligible) == 0:
    selection_status = "fpr_limit_not_met"

    best_fpr = np.min(fpr)
    choices = np.flatnonzero(fpr == best_fpr)

    best_index = max(
        choices,
        key=lambda i: (
            recall[i],
            precision[i],
            thresholds[i],
        ),
    )

else:
    meets_recall = eligible[recall[eligible] >= RECALL_TARGET]

    if len(meets_recall) > 0:
        selection_status = "recall_target_and_fpr_limit_met"

        best_index = max(
            meets_recall,
            key=lambda i: (
                precision[i],
                f1[i],
                thresholds[i],
            ),
        )

    else:
        selection_status = "recall_target_not_met"

        best_index = max(
            eligible,
            key=lambda i: (
                recall[i],
                precision[i],
                thresholds[i],
            ),
        )

threshold = float(thresholds[best_index])

selected = {
    "threshold": threshold,
    "precision": float(precision[best_index]),
    "recall": float(recall[best_index]),
    "f1": float(f1[best_index]),
    "fpr": float(fpr[best_index]),
    "tp": int(tp[best_index]),
    "fp": int(fp[best_index]),
    "tn": int(tn[best_index]),
    "fn": int(fn[best_index]),
}

print(f"Selection status: {selection_status}", flush=True)
print(f"Selected threshold: {threshold:.10f}", flush=True)
print(f"Validation precision: {selected['precision']:.4f}", flush=True)
print(f"Validation recall:    {selected['recall']:.4f}", flush=True)
print(f"Validation F1-score:  {selected['f1']:.4f}", flush=True)
print(f"Validation FPR:       {selected['fpr']:.6%}", flush=True)


# ============================================================
# 5. SAVE MODEL AND THRESHOLD BEFORE TEST EVALUATION
# ============================================================

model_bundle = {
    "model": model,
    "features": FEATURES,
    "threshold": threshold,
    "threshold_selection_status": selection_status,
    "threshold_selection_method": "validation_only",
}

joblib.dump(model_bundle, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}", flush=True)


# ============================================================
# 6. FINAL TEST EVALUATION
#    Threshold is frozen; do not tune it on test data
# ============================================================

print("\nEvaluating on the held-out test dataset...", flush=True)

test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= threshold).astype(int)

tn_test, fp_test, fn_test, tp_test = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1],
).ravel()

test_precision = precision_score(
    y_test, test_predictions, zero_division=0
)

test_recall = recall_score(
    y_test, test_predictions, zero_division=0
)

test_f1 = f1_score(
    y_test, test_predictions, zero_division=0
)

test_fpr = (
    fp_test / (fp_test + tn_test)
    if (fp_test + tn_test) > 0
    else 0.0
)

test_roc_auc = roc_auc_score(y_test, test_probabilities)

test_ap = average_precision_score(
    y_test, test_probabilities
)


# ============================================================
# 7. SAVE FINAL RESULTS
# ============================================================

results = {
    "model": "Random Forest",
    "evaluation_type": "chronological_train_validation_test",
    "features": FEATURES,
    "random_state": RANDOM_STATE,
    "model_parameters": {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 2,
        "class_weight": "balanced_subsample",
    },
    "dataset_counts": {
        "train_transactions": int(len(train_df)),
        "train_fraud": int(y_train.sum()),
        "validation_transactions": int(len(validation_df)),
        "validation_fraud": int(y_val.sum()),
        "test_transactions": int(len(test_df)),
        "test_fraud": int(y_test.sum()),
    },
    "threshold_selection": {
        "method": "validation_only",
        "fpr_limit": FPR_LIMIT,
        "recall_target": RECALL_TARGET,
        "status": selection_status,
        "threshold": threshold,
        "validation_metrics": selected,
    },
    "test_metrics": {
        "transactions": int(len(y_test)),
        "fraud_cases": int(y_test.sum()),
        "legitimate_cases": int((y_test == 0).sum()),
        "threshold": threshold,
        "precision": float(test_precision),
        "recall": float(test_recall),
        "f1_score": float(test_f1),
        "false_positive_rate": float(test_fpr),
        "roc_auc": float(test_roc_auc),
        "average_precision": float(test_ap),
        "true_positives": int(tp_test),
        "false_positives": int(fp_test),
        "true_negatives": int(tn_test),
        "false_negatives": int(fn_test),
        "flagged_transactions": int(test_predictions.sum()),
    },
    "training_seconds": float(training_seconds),
    "test_threshold_selected_before_test_evaluation": True,
}

with open(RESULT_PATH, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print(f"Results saved to: {RESULT_PATH}", flush=True)


# ============================================================
# 8. DISPLAY FINAL SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("RANDOM FOREST — FINAL TEST RESULTS")
print("=" * 55)

print(f"Test transactions: {len(y_test):,}")
print(f"Actual fraud cases: {int(y_test.sum()):,}")
print(f"Selected threshold: {threshold:.10f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1-score: {test_f1:.4f}")
print(f"False Positive Rate: {test_fpr:.6%}")
print(f"ROC-AUC: {test_roc_auc:.4f}")
print(f"Average Precision: {test_ap:.4f}")

print(
    f"TP: {tp_test} | FP: {fp_test} | "
    f"TN: {tn_test} | FN: {fn_test}"
)

print(f"Flagged transactions: {int(test_predictions.sum())}")
print(f"Training time: {training_seconds:.2f} seconds")
print(f"Model: {MODEL_PATH}")
print(f"Results: {RESULT_PATH}")
print("=" * 55)