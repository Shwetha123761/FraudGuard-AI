
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[1]

VALIDATION_PATH = ROOT / "results" / "scores" / "validation_scores.csv"
TEST_PATH = ROOT / "results" / "scores" / "test_scores.csv"

AE_RESULTS_PATH = ROOT / "results" / "autoencoder_results.json"
LR_PATH = ROOT / "models" / "weights" / "supervised_model.joblib"
RF_PATH = ROOT / "models" / "weights" / "random_forest_proper.joblib"
FUSION_CONFIG_PATH = ROOT / "models" / "weights" / "fusion_config.json"

OUTPUT_DIR = ROOT / "results" / "unified"

MODELS = {
    "AE": "AE",
    "LR": "LR",
    "RF": "RF",
    "AE_LR_Fusion": "AE_LR_Fusion",
}

FPR_TARGETS = [0.001, 0.005]
RECALL_TARGETS = [0.80, 0.90]

BOOTSTRAP_ITERATIONS = 2000
RANDOM_SEED = 42


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_saved_thresholds():
    ae_results = load_json(AE_RESULTS_PATH)
    fusion = load_json(FUSION_CONFIG_PATH)

    lr_bundle = joblib.load(LR_PATH)
    rf_bundle = joblib.load(RF_PATH)

    if fusion.get("selected_on") != "validation_data_only":
        raise ValueError("Fusion configuration is not validation-only.")

    if rf_bundle.get("threshold_selection_method") != "validation_only":
        raise ValueError("RF threshold was not selected on validation.")

    ae_threshold = float(
        ae_results["validation_metrics"]["threshold"]
    )
    lr_threshold = float(lr_bundle["threshold"])
    rf_threshold = float(rf_bundle["threshold"])
    fusion_threshold = float(fusion["threshold"])

    if not np.isclose(
        float(fusion["autoencoder_threshold"]),
        ae_threshold,
    ):
        raise ValueError("Fusion AE threshold mismatch.")

    if not np.isclose(
        float(fusion["supervised_model_threshold_reference"]),
        lr_threshold,
    ):
        raise ValueError("Fusion LR threshold mismatch.")

    return {
        "AE": ae_threshold,
        "LR": lr_threshold,
        "RF": rf_threshold,
        "AE_LR_Fusion": fusion_threshold,
    }


def load_scores(path, name):
    df = pd.read_csv(path)

    required = ["Time", "Class"] + list(MODELS.values())

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

    if df.empty:
        raise ValueError(f"{name} score file is empty.")

    if df[required].isna().any().any():
        raise ValueError(f"{name} contains missing values.")

    y = df["Class"].to_numpy(dtype=int)

    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"{name} has invalid class labels.")

    if len(np.unique(y)) != 2:
        raise ValueError(f"{name} must contain both classes.")

    for column in MODELS.values():
        values = df[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"{name} has non-finite values in {column}."
            )

    return df


def calculate_ranking_metrics(y, scores):
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(
            average_precision_score(y, scores)
        ),
    }


def stratified_bootstrap_ci(y, scores, iterations, rng):
    """
    Stratified bootstrap: independently resample legitimate
    and fraud cases, preserving their original class counts.
    """

    negative = scores[y == 0]
    positive = scores[y == 1]

    roc_values = []
    ap_values = []

    for _ in range(iterations):
        sampled_negative = rng.choice(
            negative,
            size=len(negative),
            replace=True,
        )

        sampled_positive = rng.choice(
            positive,
            size=len(positive),
            replace=True,
        )

        sampled_scores = np.concatenate(
            [sampled_negative, sampled_positive]
        )

        sampled_y = np.concatenate([
            np.zeros(len(sampled_negative), dtype=int),
            np.ones(len(sampled_positive), dtype=int),
        ])

        roc_values.append(
            roc_auc_score(sampled_y, sampled_scores)
        )

        ap_values.append(
            average_precision_score(sampled_y, sampled_scores)
        )

    return {
        "roc_auc_95_ci": [
            float(np.percentile(roc_values, 2.5)),
            float(np.percentile(roc_values, 97.5)),
        ],
        "average_precision_95_ci": [
            float(np.percentile(ap_values, 2.5)),
            float(np.percentile(ap_values, 97.5)),
        ],
        "bootstrap_iterations": iterations,
        "method": "stratified_percentile_bootstrap",
    }


def operating_metrics(y, scores, threshold):
    predicted = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predicted,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(y, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predicted, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y, predicted, zero_division=0)
        ),
        "false_positive_rate": float(
            fp / (fp + tn)
        ) if (fp + tn) else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_fpr_threshold(y, scores, target_fpr):
    fpr, tpr, thresholds = roc_curve(y, scores)

    eligible = np.flatnonzero(
        (fpr <= target_fpr) & np.isfinite(thresholds)
    )

    if len(eligible) == 0:
        raise ValueError(
            f"No finite threshold meets FPR target {target_fpr}."
        )

    # Maximize recall while satisfying the FPR constraint.
    best = max(
        eligible,
        key=lambda i: (tpr[i], thresholds[i]),
    )

    return float(thresholds[best])


def select_recall_threshold(y, scores, target_recall):
    fpr, tpr, thresholds = roc_curve(y, scores)

    eligible = np.flatnonzero(
        (tpr >= target_recall) & np.isfinite(thresholds)
    )

    if len(eligible) == 0:
        raise ValueError(
            f"No finite threshold meets recall target {target_recall}."
        )

    # Minimize FPR while satisfying the recall constraint.
    best = min(
        eligible,
        key=lambda i: (fpr[i], -thresholds[i]),
    )

    return float(thresholds[best])


def evaluate_partition(name, df, saved_thresholds, matched_thresholds):
    y = df["Class"].to_numpy(dtype=int)

    result = {
        "rows": int(len(df)),
        "fraud_cases": int(y.sum()),
        "legitimate_cases": int((y == 0).sum()),
        "models": {},
    }

    summary_rows = []

    rng = np.random.default_rng(RANDOM_SEED)

    for model_name, column in MODELS.items():
        scores = df[column].to_numpy(dtype=float)

        ranking = calculate_ranking_metrics(y, scores)

        ci = stratified_bootstrap_ci(
            y,
            scores,
            BOOTSTRAP_ITERATIONS,
            rng,
        )

        operating_points = {
            "saved_threshold": operating_metrics(
                y,
                scores,
                saved_thresholds[model_name],
            )
        }

        for point_name, threshold in matched_thresholds[
            model_name
        ].items():
            operating_points[point_name] = operating_metrics(
                y,
                scores,
                threshold,
            )

        result["models"][model_name] = {
            "score_column": column,
            "ranking_metrics": ranking,
            "confidence_intervals": ci,
            "operating_points": operating_points,
        }

        for point_name, metrics in operating_points.items():
            summary_rows.append({
                "partition": name,
                "model": model_name,
                "operating_point": point_name,
                **metrics,
                **ranking,
                "roc_auc_ci_lower": ci["roc_auc_95_ci"][0],
                "roc_auc_ci_upper": ci["roc_auc_95_ci"][1],
                "ap_ci_lower": ci["average_precision_95_ci"][0],
                "ap_ci_upper": ci["average_precision_95_ci"][1],
            })

    return result, summary_rows


def build_rolling_windows(name, df, window_count):
    """
    Descriptive chronological windows.
    Rows are sorted by Time and divided into equal-sized groups.
    """

    ordered = df.sort_values(
        "Time",
        kind="stable",
    ).reset_index(drop=True)

    windows = np.array_split(
        np.arange(len(ordered)),
        window_count,
    )

    results = []

    for index, positions in enumerate(windows, start=1):
        window = ordered.iloc[positions]
        y = window["Class"].to_numpy(dtype=int)

        entry = {
            "window": index,
            "rows": int(len(window)),
            "fraud_cases": int(y.sum()),
            "legitimate_cases": int((y == 0).sum()),
            "time_start": float(window["Time"].min()),
            "time_end": float(window["Time"].max()),
            "models": {},
        }

        for model_name, column in MODELS.items():
            scores = window[column].to_numpy(dtype=float)

            if len(np.unique(y)) == 2:
                entry["models"][model_name] = (
                    calculate_ranking_metrics(y, scores)
                )
            else:
                entry["models"][model_name] = {
                    "roc_auc": None,
                    "average_precision": None,
                    "note": "Only one class present in this window.",
                }

        results.append(entry)

    return results


def write_markdown_report(result, summary):
    lines = [
        "# FraudGuard AI â€” Unified Evaluation Report",
        "",
        "## Evaluation design",
        "",
        "- Saved model weights and thresholds were not modified.",
        "- Matched thresholds were selected using validation scores only.",
        "- Matched thresholds were applied unchanged to test scores.",
        "- Test data was not used for threshold selection or training.",
        "- Confidence intervals use stratified percentile bootstrap.",
        "- Rolling windows are descriptive chronological analyses.",
        "- GNN and Hybrid scores are excluded from this evaluation.",
        "",
        "## Dataset summary",
        "",
        "| Partition | Rows | Fraud | Legitimate |",
        "|---|---:|---:|---:|",
    ]

    for name in ["validation", "test"]:
        partition = result[name]
        lines.append(
            f"| {name} | {partition['rows']} | "
            f"{partition['fraud_cases']} | "
            f"{partition['legitimate_cases']} |"
        )

    lines.extend([
        "",
        "## Ranking and operating-point results",
        "",
        "| Partition | Model | Operating point | Threshold | "
        "Precision | Recall | F1 | FPR | ROC-AUC | AP |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for row in summary:
        lines.append(
            f"| {row['partition']} | {row['model']} | "
            f"{row['operating_point']} | "
            f"{row['threshold']:.8f} | "
            f"{row['precision']:.4f} | "
            f"{row['recall']:.4f} | "
            f"{row['f1_score']:.4f} | "
            f"{row['false_positive_rate']:.6f} | "
            f"{row['roc_auc']:.4f} | "
            f"{row['average_precision']:.4f} |"
        )

    lines.extend([
        "",
        "## Interpretation notes",
        "",
        "- ROC-AUC and average precision evaluate score ranking.",
        "- Precision, recall, F1 and FPR depend on the selected threshold.",
        "- AE uses reconstruction error as its ranking score.",
        "- AE+LR Fusion is not a calibrated fraud probability.",
        "- Rolling-window results should be interpreted descriptively.",
        "",
    ])

    (OUTPUT_DIR / "unified_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    for path in [
        VALIDATION_PATH,
        TEST_PATH,
        AE_RESULTS_PATH,
        LR_PATH,
        RF_PATH,
        FUSION_CONFIG_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading saved thresholds...")
    saved_thresholds = load_saved_thresholds()

    print("Loading validation and test scores...")
    validation = load_scores(VALIDATION_PATH, "Validation")
    test = load_scores(TEST_PATH, "Test")

    # Select matched thresholds using validation data only.
    matched_thresholds = {}

    for model_name, column in MODELS.items():
        y_val = validation["Class"].to_numpy(dtype=int)
        scores_val = validation[column].to_numpy(dtype=float)

        matched_thresholds[model_name] = {}

        for target in FPR_TARGETS:
            point_name = f"matched_fpr_{target:.3f}"
            matched_thresholds[model_name][point_name] = (
                select_fpr_threshold(y_val, scores_val, target)
            )

        for target in RECALL_TARGETS:
            point_name = f"matched_recall_{target:.2f}"
            matched_thresholds[model_name][point_name] = (
                select_recall_threshold(
                    y_val,
                    scores_val,
                    target,
                )
            )

    print("Validation-selected matched thresholds:")
    print(json.dumps(matched_thresholds, indent=2))

    print("\nEvaluating validation partition...")
    validation_result, validation_rows = evaluate_partition(
        "validation",
        validation,
        saved_thresholds,
        matched_thresholds,
    )

    print("\nEvaluating test partition...")
    test_result, test_rows = evaluate_partition(
        "test",
        test,
        saved_thresholds,
        matched_thresholds,
    )

    print("\nCalculating chronological rolling windows...")
    validation_windows = build_rolling_windows(
        "validation",
        validation,
        3,
    )

    test_windows = build_rolling_windows(
        "test",
        test,
        4,
    )

    result = {
        "project": "FraudGuard AI",
        "evaluation_scope": [
            "AE",
            "LR",
            "RF",
            "AE_LR_Fusion",
        ],
        "test_used_for_training_or_threshold_selection": False,
        "threshold_selection_partition": "validation",
        "saved_thresholds": saved_thresholds,
        "matched_thresholds_selected_on_validation": (
            matched_thresholds
        ),
        "bootstrap": {
            "iterations": BOOTSTRAP_ITERATIONS,
            "confidence_level": 0.95,
            "method": "stratified_percentile_bootstrap",
            "random_seed": RANDOM_SEED,
        },
        "validation": validation_result,
        "test": test_result,
        "rolling_windows": {
            "validation": validation_windows,
            "test": test_windows,
        },
        "limitations": [
            "GNN and Hybrid scores are not included in this first pass.",
            "AE+LR Fusion is not a calibrated probability.",
            "Rolling-window results are descriptive.",
            "Dataset does not contain card IDs or customer IDs "
            "for velocity evaluation.",
        ],
    }

    summary_rows = validation_rows + test_rows
    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        OUTPUT_DIR / "unified_summary.csv",
        index=False,
    )

    with (OUTPUT_DIR / "unified_evaluation.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(result, file, indent=2)

    write_markdown_report(result, summary_rows)

    print("\n=== UNIFIED EVALUATION COMPLETED ===")
    print("JSON: results/unified/unified_evaluation.json")
    print("CSV:  results/unified/unified_summary.csv")
    print("MD:   results/unified/unified_report.md")


if __name__ == "__main__":
    main()

