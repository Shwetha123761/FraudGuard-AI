import json
from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

OUTPUT_CSV = RESULTS_DIR / "fraud_model_comparison.csv"
OUTPUT_JSON = RESULTS_DIR / "fraud_model_comparison.json"

REPORTS = {
    "Logistic Regression": "supervised_current_test_results.json",
    "Random Forest": "random_forest_proper_results.json",
    "Autoencoder": "autoencoder_results.json",
    "GNN + Autoencoder": "live_style_final_test_results.json",
}


# ============================================================
# 2. LOAD JSON REPORTS
# ============================================================

def load_json(filename):
    path = RESULTS_DIR / filename

    if not path.exists():
        print(f"WARNING: Missing report: {path}")
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_metrics(model_name, report):
    if report is None:
        return {
            "Model": model_name,
            "Status": "Report missing",
        }

    # Logistic Regression report format.
    if model_name == "Logistic Regression":
        metrics = report
        threshold = report.get("threshold")
        transactions = report.get("transactions")
        fraud_cases = report.get("fraud_cases")
        dataset = report.get("dataset", "Not specified")

        tp = report.get("tp")
        fp = report.get("fp")
        tn = report.get("tn")
        fn = report.get("fn")

        precision = report.get("precision")
        recall = report.get("recall")
        f1 = report.get("f1")
        fpr = report.get("false_positive_rate")
        roc_auc = report.get("roc_auc")
        average_precision = report.get("average_precision")

    # Random Forest report format.
    elif model_name == "Random Forest":
        metrics = report.get("test_metrics", {})
        threshold = metrics.get("threshold")
        transactions = metrics.get("transactions")
        fraud_cases = metrics.get("fraud_cases")
        dataset = "data/processed/test_data.csv"

        tp = metrics.get("true_positives")
        fp = metrics.get("false_positives")
        tn = metrics.get("true_negatives")
        fn = metrics.get("false_negatives")

        precision = metrics.get("precision")
        recall = metrics.get("recall")
        f1 = metrics.get("f1_score")
        fpr = metrics.get("false_positive_rate")
        roc_auc = metrics.get("roc_auc")
        average_precision = metrics.get("average_precision")

    # Autoencoder report format.
    elif model_name == "Autoencoder":
        metrics = report.get("test_metrics", {})
        threshold = metrics.get("threshold")
        transactions = (
            metrics.get("tn", 0)
            + metrics.get("fp", 0)
            + metrics.get("fn", 0)
            + metrics.get("tp", 0)
        )
        fraud_cases = metrics.get("fn", 0) + metrics.get("tp", 0)
        dataset = "data/processed/test_data.csv"

        tp = metrics.get("tp")
        fp = metrics.get("fp")
        tn = metrics.get("tn")
        fn = metrics.get("fn")

        precision = metrics.get("precision")
        recall = metrics.get("recall")
        f1 = metrics.get("f1")
        fpr = metrics.get("false_positive_rate")
        roc_auc = metrics.get("roc_auc")
        average_precision = metrics.get("pr_auc_average_precision")

    # GNN + Autoencoder hybrid report format.
    else:
        metrics = report
        threshold = report.get("fusion_threshold")
        transactions = report.get("transactions")
        fraud_cases = report.get("fraud_cases")
        dataset = report.get("dataset", "Not specified")

        tp = report.get("tp")
        fp = report.get("fp")
        tn = report.get("tn")
        fn = report.get("fn")

        precision = report.get("precision")
        recall = report.get("recall")
        f1 = report.get("f1")
        fpr = report.get("false_positive_rate")
        roc_auc = report.get("roc_auc")
        average_precision = report.get("average_precision")

    return {
        "Model": model_name,
        "Status": "Loaded",
        "Dataset": dataset,
        "Transactions": transactions,
        "Actual fraud cases": fraud_cases,
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "False Positive Rate": fpr,
        "ROC-AUC": roc_auc,
        "Average Precision": average_precision,
        "True Positives": tp,
        "False Positives": fp,
        "True Negatives": tn,
        "False Negatives": fn,
    }


# ============================================================
# 3. BUILD COMPARISON
# ============================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for model_name, filename in REPORTS.items():
        report = load_json(filename)
        rows.append(get_metrics(model_name, report))

    df = pd.DataFrame(rows)

    # Add a note to prevent misinterpreting missing or incomparable values.
    df["Comparison note"] = (
        "Check threshold selection and evaluation methodology before ranking."
    )

    df.to_csv(OUTPUT_CSV, index=False)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=4)

    print("\n" + "=" * 95)
    print("FRAUDGUARD AI — MODEL COMPARISON")
    print("=" * 95)

    display_columns = [
        "Model",
        "Precision",
        "Recall",
        "F1-score",
        "False Positive Rate",
        "ROC-AUC",
        "Average Precision",
    ]

    available_columns = [
        column for column in display_columns if column in df.columns
    ]

    print(df[available_columns].to_string(index=False))

    print("\nConfusion-matrix counts:")
    confusion_columns = [
        "Model",
        "True Positives",
        "False Positives",
        "True Negatives",
        "False Negatives",
    ]

    print(df[confusion_columns].to_string(index=False))

    print(f"\nCSV report saved to:\n{OUTPUT_CSV}")
    print(f"JSON report saved to:\n{OUTPUT_JSON}")

    print("\nImportant:")
    print(
        "These reports summarize existing evaluations. "
        "They do not guarantee identical evaluation procedures."
    )


if __name__ == "__main__":
    main()