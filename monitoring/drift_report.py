
from pathlib import Path
import json

import pandas as pd
import mlflow

from evidently import Report
from evidently.presets import DataDriftPreset


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

REFERENCE_PATH = (
    ROOT / "data" / "processed" / "kaggle_sentiment_clean.csv"
)
PREDICTIONS_PATH = ROOT / "monitoring" / "predictions.csv"
REPORT_PATH = ROOT / "monitoring" / "drift_report.html"
RESULTS_PATH = ROOT / "monitoring" / "evidently_results.json"

EXPERIMENT_NAME = "sentiment-analysis-monitoring"


# --------------------------------------------------
# 2. Load reference dataset
# --------------------------------------------------
if not REFERENCE_PATH.exists():
    raise FileNotFoundError(
        f"Reference dataset not found: {REFERENCE_PATH}"
    )

reference_dataset = pd.read_csv(REFERENCE_PATH)

if "text" not in reference_dataset.columns:
    raise ValueError(
        "Reference dataset must contain a 'text' column."
    )

reference_data = reference_dataset[["text"]].copy()
reference_data["text"] = reference_data["text"].fillna("").astype(str)

reference_data = reference_data[
    reference_data["text"].str.strip() != ""
]

reference_data = reference_data.sample(
    n=min(1000, len(reference_data)),
    random_state=42,
)

if len(reference_data) < 5:
    raise ValueError("Not enough reference samples for drift analysis.")


# --------------------------------------------------
# 3. Load production prediction logs
# --------------------------------------------------
if not PREDICTIONS_PATH.exists():
    raise FileNotFoundError(
        f"Production predictions not found: {PREDICTIONS_PATH}"
    )

production_data = pd.read_csv(PREDICTIONS_PATH)

if "text" not in production_data.columns:
    raise ValueError(
        "Prediction logs must contain a 'text' column."
    )

production_data = production_data[["text"]].copy()
production_data["text"] = production_data["text"].fillna("").astype(str)

production_data = production_data[
    production_data["text"].str.strip() != ""
]

production_data = production_data.drop_duplicates(subset=["text"])

production_data = production_data[
    ~production_data["text"].str.contains(
        "\n",
        regex=False,
        na=False,
    )
]

current_data = production_data.reset_index(drop=True)

print("Reference samples:", len(reference_data))
print("Production samples:", len(current_data))

if len(current_data) < 5:
    raise ValueError(
        "Not enough production samples. Make at least 5 valid "
        "API predictions before running drift monitoring."
    )


# --------------------------------------------------
# 4. Run Evidently drift analysis
# --------------------------------------------------
report = Report(
    metrics=[
        DataDriftPreset(),
    ]
)

snapshot = report.run(
    current_data=current_data,
    reference_data=reference_data,
)

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

snapshot.save_html(str(REPORT_PATH))

print("\nEvidently HTML report generated:")
print(REPORT_PATH)


# --------------------------------------------------
# 5. Extract actual Evidently metrics
# --------------------------------------------------
results = snapshot.dict()

with open(RESULTS_PATH, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2, default=str)

metrics = results.get("metrics", [])

drift_score = None
drift_share = None
drifted_columns = None

for metric in metrics:
    metric_name = metric.get("metric_name", "")
    value = metric.get("value")

    # Extract the text column's actual drift score.
    if metric_name.startswith("ValueDrift(") and "column=text" in metric_name:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            drift_score = float(value)

    # Extract the count and share of drifted columns.
    elif metric_name.startswith("DriftedColumnsCount("):
        if isinstance(value, dict):
            count = value.get("count")
            share = value.get("share")

            if count is not None:
                drifted_columns = int(count)

            if share is not None:
                drift_share = float(share)

if drifted_columns is None:
    raise ValueError(
        "Could not extract DriftedColumnsCount from Evidently results. "
        f"Inspect {RESULTS_PATH}."
    )

dataset_drift = drifted_columns > 0

print("\nEvidently drift results:")
print("Text drift score:", drift_score)
print("Drifted columns:", drifted_columns)
print("Actual drift share:", drift_share)
print("Dataset drift detected:", dataset_drift)


# --------------------------------------------------
# 6. Log monitoring metrics to MLflow
# --------------------------------------------------
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(
    run_name="evidently-drift-monitoring"
) as run:

    mlflow.log_param(
        "reference_dataset",
        "kaggle_sentiment_clean",
    )

    mlflow.log_param(
        "production_source",
        "fastapi_predictions",
    )

    mlflow.log_param(
        "monitoring_tool",
        "Evidently",
    )

    mlflow.log_metric(
        "reference_samples",
        len(reference_data),
    )

    mlflow.log_metric(
        "production_samples",
        len(current_data),
    )

    mlflow.log_metric(
        "drifted_columns",
        drifted_columns,
    )

    mlflow.log_metric(
        "drift_detected",
        int(dataset_drift),
    )

    if drift_score is not None:
        mlflow.log_metric(
            "text_drift_score",
            drift_score,
        )

    if drift_share is not None:
        mlflow.log_metric(
            "drift_share",
            drift_share,
        )

    mlflow.log_artifact(
        str(REPORT_PATH),
        artifact_path="drift_report",
    )

    mlflow.log_artifact(
        str(RESULTS_PATH),
        artifact_path="evidently_results",
    )

    print("\nMLflow monitoring run created!")
    print("Run ID:", run.info.run_id)


# --------------------------------------------------
# 7. Completion summary
# --------------------------------------------------
print("\nData drift monitoring completed!")
print("HTML report:", REPORT_PATH)
print("Raw Evidently results:", RESULTS_PATH)
print("MLflow experiment:", EXPERIMENT_NAME)
