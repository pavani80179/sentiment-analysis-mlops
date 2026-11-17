
from pathlib import Path
import json

import mlflow
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "sentiment_clean.csv"
BASE_PATH = ROOT / "models" / "bert_sentiment_kaggle"
CANDIDATE_PATH = ROOT / "models" / "bert_sentiment_kaggle_candidate"
RESULT_PATH = ROOT / "monitoring" / "candidate_evaluation.json"

LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}

MAX_LENGTH = 64
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Promotion remains disabled until the evaluation policy is approved.
PROMOTION_ENABLED = False

if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)

if not BASE_PATH.exists() or not CANDIDATE_PATH.exists():
    raise FileNotFoundError("Baseline or candidate model is missing.")

data = pd.read_csv(DATA_PATH)

data = data[data["split"] == "validation"].copy()
data = data.dropna(subset=["text", "sentiment"])
data["text"] = data["text"].astype(str).str.strip()
data["sentiment"] = data["sentiment"].astype(str).str.lower()

data = data[data["sentiment"].isin(LABEL_TO_ID)]
data = data[data["text"] != ""].drop_duplicates("text")

if len(data) < 1000:
    raise ValueError("At least 1,000 independent validation examples required.")

texts = data["text"].tolist()
true_labels = np.array(
    [LABEL_TO_ID[label] for label in data["sentiment"]]
)

def evaluate_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path)
    ).to(DEVICE)

    model.eval()
    predictions = []

    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]

            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(DEVICE)
                for key, value in encoded.items()
            }

            outputs = model(**encoded)
            predictions.extend(
                outputs.logits.argmax(dim=1).cpu().tolist()
            )

    del model

    return np.array(predictions)

print("Independent validation samples:", len(texts))
print("Device:", DEVICE)

baseline_predictions = evaluate_model(BASE_PATH)
candidate_predictions = evaluate_model(CANDIDATE_PATH)

def calculate_metrics(predictions):
    return {
        "accuracy": float(
            accuracy_score(true_labels, predictions)
        ),
        "macro_f1": float(
            f1_score(
                true_labels,
                predictions,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            true_labels,
            predictions,
            labels=[0, 1, 2],
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            true_labels,
            predictions,
            labels=[0, 1, 2],
        ).tolist(),
    }

baseline_metrics = calculate_metrics(baseline_predictions)
candidate_metrics = calculate_metrics(candidate_predictions)

print("\nBASELINE MODEL")
print("Accuracy:", round(baseline_metrics["accuracy"], 4))
print("Macro F1:", round(baseline_metrics["macro_f1"], 4))

print("\nCANDIDATE MODEL")
print("Accuracy:", round(candidate_metrics["accuracy"], 4))
print("Macro F1:", round(candidate_metrics["macro_f1"], 4))

# No automatic promotion is allowed yet.
result = {
    "validation_samples": len(texts),
    "baseline": baseline_metrics,
    "candidate": candidate_metrics,
    "promotion_enabled": PROMOTION_ENABLED,
    "decision": "PENDING_APPROVAL",
}

RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(RESULT_PATH, "w", encoding="utf-8") as file:
    json.dump(result, file, indent=2)

mlflow.set_experiment("sentiment-analysis-retraining")

with mlflow.start_run(run_name="independent-candidate-evaluation"):
    mlflow.log_param("validation_dataset", "sentiment_clean.csv")
    mlflow.log_param("validation_split", "validation")
    mlflow.log_param("validation_samples", len(texts))
    mlflow.log_param("promotion_enabled", PROMOTION_ENABLED)

    for name, metrics in [
        ("baseline", baseline_metrics),
        ("candidate", candidate_metrics),
    ]:
        mlflow.log_metric(f"{name}_accuracy", metrics["accuracy"])
        mlflow.log_metric(f"{name}_macro_f1", metrics["macro_f1"])

    mlflow.log_artifact(str(RESULT_PATH))

print("\nEvaluation completed!")
print("Promotion enabled:", PROMOTION_ENABLED)
print("Results:", RESULT_PATH)
