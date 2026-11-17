
from pathlib import Path
import random

import mlflow
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Subset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "monitoring" / "verified_retraining_data.csv"

BASE_MODEL_PATH = ROOT / "models" / "bert_sentiment_kaggle"

CANDIDATE_PATH = (
    ROOT / "models" / "bert_sentiment_kaggle_candidate"
)

LABEL_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

MAX_LENGTH = 64
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-5
MIN_VERIFIED_SAMPLES = 100
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# 2. Validate retraining data
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Verified retraining data not found: {DATA_PATH}"
    )

if not BASE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Production baseline model not found: {BASE_MODEL_PATH}"
    )

data = pd.read_csv(DATA_PATH)

required_columns = {
    "text",
    "true_sentiment",
    "label_source",
}

if not required_columns.issubset(data.columns):
    raise ValueError(
        f"Missing required columns: "
        f"{required_columns - set(data.columns)}"
    )

# Only explicitly human-verified labels are eligible.
data = data[
    data["label_source"].astype(str).str.strip().eq(
        "human_verified"
    )
].copy()

data = data.dropna(
    subset=["text", "true_sentiment"]
)

data["text"] = data["text"].astype(str).str.strip()

data["true_sentiment"] = (
    data["true_sentiment"]
    .astype(str)
    .str.strip()
    .str.lower()
)

data = data[data["text"] != ""].copy()

data = data[
    data["true_sentiment"].isin(LABEL_TO_ID)
].copy()

data = data.drop_duplicates(
    subset=["text"]
).reset_index(drop=True)

print("Human-verified examples:", len(data))
print("Device:", DEVICE)

if len(data) < MIN_VERIFIED_SAMPLES:
    raise ValueError(
        f"Retraining stopped safely. At least "
        f"{MIN_VERIFIED_SAMPLES} human-verified examples "
        f"are required. Available: {len(data)}"
    )

class_counts = data["true_sentiment"].value_counts()

print("\nVerified label distribution:")
print(class_counts)

if not set(LABEL_TO_ID).issubset(class_counts.index):
    raise ValueError(
        "All three sentiment classes must be present."
    )

if class_counts.min() < 2:
    raise ValueError(
        "Each sentiment class needs at least two examples."
    )

data["label"] = data["true_sentiment"].map(LABEL_TO_ID)


# --------------------------------------------------
# 3. Stratified train/test split
# --------------------------------------------------

train_indices, test_indices = train_test_split(
    np.arange(len(data)),
    test_size=0.2,
    random_state=SEED,
    stratify=data["label"],
)

print("\nTraining examples:", len(train_indices))
print("Test examples:", len(test_indices))


# --------------------------------------------------
# 4. Tokenization
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    str(BASE_MODEL_PATH)
)

encoded = tokenizer(
    data["text"].tolist(),
    max_length=MAX_LENGTH,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)

labels = torch.tensor(
    data["label"].tolist(),
    dtype=torch.long,
)

dataset = TensorDataset(
    encoded["input_ids"],
    encoded["attention_mask"],
    labels,
)

train_dataset = Subset(
    dataset,
    train_indices.tolist()
)

test_dataset = Subset(
    dataset,
    test_indices.tolist()
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# --------------------------------------------------
# 5. Model evaluation function
# --------------------------------------------------

def evaluate_model(model):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for input_ids, attention_mask, batch_labels in test_loader:

            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            predictions = outputs.logits.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                batch_labels.tolist()
            )

    return {
        "accuracy": float(
            accuracy_score(all_labels, all_predictions)
        ),

        "macro_f1": float(
            f1_score(
                all_labels,
                all_predictions,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
    }


# --------------------------------------------------
# 6. Evaluate original production model
# --------------------------------------------------

print("\nLoading original BERT model...")

baseline_model = (
    AutoModelForSequenceClassification
    .from_pretrained(str(BASE_MODEL_PATH))
    .to(DEVICE)
)

baseline_metrics = evaluate_model(baseline_model)

print("\nOriginal model results:")
print("Accuracy:", round(baseline_metrics["accuracy"], 4))
print("Macro F1:", round(baseline_metrics["macro_f1"], 4))

del baseline_model

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()


# --------------------------------------------------
# 7. Fine-tune separate candidate model
# --------------------------------------------------

print("\nLoading candidate from original BERT weights...")

candidate_model = (
    AutoModelForSequenceClassification
    .from_pretrained(str(BASE_MODEL_PATH))
    .to(DEVICE)
)

optimizer = torch.optim.AdamW(
    candidate_model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01,
)

for epoch in range(EPOCHS):

    candidate_model.train()

    total_loss = 0.0

    for input_ids, attention_mask, batch_labels in train_loader:

        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = candidate_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=batch_labels,
        )

        loss = outputs.loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            candidate_model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Training loss: {average_loss:.4f}"
    )


# --------------------------------------------------
# 8. Evaluate candidate
# --------------------------------------------------

candidate_metrics = evaluate_model(candidate_model)

print("\nCandidate model results:")
print("Accuracy:", round(candidate_metrics["accuracy"], 4))
print("Macro F1:", round(candidate_metrics["macro_f1"], 4))


# --------------------------------------------------
# 9. Save candidate separately
# --------------------------------------------------

CANDIDATE_PATH.mkdir(
    parents=True,
    exist_ok=True,
)

candidate_model.save_pretrained(
    str(CANDIDATE_PATH)
)

tokenizer.save_pretrained(
    str(CANDIDATE_PATH)
)

print("\nCandidate saved:")
print(CANDIDATE_PATH)

del candidate_model


# --------------------------------------------------
# 10. MLflow experiment tracking
# --------------------------------------------------

mlflow.set_experiment(
    "sentiment-analysis-retraining"
)

with mlflow.start_run(
    run_name="verified-bert-retraining-candidate"
) as run:

    mlflow.log_param(
        "base_model",
        "bert_sentiment_kaggle",
    )

    mlflow.log_param(
        "training_dataset",
        "verified_retraining_data.csv",
    )

    mlflow.log_param(
        "label_source",
        "human_verified",
    )

    mlflow.log_param(
        "training_examples",
        len(train_indices),
    )

    mlflow.log_param(
        "test_examples",
        len(test_indices),
    )

    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning_rate", LEARNING_RATE)

    mlflow.log_metric(
        "baseline_accuracy",
        baseline_metrics["accuracy"],
    )

    mlflow.log_metric(
        "baseline_macro_f1",
        baseline_metrics["macro_f1"],
    )

    mlflow.log_metric(
        "candidate_accuracy",
        candidate_metrics["accuracy"],
    )

    mlflow.log_metric(
        "candidate_macro_f1",
        candidate_metrics["macro_f1"],
    )

    print("\nMLflow run created!")
    print("Run ID:", run.info.run_id)


# --------------------------------------------------
# 11. Completion summary
# --------------------------------------------------

print("\nRetraining completed successfully.")

print("Original production model remains unchanged.")
print("Candidate model saved separately.")
print("Candidate requires independent evaluation.")
print("Automatic production promotion is disabled.")
