
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

DATA_PATH = ROOT / "monitoring" / "retraining_data.csv"
BASE_MODEL_PATH = ROOT / "models" / "bert_sentiment_kaggle"
CANDIDATE_PATH = ROOT / "models" / "bert_sentiment_kaggle_candidate"

LABEL_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

MAX_LENGTH = 64
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-5
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
# 2. Validate the retraining data
# --------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Retraining data not found: {DATA_PATH}"
    )

if not BASE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Existing BERT model not found: {BASE_MODEL_PATH}"
    )

data = pd.read_csv(DATA_PATH)

required_columns = {
    "text",
    "predicted_sentiment",
    "true_sentiment",
}

if not required_columns.issubset(data.columns):
    raise ValueError(
        f"CSV must contain these columns: {required_columns}"
    )

data = data.dropna(
    subset=["text", "true_sentiment"]
).copy()

data["text"] = data["text"].astype(str).str.strip()
data["true_sentiment"] = (
    data["true_sentiment"].astype(str).str.strip().str.lower()
)

data = data[data["text"] != ""].copy()

invalid_labels = set(data["true_sentiment"]) - set(LABEL_TO_ID)

if invalid_labels:
    raise ValueError(
        f"Invalid true sentiment labels: {invalid_labels}"
    )

data = data.drop_duplicates(subset=["text"]).reset_index(drop=True)

if len(data) < 15:
    raise ValueError(
        "At least 15 labeled examples are required for this demo."
    )

if data["true_sentiment"].value_counts().min() < 2:
    raise ValueError(
        "Each sentiment class needs at least two labeled examples."
    )

data["label"] = data["true_sentiment"].map(LABEL_TO_ID)

print("Labeled examples:", len(data))
print("Label distribution:")
print(data["true_sentiment"].value_counts())
print("Device:", DEVICE)


# --------------------------------------------------
# 3. Split data into training and test sets
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
# 4. Tokenize the examples
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

train_dataset = Subset(dataset, train_indices.tolist())
test_dataset = Subset(dataset, test_indices.tolist())

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
# 5. Evaluate a model on the held-out test set
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
            all_labels.extend(batch_labels.tolist())

    return {
        "accuracy": accuracy_score(
            all_labels, all_predictions
        ),
        "macro_f1": f1_score(
            all_labels,
            all_predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        ),
    }


# --------------------------------------------------
# 6. Evaluate the original model first
# --------------------------------------------------
baseline_model = (
    AutoModelForSequenceClassification.from_pretrained(
        str(BASE_MODEL_PATH)
    ).to(DEVICE)
)

baseline_metrics = evaluate_model(baseline_model)

print("\nOriginal model test accuracy:",
      round(baseline_metrics["accuracy"], 4))
print("Original model test macro F1:",
      round(baseline_metrics["macro_f1"], 4))

del baseline_model

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()


# --------------------------------------------------
# 7. Fine-tune a separate candidate model
# --------------------------------------------------
candidate_model = (
    AutoModelForSequenceClassification.from_pretrained(
        str(BASE_MODEL_PATH)
    ).to(DEVICE)
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
# 8. Evaluate the candidate model
# --------------------------------------------------
candidate_metrics = evaluate_model(candidate_model)

print("\nCandidate model test accuracy:",
      round(candidate_metrics["accuracy"], 4))
print("Candidate model test macro F1:",
      round(candidate_metrics["macro_f1"], 4))


# --------------------------------------------------
# 9. Save candidate separately
# --------------------------------------------------
CANDIDATE_PATH.mkdir(parents=True, exist_ok=True)

candidate_model.save_pretrained(str(CANDIDATE_PATH))
tokenizer.save_pretrained(str(CANDIDATE_PATH))

print("\nCandidate model saved to:")
print(CANDIDATE_PATH)

# The original model in models/bert_sentiment_kaggle
# is deliberately not modified.


# --------------------------------------------------
# 10. Track retraining in MLflow
# --------------------------------------------------
mlflow.set_experiment("sentiment-analysis-retraining")

with mlflow.start_run(
    run_name="bert-retraining-candidate"
) as run:

    mlflow.log_param("base_model", "bert-base-uncased")
    mlflow.log_param("training_data", "retraining_data.csv")
    mlflow.log_param("training_examples", len(train_indices))
    mlflow.log_param("test_examples", len(test_indices))
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("seed", SEED)

    mlflow.log_metric(
        "baseline_test_accuracy",
        baseline_metrics["accuracy"],
    )
    mlflow.log_metric(
        "baseline_test_macro_f1",
        baseline_metrics["macro_f1"],
    )
    mlflow.log_metric(
        "candidate_test_accuracy",
        candidate_metrics["accuracy"],
    )
    mlflow.log_metric(
        "candidate_test_macro_f1",
        candidate_metrics["macro_f1"],
    )

    mlflow.log_artifact(str(DATA_PATH), artifact_path="training_data")

    print("\nMLflow retraining run created!")
    print("Run ID:", run.info.run_id)


# --------------------------------------------------
# 11. Completion summary
# --------------------------------------------------
print("\nRetraining demonstration completed.")
print("Original production model remains unchanged.")
print("Candidate model is saved separately for review.")
print("Do not promote it without further validation.")
