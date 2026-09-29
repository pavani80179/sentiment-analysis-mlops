import pandas as pd
import mlflow

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# ==========================================================
# 1. Load Kaggle dataset
# ==========================================================

df = pd.read_csv(
    "data/processed/kaggle_sentiment_clean.csv"
)

print("Dataset shape:", df.shape)


# ==========================================================
# 2. Convert sentiment labels to numbers
# ==========================================================

label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

df["label"] = df["sentiment"].map(label_map)


# ==========================================================
# 3. Train / Validation / Test split
# ==========================================================

train_df, temp_df = train_test_split(
    df[["text", "label"]],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

print("\nSplit sizes:")
print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))


# ==========================================================
# 4. Load BERT tokenizer
# ==========================================================

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)


# ==========================================================
# 5. Convert to Hugging Face datasets
# ==========================================================

train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df,
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df,
    preserve_index=False
)


# ==========================================================
# 6. Tokenization
# ==========================================================

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})


# ==========================================================
# 7. Load BERT model
# ==========================================================

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3,
    id2label={
        0: "negative",
        1: "neutral",
        2: "positive"
    },
    label2id={
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }
)


# ==========================================================
# 8. Evaluation metrics
# ==========================================================

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = predictions.argmax(axis=-1)

    return {
        "accuracy": accuracy_score(
            labels,
            predictions
        ),
        "f1": f1_score(
            labels,
            predictions,
            average="weighted"
        )
    }


# ==========================================================
# 9. Training configuration
# ==========================================================

training_args = TrainingArguments(
    output_dir="models/bert_sentiment_kaggle",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    max_steps=300,
    weight_decay=0.01,
    eval_strategy="no",
    save_strategy="no",
    logging_steps=25,
    report_to="none"
)


# ==========================================================
# 10. Trainer
# ==========================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)


# ==========================================================
# 11. MLflow experiment
# ==========================================================

mlflow.set_experiment(
    "sentiment-analysis-bert"
)

print("\nStarting BERT training...")
print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Test samples:", len(test_df))
print("Training steps: 300")


# ==========================================================
# 12. Train and track with MLflow
# ==========================================================

with mlflow.start_run():

    # ------------------------------
    # Log parameters
    # ------------------------------

    mlflow.log_param(
        "dataset",
        "kaggle_sentiment"
    )

    mlflow.log_param(
        "model",
        "bert-base-uncased"
    )

    mlflow.log_param(
        "learning_rate",
        2e-5
    )

    mlflow.log_param(
        "batch_size",
        8
    )

    mlflow.log_param(
        "max_length",
        64
    )

    mlflow.log_param(
        "max_steps",
        300
    )

    mlflow.log_param(
        "train_samples",
        len(train_df)
    )

    mlflow.log_param(
        "validation_samples",
        len(val_df)
    )

    mlflow.log_param(
        "test_samples",
        len(test_df)
    )


    # ------------------------------
    # Train
    # ------------------------------

    trainer.train()


    # ======================================================
    # Validation results
    # ======================================================

    print("\nValidation Results:")

    val_results = trainer.evaluate(
        dataset["validation"]
    )

    print(val_results)

    mlflow.log_metric(
        "validation_accuracy",
        val_results["eval_accuracy"]
    )

    mlflow.log_metric(
        "validation_f1",
        val_results["eval_f1"]
    )

    mlflow.log_metric(
        "validation_loss",
        val_results["eval_loss"]
    )


    # ======================================================
    # Test results
    # ======================================================

    print("\nTest Results:")

    test_results = trainer.evaluate(
        dataset["test"]
    )

    print(test_results)

    mlflow.log_metric(
        "test_accuracy",
        test_results["eval_accuracy"]
    )

    mlflow.log_metric(
        "test_f1",
        test_results["eval_f1"]
    )

    mlflow.log_metric(
        "test_loss",
        test_results["eval_loss"]
    )


    # ======================================================
    # Save NEW Kaggle model
    # ======================================================

    model_path = "models/bert_sentiment_kaggle"

    trainer.save_model(model_path)

    tokenizer.save_pretrained(model_path)

    print("\nModel saved successfully!")
    print("Model location:", model_path)


print("\nMLflow run completed!")