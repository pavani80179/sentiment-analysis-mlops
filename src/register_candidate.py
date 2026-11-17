
from pathlib import Path
import mlflow

ROOT = Path(__file__).resolve().parent.parent

CANDIDATE_PATH = ROOT / "models" / "bert_sentiment_kaggle_candidate"

CANDIDATE_MODEL_NAME = "sentiment-bert-kaggle-candidate"

if not CANDIDATE_PATH.exists():
    raise FileNotFoundError(
        f"Candidate model not found: {CANDIDATE_PATH}"
    )

print("Registering candidate model in MLflow...")

mlflow.set_experiment("sentiment-analysis-retraining")

with mlflow.start_run(
    run_name="register-bert-candidate"
) as run:

    model_info = mlflow.transformers.log_model(
        transformers_model=str(CANDIDATE_PATH),
        name="bert_candidate_model",
        task="text-classification",
        registered_model_name=CANDIDATE_MODEL_NAME,
    )

    print("\nCandidate model registered!")
    print("Run ID:", run.info.run_id)
    print("Model URI:", model_info.model_uri)

print("\nCandidate registration completed.")
print("Original production model remains unchanged.")

