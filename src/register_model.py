import mlflow

model_path = "models/bert_sentiment_kaggle"

print("Registering Kaggle BERT model in MLflow...")

with mlflow.start_run() as run:

    model_info = mlflow.transformers.log_model(
        transformers_model=model_path,
        name="bert_sentiment_kaggle_model",
        task="text-classification",
        registered_model_name="sentiment-bert-kaggle"
    )

    print("\nModel logged successfully!")
    print("Run ID:", run.info.run_id)
    print("Model URI:", model_info.model_uri)

print("\nKaggle BERT model registration completed!")