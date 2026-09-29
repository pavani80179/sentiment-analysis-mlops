from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

import csv
import os
from datetime import datetime


# ==========================================================
# FastAPI application
# ==========================================================

app = FastAPI(
    title="Sentiment Analysis API",
    description="Kaggle BERT-based sentiment analysis API",
    version="1.0"
)


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# Load Kaggle-trained BERT model
# ==========================================================

sentiment_model = pipeline(
    "sentiment-analysis",
    model="models/bert_sentiment_kaggle",
    tokenizer="models/bert_sentiment_kaggle"
)


# ==========================================================
# Request format
# ==========================================================

class TextRequest(BaseModel):
    text: str


# ==========================================================
# Prediction log
# ==========================================================

LOG_FILE = "monitoring/predictions.csv"

os.makedirs("monitoring", exist_ok=True)


if not os.path.exists(LOG_FILE):

    with open(
        LOG_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "text",
            "sentiment",
            "confidence"
        ])


# ==========================================================
# Health check
# ==========================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "bert_sentiment_kaggle"
    }


# ==========================================================
# Sentiment prediction
# ==========================================================

@app.post("/predict")
def predict_sentiment(request: TextRequest):

    result = sentiment_model(
        request.text
    )[0]

    sentiment = result["label"]

    confidence = round(
        result["score"],
        4
    )

    # Save prediction for monitoring
    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            datetime.now().isoformat(),
            request.text,
            sentiment,
            confidence
        ])

    return {
        "text": request.text,
        "sentiment": sentiment,
        "confidence": confidence
    }