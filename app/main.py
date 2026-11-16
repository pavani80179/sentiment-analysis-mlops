
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
# CORS configuration
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)


# ==========================================================
# Serve frontend
# ==========================================================

@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse("frontend/index.html")


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
# Prediction logging
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

    result = sentiment_model(request.text)[0]

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
