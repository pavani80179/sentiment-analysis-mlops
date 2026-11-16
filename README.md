\# Sentiment Analysis Pipeline with Monitoring



An end-to-end MLOps project that performs sentiment classification using BERT, tracks experiments, versions datasets, monitors data drift, and serves predictions through a deployed FastAPI application.



\## Project Overview



This project classifies text into three sentiment categories:



\- Positive

\- Negative

\- Neutral



It integrates machine learning, experiment tracking, data versioning, model monitoring, API deployment, and CI automation.



\## Architecture



\*\*Data → Preprocessing → BERT Model → MLflow → FastAPI → Docker → Google Cloud Run\*\*



Evidently AI monitors production data for drift, while the retraining workflow creates and evaluates a separate candidate model.



\## Technology Stack



| Component | Technology |

|---|---|

| NLP Model | BERT (`bert-base-uncased`) |

| Programming | Python |

| Dataset Versioning | DVC + Google Cloud Storage |

| Experiment Tracking | MLflow |

| Model Registry | MLflow Model Registry |

| Monitoring | Evidently AI |

| API | FastAPI |

| Containerization | Docker |

| Cloud Deployment | Google Cloud Run |

| CI | GitHub Actions |



\## Project Structure



```text

sentiment-analysis-mlops/

│

├── app/

│   └── main.py

├── data/

│   ├── raw/

│   └── processed/

├── frontend/

│   └── index.html

├── models/

│   ├── bert\_sentiment\_kaggle/

│   └── bert\_sentiment\_kaggle\_candidate/

├── monitoring/

│   ├── predictions.csv

│   ├── retraining\_data.csv

│   ├── drift\_report.py

│   └── drift\_report.html

├── src/

│   ├── preprocess.py

│   ├── train.py

│   ├── register\_model.py

│   └── retrain.py

├── .github/

│   └── workflows/

│       └── ci.yml

├── Dockerfile.cloud

├── requirements-cloud.txt

└── README.md

```



\## Key Features



\### 1. BERT Sentiment Classification

Uses a fine-tuned BERT model to classify text into positive, negative, and neutral categories.



\### 2. DVC Dataset Versioning

Tracks raw and processed datasets using DVC and stores dataset objects in Google Cloud Storage.



\### 3. MLflow Experiment Tracking

Records model experiments, evaluation metrics, parameters, and registered model versions.



\### 4. Data Drift Monitoring

Uses Evidently AI to compare reference data with production data and generate drift reports.



\### 5. Model Retraining

Implements a retraining script that fine-tunes a separate candidate model and evaluates it against the baseline model.



\### 6. FastAPI Model Serving

Provides REST API endpoints for health checks and sentiment predictions.



\### 7. Docker and Cloud Deployment

Containerizes the application and deploys it using Google Cloud Run.



\### 8. GitHub Actions CI

Runs Python syntax checks automatically on pushes and pull requests to the master branch.



\## Live Application



\*\*Cloud Run URL:\*\*  

https://sentiment-api-302901775614.asia-south1.run.app



\### API Endpoints



| Method | Endpoint | Purpose |

|---|---|---|

| GET | `/health` | Check API health |

| POST | `/predict` | Predict text sentiment |

| GET | `/docs` | Interactive Swagger documentation |



\### Prediction Example



Request:



```json

{

&#x20; "text": "I absolutely love this project!"

}

```



The API returns a sentiment label and confidence score.



\## Monitoring and Retraining



1\. Production predictions are recorded in a CSV file.

2\. Evidently AI compares production data with reference data.

3\. Drift results are saved as HTML and JSON reports.

4\. MLflow records monitoring experiments.

5\. The retraining script trains and evaluates a separate candidate model.



Candidate models are kept separate from the deployed baseline until they are sufficiently validated.



\## Running Locally



Install dependencies:



```bash

pip install -r requirements.txt

```



Start the API:



```bash

uvicorn app.main:app --reload

```



Open Swagger documentation:



```text

http://127.0.0.1:8000/docs

```



\## Project Outcomes



\- BERT-based sentiment classification.

\- Dataset versioning using DVC.

\- Experiment tracking and model registration using MLflow.

\- Data drift detection using Evidently AI.

\- Separate candidate model retraining.

\- REST API deployment using FastAPI.

\- Docker containerization.

\- Public cloud deployment using Google Cloud Run.

\- Automated CI checks using GitHub Actions.



\## Repository



GitHub: https://github.com/pavani80179/sentiment-analysis-mlops



\---



\*\*Project:\*\* Sentiment Analysis Pipeline with Monitoring  

\*\*Domain:\*\* Natural Language Processing and MLOps

