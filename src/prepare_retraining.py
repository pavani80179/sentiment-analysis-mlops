
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

PRODUCTION_FILE = ROOT / "monitoring" / "retraining_data.csv"

if not PRODUCTION_FILE.exists():
    raise FileNotFoundError(PRODUCTION_FILE)

data = pd.read_csv(PRODUCTION_FILE)

required = {"text", "predicted_sentiment", "true_sentiment"}

if not required.issubset(data.columns):
    raise ValueError(f"Missing columns: {required - set(data.columns)}")

# Preserve existing labels and record their source.
if "label_source" not in data.columns:
    data["label_source"] = "legacy_unverified"

data["true_sentiment"] = data["true_sentiment"].fillna("")

# Never copy predictions into true_sentiment.
unlabelled = data["true_sentiment"].astype(str).str.strip().eq("")

data.loc[unlabelled, "label_source"] = "unlabelled"

data.to_csv(PRODUCTION_FILE, index=False)

print("Retraining data preparation completed.")
print("Total examples:", len(data))
print("Existing labels preserved:", (~unlabelled).sum())
print("Unlabelled examples:", unlabelled.sum())
print("Predictions converted into true labels: 0")
