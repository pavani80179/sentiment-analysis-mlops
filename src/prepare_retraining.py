import pandas as pd

production_file = "monitoring/retraining_data.csv"
reference_file = "data/processed/kaggle_sentiment_clean.csv"

production = pd.read_csv(production_file)
reference = pd.read_csv(reference_file)

# Match production text with original Kaggle dataset
label_map = reference.set_index("text")["sentiment"]

production["true_sentiment"] = production["text"].map(label_map)

production.to_csv(production_file, index=False)

matched = production["true_sentiment"].notna().sum()
unmatched = production["true_sentiment"].isna().sum()

print("Retraining data updated!")
print("Total samples:", len(production))
print("Labels found:", matched)
print("Labels still missing:", unmatched)