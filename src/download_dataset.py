from datasets import load_dataset
import pandas as pd

dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")

label_names = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

rows = []

for split in ["train", "validation", "test"]:
    for item in dataset[split]:
        rows.append({
            "text": item["text"],
            "sentiment": label_names[item["label"]],
            "split": split
        })

df = pd.DataFrame(rows)

df.to_csv("data/raw/sentiment.csv", index=False)

print("Dataset downloaded successfully!")
print("Shape:", df.shape)
print("\nClass distribution:")
print(df["sentiment"].value_counts())
print("\nSplits:")
print(df["split"].value_counts())