import pandas as pd

df = pd.read_csv("data/raw/sentiment.csv")

print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nSplit distribution:")
print(df["split"].value_counts())

print("\nEmpty texts:", (df["text"].str.strip() == "").sum())