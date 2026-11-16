import pandas as pd

input_file = "data/raw/kaggle_sentiment.xlsx"
output_file = "data/processed/kaggle_sentiment_clean.csv"

df = pd.read_excel(input_file)

df = df[["text", "sentiment"]]

df["text"] = df["text"].str.strip()

df = df.drop_duplicates(subset=["text"])

df.to_csv(output_file, index=False)

print("Preprocessing completed!")
print("Final shape:", df.shape)
print("\nSentiment distribution:")
print(df["sentiment"].value_counts())
print("\nDuplicates:", df["text"].duplicated().sum())