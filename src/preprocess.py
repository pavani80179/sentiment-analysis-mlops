import pandas as pd

input_file = "data/raw/kaggle_sentiment.xlsx"
output_file = "data/processed/kaggle_sentiment_clean.csv"

# Load dataset
df = pd.read_excel(input_file)

# Keep only the columns needed for sentiment analysis
df = df[["text", "sentiment"]]

# Remove unnecessary spaces
df["text"] = df["text"].str.strip()

# Remove duplicate texts just in case
df = df.drop_duplicates(subset=["text"])

# Save cleaned dataset
df.to_csv(output_file, index=False)

print("Preprocessing completed!")
print("Final shape:", df.shape)
print("\nSentiment distribution:")
print(df["sentiment"].value_counts())
print("\nDuplicates:", df["text"].duplicated().sum())