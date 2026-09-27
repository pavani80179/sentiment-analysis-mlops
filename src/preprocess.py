import pandas as pd

input_file = "data/raw/sentiment.csv"
output_file = "data/processed/sentiment_clean.csv"

df = pd.read_csv(input_file)

# Remove duplicate rows
df = df.drop_duplicates()

# Remove leading/trailing spaces from text
df["text"] = df["text"].str.strip()

# Save cleaned dataset
df.to_csv(output_file, index=False)

print("Preprocessing completed!")
print("Final shape:", df.shape)
print("Duplicates:", df.duplicated().sum())
print("\nSentiment distribution:")
print(df["sentiment"].value_counts())