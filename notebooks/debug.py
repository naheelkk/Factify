import csv
import pandas as pd

# df = pd.read_csv("../data/processed/balanced_df.csv")
# print(df.head())
# print()
# print(df.isnull().sum())
# print()
# print(df.shape)

# Check what's actually in the CSV file
with open("../data/processed/balanced_df.csv", 'r', encoding='utf-8') as f:
    print("First few lines of raw CSV:")
    for i, line in enumerate(f):
        print(f"Line {i}: {repr(line)}")
        if i > 5:
            break

# Try loading with different parameters
df_test = pd.read_csv("../data/processed/balanced_df.csv", 
                      encoding='utf-8',
                      quoting=csv.QUOTE_ALL,  # or csv.QUOTE_MINIMAL
                      escapechar='\\')