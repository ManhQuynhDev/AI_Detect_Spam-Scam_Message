
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

df = pd.read_csv(OUTPUT_DIR / "cleaned_dataset.csv")

df["has_number"] = (
    df["text_clean"]
    .fillna("")
    .str.contains(r"\[NUMBER\]", case=False, regex=True)
)

print("===== NUMBER PLACEHOLDER BY LABEL =====")

summary = pd.crosstab(
    df["label"],
    df["has_number"],
    normalize="index"
) * 100

print(summary.round(2))

print("\n===== SAMPLE BY LABEL =====")

for label in sorted(df["label"].dropna().unique()):
    print(f"\nLABEL {label}")

    samples = df[
        (df["label"] == label) & df["has_number"]
    ]["text_clean"].head(10)

    for text in samples:
        print("-", text)
