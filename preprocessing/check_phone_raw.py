
import re
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "output" / "cleaned_dataset.csv"

df = pd.read_csv(DATASET_PATH)

patterns = {
    "placeholder_number": r"\[NUMBER\]",
    "vietnam_phone": r"(?<!\d)(?:\+84|84|0)(?:[\s.-]?\d){8,10}(?!\d)",
    "short_number": r"(?<!\d)\d{3,6}(?!\d)"
}

for name, pattern in patterns.items():
    matched = df["text_clean"].astype(str).str.contains(
        pattern, regex=True, case=False
    )

    print(f"\n{name}: {matched.sum()} tin nhắn")

    samples = df.loc[matched, "text_clean"].head(5)

    for text in samples:
        print("-", text)
