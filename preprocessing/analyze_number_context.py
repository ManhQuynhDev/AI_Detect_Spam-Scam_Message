
from pathlib import Path
from collections import Counter
import pandas as pd
import re

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "output" / "cleaned_dataset.csv"

df = pd.read_csv(DATASET_PATH)

contexts = Counter()

for text in df["text_clean"].fillna("").astype(str):
    for match in re.finditer(r"\[NUMBER\]", text, re.IGNORECASE):
        start = max(0, match.start() - 60)
        end = min(len(text), match.end() + 60)

        context = text[start:end].replace("\n", " ")
        contexts[context] += 1

print("TONG SO NGU CANH:", sum(contexts.values()))
print("\n===== 30 NGU CANH PHO BIEN =====")

for context, count in contexts.most_common(30):
    print(f"{count:>4} | {context}")
