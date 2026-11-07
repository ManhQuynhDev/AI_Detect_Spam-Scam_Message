from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "output" / "unknown_number_review.csv"
OUTPUT_FILE = BASE_DIR / "output" / "unknown_number_first_100.csv"

df = pd.read_csv(INPUT_FILE)

sample = df.head(100).copy()

sample["manual_component"] = ""
sample["review_note"] = ""

sample.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("Total samples:", len(sample))
print("Unique messages:", sample["message_id"].nunique())
print("Saved:", OUTPUT_FILE)