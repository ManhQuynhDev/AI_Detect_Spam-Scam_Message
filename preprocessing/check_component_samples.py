
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

df = pd.read_csv(
    OUTPUT_DIR / "extracted_dataset.csv"
)

for component in ["phone", "otp", "url", "date"]:

    column = f"component_{component}"

    print(f"\n===== {component.upper()} =====")

    if column not in df.columns:
        print("Khong tim thay cot:", column)
        continue

    count = 0

    for _, row in df.iterrows():

        try:
            values = json.loads(row[column])
        except (json.JSONDecodeError, TypeError):
            values = []

        if values:
            print("ID:", row["message_id"])
            print("Text:", row["text_clean"])
            print("Extracted:", values)

            count += 1

            if count >= 5:
                break

    if count == 0:
        print("Khong tim thay mau nao.")

    print("So mau hien thi:", count)
