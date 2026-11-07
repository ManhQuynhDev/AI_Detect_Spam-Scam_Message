
import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "output" / "number_context_analysis.csv"

df = pd.read_csv(INPUT_PATH)

for label in sorted(df["label"].dropna().unique()):
    print(f"\n{'=' * 60}")
    print(f"LABEL {label}")
    print("=" * 60)

    subset = df[df["label"] == label]

    for category in [
        "OTP",
        "CONTACT",
        "SHORTCODE",
        "SUBSCRIBER_ID",
        "URL_COMPONENT",
        "DOCUMENT_NUMBER",
        "NUMBER_UNKNOWN"
    ]:
        samples = []

        for _, row in subset.iterrows():
            try:
                categories = json.loads(row["number_context_json"])
            except (json.JSONDecodeError, TypeError):
                continue

            if category in categories:
                samples.append(row["text_clean"])

            if len(samples) >= 5:
                break

        print(f"\n--- {category} ---")

        if not samples:
            print("Khong co mau.")
        else:
            for text in samples:
                print("-", text)
