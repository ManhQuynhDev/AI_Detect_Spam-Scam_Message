
from pathlib import Path
import pandas as pd
import re

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

INPUT_PATH = OUTPUT_DIR / "number_occurrence_analysis.csv"

BEFORE_OUTPUT = OUTPUT_DIR / "unknown_context_before.csv"
AFTER_OUTPUT = OUTPUT_DIR / "unknown_context_after.csv"
SAMPLE_OUTPUT = OUTPUT_DIR / "unknown_context_samples.csv"


def normalize_context(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    df = pd.read_csv(INPUT_PATH)

    unknown = df[
        df["predicted_component"] == "NUMBER_UNKNOWN"
    ].copy()

    unknown["context_before_clean"] = (
        unknown["context_before"].apply(normalize_context)
    )

    unknown["context_after_clean"] = (
        unknown["context_after"].apply(normalize_context)
    )

    # Thống kê cụm từ ngay trước NUMBER
    before = (
        unknown["context_before_clean"]
        .value_counts()
        .rename_axis("context")
        .reset_index(name="frequency")
    )

    # Thống kê cụm từ ngay sau NUMBER
    after = (
        unknown["context_after_clean"]
        .value_counts()
        .rename_axis("context")
        .reset_index(name="frequency")
    )

    before.to_csv(BEFORE_OUTPUT, index=False, encoding="utf-8-sig")
    after.to_csv(AFTER_OUTPUT, index=False, encoding="utf-8-sig")

    # Lấy mẫu để kiểm tra thủ công
    sample_size = min(50, len(unknown))
    samples = unknown.sample(
        n=sample_size,
        random_state=42
    )

    samples[
        [
            "message_id",
            "label",
            "occurrence_index",
            "context_before",
            "context_after",
            "original_text"
        ]
    ].to_csv(
        SAMPLE_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print("===== UNKNOWN CONTEXT ANALYSIS =====")
    print("Tong occurrence:", len(unknown))
    print("So tin nhan:", unknown["message_id"].nunique())

    print("\n===== TOP 20 CONTEXT BEFORE =====")
    print(before.head(20).to_string(index=False))

    print("\n===== TOP 20 CONTEXT AFTER =====")
    print(after.head(20).to_string(index=False))

    print("\n===== OUTPUT FILES =====")
    print(BEFORE_OUTPUT)
    print(AFTER_OUTPUT)
    print(SAMPLE_OUTPUT)


if __name__ == "__main__":
    main()
