from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "output" / "number_occurrence_analysis.csv"
OUTPUT_FILE = BASE_DIR / "output" / "unknown_number_review.csv"

df = pd.read_csv(INPUT_FILE)

unknown = df[
    df["predicted_component"] == "NUMBER_UNKNOWN"
].copy()

# Giữ lại những cột cần thiết để kiểm tra
review = unknown[
    [
        "message_id",
        "label",
        "occurrence_index",
        "context_before",
        "context_after",
        "original_text"
    ]
].copy()

# Thêm cột để ghi nhãn thủ công
review["manual_component"] = ""
review["review_note"] = ""

review.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("Total unknown occurrences:", len(review))
print("Unique messages:", review["message_id"].nunique())
print("Saved:", OUTPUT_FILE)