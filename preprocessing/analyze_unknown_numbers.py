
from pathlib import Path
import pandas as pd

# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "output" / "number_occurrence_analysis.csv"

# Số lượng ví dụ muốn hiển thị
NUM_EXAMPLES = 50


# =====================================================
# 2. LOAD DATA
# =====================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Không tìm thấy file: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

# Kiểm tra các cột cần thiết
required_columns = [
    "message_id",
    "label",
    "predicted_component",
    "context_before",
    "context_after",
    "text_clean"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"File dữ liệu thiếu các cột: {missing_columns}\n"
        f"Các cột hiện có: {list(df.columns)}"
    )


# =====================================================
# 3. FILTER UNKNOWN
# =====================================================

unknown = df[
    df["predicted_component"] == "NUMBER_UNKNOWN"
].copy()

print("=" * 70)
print("NUMBER_UNKNOWN ANALYSIS")
print("=" * 70)

print("Total occurrences:", len(unknown))
print("Unique messages:", unknown["message_id"].nunique())


# =====================================================
# 4. CONTEXT BEFORE ANALYSIS
# =====================================================

print("\n===== CONTEXT BEFORE =====")

print(
    unknown["context_before"]
    .fillna("")
    .value_counts()
    .head(30)
    .to_string()
)


# =====================================================
# 5. CONTEXT AFTER ANALYSIS
# =====================================================

print("\n===== CONTEXT AFTER =====")

print(
    unknown["context_after"]
    .fillna("")
    .value_counts()
    .head(30)
    .to_string()
)


# =====================================================
# 6. UNKNOWN EXAMPLES
# =====================================================

print(f"\n===== {NUM_EXAMPLES} UNKNOWN EXAMPLES =====")

display_columns = [
    "message_id",
    "label",
    "context_before",
    "context_after",
    "text_clean"
]

print(
    unknown[display_columns]
    .head(NUM_EXAMPLES)
    .to_string(index=False)
)


# =====================================================
# 7. EXPORT UNKNOWN DATA FOR MANUAL REVIEW
# =====================================================

OUTPUT_FILE = BASE_DIR / "output" / "unknown_number_examples.csv"

unknown.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n===== EXPORT COMPLETED =====")
print(f"Đã lưu toàn bộ dữ liệu UNKNOWN tại: {OUTPUT_FILE}")
print(f"Tổng số lần xuất hiện được xuất: {len(unknown)}")
