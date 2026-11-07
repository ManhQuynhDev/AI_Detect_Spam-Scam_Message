
import re
import json
from pathlib import Path
from collections import Counter

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

INPUT_PATH = OUTPUT_DIR / "cleaned_dataset.csv"
OUTPUT_PATH = OUTPUT_DIR / "number_context_analysis.csv"

df = pd.read_csv(INPUT_PATH)


def classify_number(text, start, end):
    # Lấy ngữ cảnh trước và sau [NUMBER]
    before = text[max(0, start - 100):start].lower()
    after = text[end:min(len(text), end + 100)].lower()

    context = before + " [NUMBER] " + after

    # Ưu tiên các trường hợp có ý nghĩa rõ ràng
    if re.search(r"otp|mã xác thực|mã bảo mật", before):
        return "OTP"

    if re.search(r"thông tư|nghị định|văn bản", before):
        return "DOCUMENT_NUMBER"

    if re.search(r"\b(?:tb|thuê bao)\s*$", before):
        return "SUBSCRIBER_ID"

    if re.search(r"\b(?:soạn|soan)\b.{0,50}\b(?:gửi|gui)\s*$", before):
        return "SHORTCODE"

    if re.search(r"\b(?:zalo|hotline|cskh|liên hệ|lien he|lh)\s*:?\s*$", before):
        return "CONTACT"

    if re.search(r"\b(?:gọi|goi)\s+(?:ngay\s+)?$", before):
        return "CONTACT"

    # Placeholder nằm trong đường dẫn hoặc tên miền
    if re.search(r"https?://\S*$|www\.\S*$", before):
        return "URL_COMPONENT"

    return "NUMBER_UNKNOWN"


def extract_number_contexts(text):
    results = []

    for match in re.finditer(r"\[NUMBER\]", text, re.IGNORECASE):
        category = classify_number(
            text,
            match.start(),
            match.end()
        )

        results.append(category)

    return results


df["number_context"] = df["text_clean"].fillna("").apply(
    extract_number_contexts
)

df["number_context_json"] = df["number_context"].apply(
    json.dumps
)

# Thống kê số tin nhắn có từng loại component
for label in sorted(df["label"].dropna().unique()):
    subset = df[df["label"] == label]

    counts = Counter()

    for categories in subset["number_context"]:
        counts.update(set(categories))

    print(f"\n===== LABEL {label} =====")
    for category, count in counts.most_common():
        percentage = count / len(subset) * 100
        print(f"{category:20s}: {count:5d} ({percentage:.2f}%)")

# Lưu kết quả
result = df[[
    "message_id",
    "label",
    "text_clean",
    "number_context_json"
]].copy()

result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("\nDa luu:", OUTPUT_PATH)
