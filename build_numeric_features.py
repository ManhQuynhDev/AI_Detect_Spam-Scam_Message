
from pathlib import Path
import re
import pandas as pd

# =====================================
# 1. CONFIG
# =====================================

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "output"

INPUT_FILE = OUTPUT_DIR / "number_component_features.csv"
OUTPUT_FILE = OUTPUT_DIR / "combined_message_features.csv"

# =====================================
# 2. REGEX AND KEYWORDS
# =====================================

# Nhận diện các dãy số thực tế
NUMBER_PATTERN = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)")

OTP_KEYWORDS = [
    "otp", "ma xac thuc", "ma xac minh",
    "ma bao mat", "ma giao dich",
    "ma kich hoat", "mat khau mot lan",
    "xac thuc giao dich"
]

MONEY_KEYWORDS = [
    "vnd", "vnđ", "dong", "tien",
    "so tien", "thanh toan", "chuyen khoan",
    "nap tien", "rut tien", "gia tri giao dich"
]

PHONE_KEYWORDS = [
    "so dien thoai", "sdt", "dien thoai",
    "hotline", "lien he", "goi den",
    "zalo", "sms"
]

ACCOUNT_KEYWORDS = [
    "so tai khoan", "tai khoan", "stk",
    "account", "the ngan hang",
    "tai khoan thanh toan"
]

URL_PATTERN = re.compile(
    r"(https?://|www\.|[a-zA-Z0-9-]+\.(com|vn|net|org)\b)",
    re.IGNORECASE
)

# =====================================
# 3. HELPER FUNCTIONS
# =====================================

def contains_keyword(text, keywords):
    text = str(text).lower()
    return int(any(keyword in text for keyword in keywords))


def extract_numeric_features(text):
    text = str(text)

    # Lấy tất cả dãy số thực tế
    numbers = NUMBER_PATTERN.findall(text)

    digit_lengths = [
        len(re.sub(r"\D", "", number))
        for number in numbers
    ]

    features = {
        "numeric_count": len(numbers),

        "total_numeric_digits": sum(digit_lengths),

        "has_4_to_8_digit_number": int(
            any(4 <= length <= 8 for length in digit_lengths)
        ),

        "has_6_digit_number": int(
            any(length == 6 for length in digit_lengths)
        ),

        "has_otp_context": contains_keyword(
            text, OTP_KEYWORDS
        ),

        "has_money_context": contains_keyword(
            text, MONEY_KEYWORDS
        ),

        "has_phone_context": contains_keyword(
            text, PHONE_KEYWORDS
        ),

        "has_account_context": contains_keyword(
            text, ACCOUNT_KEYWORDS
        ),

        "has_url": int(
            bool(URL_PATTERN.search(text))
        )
    }

    return pd.Series(features)


# =====================================
# 4. LOAD DATA
# =====================================

print("Loading feature dataset...")

df = pd.read_csv(INPUT_FILE)

if "text_clean" not in df.columns:
    raise ValueError("Không tìm thấy cột text_clean")

print(f"Total messages: {len(df)}")

# =====================================
# 5. EXTRACT NUMERIC FEATURES
# =====================================

print("Extracting numeric features...")

numeric_features = df["text_clean"].apply(
    extract_numeric_features
)

# =====================================
# 6. COMBINE FEATURES
# =====================================

result = pd.concat(
    [
        df.reset_index(drop=True),
        numeric_features.reset_index(drop=True)
    ],
    axis=1
)

# =====================================
# 7. SAVE
# =====================================

result.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nNumeric feature extraction completed!")

print(f"Output: {OUTPUT_FILE}")
print(f"Rows: {len(result)}")
print(f"Columns: {len(result.columns)}")

print("\nNumeric feature statistics:")

numeric_columns = numeric_features.columns.tolist()

print(
    result[numeric_columns].sum().sort_values(
        ascending=False
    )
)

print("\nPreview:")
print(result.head())
