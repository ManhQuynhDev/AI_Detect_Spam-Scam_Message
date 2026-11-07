
from pathlib import Path
import pandas as pd
import re

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR
    / "output"
    / "number_occurrence_analysis.csv"
)

SAMPLE_LIMIT = 20


# ==========================================
# 2. LOAD DATA
# ==========================================

def load_data():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay file: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "message_id",
        "label",
        "context_before",
        "context_after",
        "predicted_component"
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset thieu cac cot: {missing}"
        )

    # Tránh lỗi NaN khi tìm kiếm bằng regex
    for col in ["context_before", "context_after"]:
        df[col] = df[col].fillna("").astype(str)

    print("Da doc du lieu thanh cong.")
    print("Tong so occurrence:", len(df))

    return df


# ==========================================
# 3. DISPLAY SAMPLES
# ==========================================

def show_samples(title, samples, limit=SAMPLE_LIMIT):

    print("\n" + "=" * 60)
    print(f"===== {title} =====")
    print("=" * 60)

    if samples.empty:
        print("Khong tim thay mau phu hop.")
        return

    samples = samples.head(limit)

    print("So mau hien thi:", len(samples))

    for _, row in samples.iterrows():

        print("\n------------------------------")
        print("Message ID:", row["message_id"])
        print("Label:", row["label"])
        print("Category:", row["predicted_component"])
        print("Before:", row["context_before"])
        print("[NUMBER]")
        print("After:", row["context_after"])


# ==========================================
# 4. ANALYZE OTP CANDIDATES
# ==========================================

def analyze_otp(df):

    otp_pattern = (
        r"otp|"
        r"mã xác thực|ma xac thuc|"
        r"mã bảo mật|ma bao mat|"
        r"verification code|"
        r"security code"
    )

    has_otp_context = (
        df["context_before"].str.contains(
            otp_pattern,
            case=False,
            regex=True,
            na=False
        )
        |
        df["context_after"].str.contains(
            otp_pattern,
            case=False,
            regex=True,
            na=False
        )
    )

    candidates = df[
        has_otp_context
        & (df["predicted_component"] != "OTP")
    ]

    show_samples(
        "OTP CANDIDATES",
        candidates
    )


# ==========================================
# 5. ANALYZE METER / ELECTRICITY
# ==========================================

def analyze_meter(df):

    meter_pattern = (
        r"công\s*tơ|"
        r"cong\s*to|"
        r"c\.to|"
        r"công\s*tơ\s*điện|"
        r"chỉ\s*số|"
        r"chi\s*so|"
        r"\bkwh\b|"
        r"điện|dien"
    )

    has_meter_context = (
        df["context_before"].str.contains(
            meter_pattern,
            case=False,
            regex=True,
            na=False
        )
        |
        df["context_after"].str.contains(
            meter_pattern,
            case=False,
            regex=True,
            na=False
        )
    )

    candidates = df[
        has_meter_context
        & (df["predicted_component"] == "NUMBER_UNKNOWN")
    ]

    show_samples(
        "METER AND ELECTRICITY CANDIDATES",
        candidates
    )


# ==========================================
# 6. ANALYZE DOCUMENT NUMBER
# ==========================================

def analyze_document(df):

    candidates = df[
        df["predicted_component"] == "DOCUMENT_NUMBER"
    ]

    show_samples(
        "DOCUMENT_NUMBER SAMPLES",
        candidates,
        limit=30
    )


# ==========================================
# 7. MAIN
# ==========================================

def main():

    df = load_data()

    print("\nBAT DAU KIEM TRA COMPONENT...")

    analyze_otp(df)

    analyze_meter(df)

    analyze_document(df)

    print("\n" + "=" * 60)
    print("HOAN THANH KIEM TRA COMPONENT")
    print("=" * 60)


if __name__ == "__main__":
    main()
