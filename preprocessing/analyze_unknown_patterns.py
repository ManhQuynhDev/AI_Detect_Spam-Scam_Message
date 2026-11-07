from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE_DIR / "output" / "number_occurrence_analysis.csv"
)

unknown = df[
    df["predicted_component"] == "NUMBER_UNKNOWN"
].copy()

# Tạo ngữ cảnh đầy đủ để kiểm tra
unknown["full_context"] = (
    unknown["context_before"].fillna("")
    + " [NUMBER] "
    + unknown["context_after"].fillna("")
)

patterns = {
    "YEAR": [
        "thang", "nam", "nghi dinh",
        "thong tu", "nq-cp", "nd-cp"
    ],
    "OTP": [
        "ma otp", "ma xac thuc",
        "ma xac minh", "verification code",
        "xac thuc", "xac minh"
    ],
    "SHORTCODE": [
        "soan", "gui", "gui tin nhan"
    ],
    "CONTACT": [
        "lh", "lien he", "goi", "hotline",
        "cham soc khach hang", "cskh"
    ],
    "TRANSACTION_ID": [
        "ma giao dich", "ma gd", "gd:"
    ],
    "ORDER_ID": [
        "don hang", "dh cua ban la",
        "ma don hang"
    ],
    "RECIPIENT_ID": [
        "nguoi nhan"
    ],
    "SERVICE_QUOTA": [
        "sms noi mang va", "mien phi sms",
        "dung luong"
    ]
}

for component, keywords in patterns.items():
    mask = unknown["full_context"].str.lower().apply(
        lambda text: any(keyword in text for keyword in keywords)
    )

    matched = unknown[mask]

    print("\n" + "=" * 65)
    print(component)
    print("Occurrences:", len(matched))
    print("Unique messages:", matched["message_id"].nunique())

    if not matched.empty:
        print(
            matched[
                ["message_id", "context_before", "context_after"]
            ].head(10).to_string(index=False)
        )
        