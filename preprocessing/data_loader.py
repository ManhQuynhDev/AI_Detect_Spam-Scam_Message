
import pandas as pd
from pathlib import Path


DATA_PATH = Path("dataset/vietnamese_sms_message.csv")


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    # Đổi tên message thành text để thống nhất
    if "message" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"message": "text"})

    print("\n===== THÔNG TIN DATASET =====")
    print("Số dòng:", len(df))
    print("Các cột:", df.columns.tolist())

    print("\n===== KIỂU DỮ LIỆU =====")
    print(df.dtypes)

    print("\n===== DỮ LIỆU THIẾU =====")
    print(df.isnull().sum())

    if "label" in df.columns:
        print("\n===== PHÂN BỐ NHÃN =====")
        print(df["label"].value_counts())

    print("\n===== 5 TIN NHẮN MẪU =====")
    print(df.head())

    return df


if __name__ == "__main__":
    df = load_dataset()
