
from pathlib import Path
import pandas as pd

from text_cleaning import clean_sms_text

# Xác định thư mục gốc của dự án
BASE_DIR = Path(__file__).resolve().parent.parent

# Đường dẫn dữ liệu
INPUT_PATH = BASE_DIR / "dataset" / "vietnamese_sms_message.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_PATH = OUTPUT_DIR / "cleaned_dataset.csv"


def main():
    # Tự động tạo thư mục output nếu chưa tồn tại
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Đọc dữ liệu
    df = pd.read_csv(INPUT_PATH)

    # Chuẩn hóa tên cột
    if "message" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"message": "text"})

    if "text" not in df.columns:
        raise ValueError("Dataset không có cột message hoặc text")

    # Giữ lại nội dung gốc
    df["text_raw"] = df["text"]

    # Làm sạch nội dung
    df["text_clean"] = df["text_raw"].apply(clean_sms_text)

    # Kiểm tra trạng thái xử lý
    df["processing_status"] = df["text_clean"].apply(
        lambda x: "EMPTY" if not x else "SUCCESS"
    )

    # Thống kê
    print("Tổng số tin nhắn:", len(df))
    print("Thành công:", (df["processing_status"] == "SUCCESS").sum())
    print("Rỗng:", (df["processing_status"] == "EMPTY").sum())

    # Lưu kết quả
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\nĐã lưu dữ liệu tại:")
    print(OUTPUT_PATH)

    # Hiển thị 5 mẫu
    print("\n===== SO SÁNH TRƯỚC VÀ SAU =====")

    for _, row in df.head(5).iterrows():
        print("\nGốc:", row["text_raw"])
        print("Sạch:", row["text_clean"])


if __name__ == "__main__":
    main()
