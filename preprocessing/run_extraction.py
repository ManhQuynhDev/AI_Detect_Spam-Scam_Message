
import json
from pathlib import Path

import pandas as pd

from .component_extraction import extract_components


# =====================================================
# 1. ĐƯỜNG DẪN
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "output" / "cleaned_dataset.csv"
OUTPUT_PATH = BASE_DIR / "output" / "extracted_dataset.csv"


# =====================================================
# 2. HÀM CHUYỂN DANH SÁCH THÀNH JSON
# =====================================================

def convert_to_json(value):

    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)

    return json.dumps([], ensure_ascii=False)


# =====================================================
# 3. HÀM XỬ LÝ CHÍNH
# =====================================================

def main():

    print("ĐANG ĐỌC DỮ LIỆU...")

    df = pd.read_csv(INPUT_PATH)

    if "text_clean" not in df.columns:
        raise ValueError("Không tìm thấy cột text_clean")

    print("Tổng số tin nhắn:", len(df))

    # ---------------------------------------------
    # Trích xuất thành phần từ từng tin nhắn
    # ---------------------------------------------

    print("\nĐANG TRÍCH XUẤT THÀNH PHẦN...")

    extracted_data = df["text_clean"].apply(
        extract_components
    )

    # Chuyển danh sách dictionary thành DataFrame
    component_df = pd.DataFrame(
        extracted_data.tolist(),
        index=df.index
    )

    # Đặt tiền tố để tránh trùng tên cột
    component_df = component_df.add_prefix("component_")

    # ---------------------------------------------
    # Chuyển các danh sách thành JSON
    # ---------------------------------------------

    component_columns = [
        "component_url",
        "component_date",
        "component_time",
        "component_money",
        "component_phone",
        "component_otp",
        "component_brand",
        "component_action"
    ]

    for col in component_columns:

        if col in component_df.columns:
            component_df[col] = component_df[col].apply(
                convert_to_json
            )

    # ---------------------------------------------
    # Ghép dữ liệu gốc và dữ liệu trích xuất
    # ---------------------------------------------

    result = pd.concat(
        [df, component_df],
        axis=1
    )

    # ---------------------------------------------
    # Thống kê kết quả
    # ---------------------------------------------

    print("\n" + "=" * 50)
    print("THỐNG KÊ KẾT QUẢ TRÍCH XUẤT")
    print("=" * 50)

    for col in component_columns:

        if col not in result.columns:
            continue

        count = result[col].apply(
            lambda x: len(json.loads(x)) > 0
        ).sum()

        print(f"{col.upper():<22}: {count}")

    # ---------------------------------------------
    # Lưu dữ liệu
    # ---------------------------------------------

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 50)
    print("HOÀN THÀNH TRÍCH XUẤT")
    print("=" * 50)

    print("Tổng số tin nhắn:", len(result))
    print("Số cột dữ liệu:", len(result.columns))
    print("Đường dẫn:", OUTPUT_PATH)

    # ---------------------------------------------
    # Hiển thị kết quả mẫu
    # ---------------------------------------------

    print("\n===== KẾT QUẢ MẪU =====")

    for _, row in result.head(3).iterrows():

        print("\nNội dung:", row["text_clean"])

        for col in component_columns:
            print(
                col.replace("component_", "").upper(),
                ":",
                row[col]
            )


if __name__ == "__main__":
    main()
