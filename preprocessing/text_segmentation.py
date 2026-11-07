
import re
import pandas as pd


def segment_text(text):

    if pd.isna(text):
        return []

    text = str(text).strip()

    if not text:
        return []

    # Tách theo dấu kết thúc câu, dấu chấm phẩy
    # hoặc dấu gạch ngang phân tách ý
    segments = re.split(
        r'(?<=[.!?;])\s+|\s+-\s+',
        text
    )

    # Làm sạch khoảng trắng và loại bỏ đoạn rỗng
    segments = [
        segment.strip()
        for segment in segments
        if segment.strip()
    ]

    return segments


if __name__ == "__main__":

    sample = (
        "Xác thực NGAY để không bị TẠM NGỪNG dịch vụ data! "
        "Truy cập https://viettel.vn/xacthucTB để thực hiện "
        "và nhận [MONEY]. "
        "Hoặc truy cập VNeID để xác nhận sử dụng. "
        "Chi tiết liên hệ 198."
    )

    result = segment_text(sample)

    print("NỘI DUNG GỐC:")
    print(sample)

    print("\nKẾT QUẢ PHÂN ĐOẠN:")

    for index, segment in enumerate(result, start=1):
        print(f"Đoạn {index}: {segment}")
