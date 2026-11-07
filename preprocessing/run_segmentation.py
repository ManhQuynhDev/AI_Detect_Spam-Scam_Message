
from pathlib import Path
import pandas as pd

from .text_segmentation import segment_text


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "output" / "extracted_dataset.csv"
OUTPUT_PATH = BASE_DIR / "output" / "segmented_dataset.csv"


def main():

    print("ĐANG ĐỌC DỮ LIỆU...")

    df = pd.read_csv(INPUT_PATH)

    if "text_clean" not in df.columns:
        raise ValueError("Không tìm thấy cột text_clean")

    segmented_rows = []

    print("TỔNG SỐ TIN NHẮN:", len(df))

    # Duyệt từng tin nhắn
    for row_index, row in df.iterrows():

        message_id = row.get("message_id", row_index)

        original_text = row["text_clean"]

        segments = segment_text(original_text)

        # Tạo bản ghi cho từng đoạn
        for segment_index, segment in enumerate(
            segments, start=1
        ):

            segmented_rows.append({
                "message_id": message_id,
                "segment_id": f"{message_id}_S{segment_index:02d}",
                "segment_index": segment_index,
                "total_segments": len(segments),
                "original_text": original_text,
                "segment_text": segment,
                "segment_length": len(segment)
            })

    # Chuyển thành DataFrame
    result = pd.DataFrame(segmented_rows)

    # Tạo thư mục lưu kết quả
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Lưu dữ liệu
    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # Thống kê
    print("\n===== THỐNG KÊ PHÂN ĐOẠN =====")

    print("Tổng tin nhắn:", len(df))
    print("Tổng số đoạn:", len(result))

    if not result.empty:
        print(
            "Số đoạn trung bình/tin nhắn:",
            round(
                result.groupby("message_id").size().mean(),
                2
            )
        )

        print(
            "Đoạn ngắn nhất:",
            result["segment_length"].min()
        )

        print(
            "Đoạn dài nhất:",
            result["segment_length"].max()
        )

    print("\nĐÃ LƯU:", OUTPUT_PATH)

    print("\n===== 5 MẪU PHÂN ĐOẠN =====")

    print(
        result[
            ["message_id", "segment_index", "segment_text"]
        ].head(5).to_string(index=False)
    )


if __name__ == "__main__":
    main()
