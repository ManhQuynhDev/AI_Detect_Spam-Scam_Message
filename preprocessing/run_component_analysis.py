
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

INPUT_PATH = OUTPUT_DIR / "extracted_dataset.csv"
LABEL_PATH = OUTPUT_DIR / "cleaned_dataset.csv"

COMPONENTS = [
    "url",
    "otp",
    "money",
    "time",
    "date",
    "phone",
    "brand",
    "action"
]


def parse_component(value):
    if pd.isna(value):
        return []

    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def main():

    print("DANG DOC DU LIEU...")

    df = pd.read_csv(INPUT_PATH)
    labels = pd.read_csv(LABEL_PATH)

    # 1. Kiem tra du lieu
    if "message_id" not in df.columns:
        raise ValueError("Thieu message_id trong extracted_dataset")

    if "message_id" not in labels.columns or "label" not in labels.columns:
        raise ValueError("Thieu message_id hoac label trong cleaned_dataset")

    print("Tong so tin nhan:", len(df))

    # 2. Gan nhan tu cleaned_dataset.csv

    # Loai bo cot label cu neu da ton tai
    df = df.drop(
        columns=["label", "label_x", "label_y"],
        errors="ignore"
    )

    label_data = labels[["message_id", "label"]].copy()

    # Kiem tra message_id trung lap
    if label_data["message_id"].duplicated().any():
        raise ValueError(
            "message_id bi trung lap trong cleaned_dataset"
        )

    # Kiem tra du lieu nhan
    if label_data["label"].isna().any():
        raise ValueError("Co nhan bi thieu trong cleaned_dataset")

    # Merge nhan
    df = df.merge(
        label_data,
        on="message_id",
        how="left",
        validate="one_to_one"
    )

    if df["label"].isna().any():
        raise ValueError("Co tin nhan chua duoc gan nhan")

    print("\nDA GAN NHAN THANH CONG!")
    print("\nPHAN BO NHAN:")
    print(df["label"].value_counts().sort_index())

    # 3. Xu ly cac cot Component
    print("\nDANG PHAN TICH COMPONENT...")

    for component in COMPONENTS:

        column = f"component_{component}"

        if column not in df.columns:
            raise ValueError(f"Khong tim thay cot {column}")

        df[column] = df[column].apply(parse_component)

        df[f"has_{component}"] = df[column].apply(
            lambda values: int(len(values) > 0)
        )

    # 4. Thong ke Component theo nhan
    summary = []

    for label, group in df.groupby("label"):

        total = len(group)

        for component in COMPONENTS:

            count = group[f"has_{component}"].sum()

            summary.append({
                "label": label,
                "component": component,
                "message_count": total,
                "count_with_component": int(count),
                "percentage": round(count / total * 100, 2)
            })

    summary_df = pd.DataFrame(summary)

    # 5. Luu bang thong ke
    summary_path = OUTPUT_DIR / "component_summary_by_label.csv"

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    # 6. Luu du lieu chi tiet
    detail_columns = (
        ["message_id", "label"]
        + [f"has_{c}" for c in COMPONENTS]
    )

    detail_path = OUTPUT_DIR / "component_detail_by_message.csv"

    df[detail_columns].to_csv(
        detail_path,
        index=False,
        encoding="utf-8-sig"
    )

    # 7. Hien thi ket qua
    print("\n===== COMPONENT ANALYSIS =====")

    pivot_table = summary_df.pivot(
        index="component",
        columns="label",
        values="percentage"
    )

    print(pivot_table.round(2))

    print("\n===== DA LUU KET QUA =====")
    print(summary_path)
    print(detail_path)


if __name__ == "__main__":
    main()
