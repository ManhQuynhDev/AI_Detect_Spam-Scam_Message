
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

INPUT_PATH = OUTPUT_DIR / "number_occurrence_analysis.csv"

SUMMARY_PATH = OUTPUT_DIR / "number_component_summary.csv"
LABEL_PATH = OUTPUT_DIR / "number_component_by_label.csv"
MESSAGE_PATH = OUTPUT_DIR / "number_component_message_stats.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    # Tổng số occurrence theo component
    summary = (
        df.groupby("predicted_component")
        .agg(
            total_occurrences=("message_id", "size"),
            unique_messages=("message_id", "nunique")
        )
        .reset_index()
        .sort_values("total_occurrences", ascending=False)
    )

    summary["percentage"] = (
        summary["total_occurrences"]
        / summary["total_occurrences"].sum()
        * 100
    ).round(2)

    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    # Phân bố component theo label
    label_summary = (
        df.groupby(["label", "predicted_component"])
        .agg(
            total_occurrences=("message_id", "size"),
            unique_messages=("message_id", "nunique")
        )
        .reset_index()
    )

    label_totals = df.groupby("label").size().to_dict()

    label_summary["percentage"] = label_summary.apply(
        lambda row: round(
            row["total_occurrences"] / label_totals[row["label"]] * 100,
            2
        ),
        axis=1
    )

    label_summary.to_csv(
        LABEL_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # Thống kê số tin nhắn có từng component
    message_stats = (
        df.drop_duplicates(["message_id", "predicted_component"])
        .groupby("predicted_component")
        .agg(
            unique_messages=("message_id", "nunique")
        )
        .reset_index()
    )

    message_stats.to_csv(
        MESSAGE_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("===== NUMBER ANALYSIS SUMMARY =====")
    print(summary.to_string(index=False))

    print("\n===== COMPONENT BY LABEL =====")
    print(label_summary.to_string(index=False))

    print("\n===== OUTPUT FILES =====")
    print(SUMMARY_PATH)
    print(LABEL_PATH)
    print(MESSAGE_PATH)


if __name__ == "__main__":
    main()
