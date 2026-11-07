
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR
    / "output"
    / "number_occurrence_analysis.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "output"
    / "document_number_review.csv"
)


def main():

    df = pd.read_csv(INPUT_PATH)

    samples = df[
        df["predicted_component"] == "DOCUMENT_NUMBER"
    ].copy()

    print("\n===== DOCUMENT NUMBER REVIEW =====")

    print("Tong occurrence:", len(samples))
    print(
        "So tin nhan:",
        samples["message_id"].nunique()
    )

    if samples.empty:
        print("Khong co DOCUMENT_NUMBER.")
        return

    for _, row in samples.iterrows():

        print("\n--------------------------------")
        print("Message ID:", row["message_id"])
        print("Label:", row["label"])
        print("Occurrence:", row["occurrence_index"])
        print("Before:", row["context_before"])
        print("[NUMBER]")
        print("After:", row["context_after"])

    samples.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDa luu:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
