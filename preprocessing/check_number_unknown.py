
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "output" / "number_occurrence_analysis.csv"
OUTPUT_PATH = BASE_DIR / "output" / "number_unknown_samples.csv"

SAMPLES_PER_LABEL = 50

def main():
    df = pd.read_csv(INPUT_PATH)

    unknown = df[
        df["predicted_component"] == "NUMBER_UNKNOWN"
    ].copy()

    if unknown.empty:
        print("Khong co NUMBER_UNKNOWN.")
        return

    print("Tong NUMBER_UNKNOWN:", len(unknown))

    # Lấy tối đa 50 occurrence cho mỗi label
    samples = (
        unknown.groupby("label", group_keys=False)
        .head(SAMPLES_PER_LABEL)
        .copy()
    )

    samples.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== NUMBER_UNKNOWN SAMPLES =====")

    for label, group in samples.groupby("label"):
        print(f"\nLABEL {label} - {len(group)} samples")

        for _, row in group.iterrows():
            print("\n------------------------------")
            print("Message ID:", row["message_id"])
            print("Occurrence:", row["occurrence_index"])
            print("Before:", row["context_before"])
            print("[NUMBER]")
            print("After:", row["context_after"])

    print("\nDa luu:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
