
import json
from pathlib import Path

import pandas as pd
from .mbert_tokenizer import load_tokenizer

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "output" / "segmented_dataset.csv"
OUTPUT_PATH = BASE_DIR / "output" / "tokenized_dataset.csv"

MAX_LENGTH = 128


def main():
    print("DANG DOC DU LIEU...")

    df = pd.read_csv(INPUT_PATH)

    if "segment_text" not in df.columns:
        raise ValueError("Khong tim thay cot segment_text")

    print("Tong so doan:", len(df))

    print("\nDANG TAI TOKENIZER mBERT...")
    tokenizer = load_tokenizer()

    token_results = []

    print("\nDANG TOKENIZE DU LIEU...")

    for text in df["segment_text"].fillna(""):
        encoded = tokenizer(
            str(text),
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False
        )

        tokens = tokenizer.convert_ids_to_tokens(
            encoded["input_ids"]
        )

        token_results.append({
            "tokens": tokens,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "token_count": len(encoded["input_ids"])
        })

    token_df = pd.DataFrame(token_results)

    for col in ["tokens", "input_ids", "attention_mask"]:
        token_df[col] = token_df[col].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )

    result = pd.concat(
        [df.reset_index(drop=True), token_df],
        axis=1
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== KET QUA TOKENIZATION =====")
    print("Tong so doan:", len(result))
    print("So Token trung binh:", round(result["token_count"].mean(), 2))
    print("So Token lon nhat:", result["token_count"].max())
    print("So doan vuot MAX_LENGTH:", (result["token_count"] >= MAX_LENGTH).sum())

    print("\nDA LUU FILE:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
