
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "output" / "tokenized_dataset.csv"
OUTPUT_DIR = BASE_DIR / "output"
EMBEDDING_PATH = OUTPUT_DIR / "contextual_embeddings.npz"
INDEX_PATH = OUTPUT_DIR / "embedding_index.csv"

MODEL_NAME = "bert-base-multilingual-cased"
BATCH_SIZE = 16


def main():
    print("DANG DOC DU LIEU...")

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "segment_id",
        "input_ids",
        "attention_mask"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Thieu cot: {col}")

    print("Tong so doan:", len(df))

    # Chon thiet bi xu ly
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Thiet bi:", device)

    # Tai mBERT Encoder
    print("\nDANG TAI mBERT ENCODER...")

    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    print("Tai mo hinh thanh cong!")

    embedding_data = {}
    index_data = []

    # Xu ly theo tung batch
    print("\nDANG TRICH XUAT EMBEDDINGS...")

    for start in range(0, len(df), BATCH_SIZE):
        batch_df = df.iloc[start:start + BATCH_SIZE]

        input_ids_list = [
            json.loads(x) for x in batch_df["input_ids"]
        ]

        attention_list = [
            json.loads(x) for x in batch_df["attention_mask"]
        ]

        # Padding cac chuoi trong batch
        max_len = max(len(x) for x in input_ids_list)

        padded_ids = [
            x + [0] * (max_len - len(x))
            for x in input_ids_list
        ]

        padded_attention = [
            x + [0] * (max_len - len(x))
            for x in attention_list
        ]

        input_ids = torch.tensor(
            padded_ids, dtype=torch.long
        ).to(device)

        attention_mask = torch.tensor(
            padded_attention, dtype=torch.long
        ).to(device)

        # Khong tinh gradient vi khong huan luyen
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # Lay Hidden States cua layer cuoi
        hidden_states = outputs.last_hidden_state

        # Chuyen ve CPU va luu Vector theo tung doan
        hidden_states = hidden_states.cpu().numpy()

        for i, (_, row) in enumerate(batch_df.iterrows()):
            token_length = len(input_ids_list[i])

            # Loai bo phan Padding
            vector = hidden_states[i, :token_length, :]

            key = f"embedding_{start + i:06d}"

            embedding_data[key] = vector.astype(np.float32)

            index_data.append({
                "row_index": start + i,
                "segment_id": row["segment_id"],
                "token_count": token_length,
                "embedding_key": key
            })

        print(
            f"Da xu ly: {min(start + BATCH_SIZE, len(df))}/{len(df)}"
        )

    # Luu embeddings
    print("\nDANG LUU KET QUA...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        EMBEDDING_PATH,
        **embedding_data
    )

    index_df = pd.DataFrame(index_data)

    index_df.to_csv(
        INDEX_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== KET QUA EMBEDDINGS =====")
    print("Tong so doan:", len(index_df))
    print("Hidden size:", model.config.hidden_size)
    print("So layer:", model.config.num_hidden_layers)
    print("File Vector:", EMBEDDING_PATH)
    print("File Index:", INDEX_PATH)


if __name__ == "__main__":
    main()
