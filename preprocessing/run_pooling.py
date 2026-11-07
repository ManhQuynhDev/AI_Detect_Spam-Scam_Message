
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

EMBEDDING_PATH = BASE_DIR / "output" / "contextual_embeddings.npz"
INDEX_PATH = BASE_DIR / "output" / "embedding_index.csv"

MEAN_OUTPUT = BASE_DIR / "output" / "mean_pooling.npy"
CLS_OUTPUT = BASE_DIR / "output" / "cls_pooling.npy"
INDEX_OUTPUT = BASE_DIR / "output" / "pooled_embeddings_index.csv"


def main():
    print("DANG DOC EMBEDDINGS...")

    embeddings = np.load(EMBEDDING_PATH)
    index_df = pd.read_csv(INDEX_PATH)

    if len(embeddings.files) != len(index_df):
        raise ValueError("So luong Embeddings khong khop Index")

    mean_vectors = []
    cls_vectors = []

    print("DANG THUC HIEN POOLING...")

    for _, row in index_df.iterrows():

        key = row["embedding_key"]

        if key not in embeddings:
            raise ValueError(f"Khong tim thay Embedding: {key}")

        matrix = embeddings[key]

        if matrix.ndim != 2 or matrix.shape[1] != 768:
            raise ValueError(f"Ma tran khong hop le: {key}")

        if matrix.shape[0] < 3:
            raise ValueError(f"Doan qua ngan: {key}")

        # CLS Pooling
        cls_vector = matrix[0]

        # Mean Pooling: bo qua CLS va SEP
        content_vectors = matrix[1:-1]

        mean_vector = np.mean(
            content_vectors,
            axis=0
        )

        mean_vectors.append(mean_vector)
        cls_vectors.append(cls_vector)

    mean_vectors = np.asarray(
        mean_vectors,
        dtype=np.float32
    )

    cls_vectors = np.asarray(
        cls_vectors,
        dtype=np.float32
    )

    # Luu Vector
    np.save(MEAN_OUTPUT, mean_vectors)
    np.save(CLS_OUTPUT, cls_vectors)

    # Luu Index
    index_df["mean_vector_index"] = range(len(index_df))
    index_df["cls_vector_index"] = range(len(index_df))

    index_df.to_csv(
        INDEX_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== KET QUA POOLING =====")
    print("Tong so doan:", len(index_df))
    print("Mean Pooling Shape:", mean_vectors.shape)
    print("CLS Pooling Shape:", cls_vectors.shape)

    print("Mean NaN:", np.isnan(mean_vectors).any())
    print("CLS NaN:", np.isnan(cls_vectors).any())

    print("\nDA LUU KET QUA!")
    print("Mean:", MEAN_OUTPUT)
    print("CLS:", CLS_OUTPUT)
    print("Index:", INDEX_OUTPUT)


if __name__ == "__main__":
    main()
