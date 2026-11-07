
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

def analyze_vectors(name, vectors):
    print(f"\n===== {name} =====")

    print("Shape:", vectors.shape)
    print("Data type:", vectors.dtype)
    print("NaN:", np.isnan(vectors).any())
    print("Inf:", np.isinf(vectors).any())

    print("Min:", vectors.min())
    print("Max:", vectors.max())
    print("Mean:", vectors.mean())
    print("Std:", vectors.std())

    norms = np.linalg.norm(vectors, axis=1)

    print("Norm trung binh:", norms.mean())
    print("Norm nho nhat:", norms.min())
    print("Norm lon nhat:", norms.max())


def main():
    mean_vectors = np.load(
        OUTPUT_DIR / "mean_pooling.npy"
    )

    cls_vectors = np.load(
        OUTPUT_DIR / "cls_pooling.npy"
    )

    index_df = pd.read_csv(
        OUTPUT_DIR / "pooled_embeddings_index.csv"
    )

    analyze_vectors("MEAN POOLING", mean_vectors)
    analyze_vectors("CLS POOLING", cls_vectors)

    print("\n===== KIEM TRA INDEX =====")
    print("So dong:", len(index_df))
    print(index_df.head())

    print("\nKIEM TRA HOAN TAT!")


if __name__ == "__main__":
    main()
