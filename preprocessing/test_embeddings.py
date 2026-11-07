
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

EMBEDDING_PATH = BASE_DIR / "output" / "contextual_embeddings.npz"
INDEX_PATH = BASE_DIR / "output" / "embedding_index.csv"

def main():
    print("DANG KIEM TRA EMBEDDINGS...")

    embeddings = np.load(EMBEDDING_PATH)
    index_df = pd.read_csv(INDEX_PATH)

    print("Tong so ma tran:", len(embeddings.files))
    print("Tong so doan:", len(index_df))

    first_key = embeddings.files[0]
    first_vector = embeddings[first_key]

    print("\n===== MA TRAN DAU TIEN =====")
    print("Embedding key:", first_key)
    print("Shape:", first_vector.shape)
    print("Data type:", first_vector.dtype)
    print("Gia tri NaN:", np.isnan(first_vector).any())
    print("Gia tri Inf:", np.isinf(first_vector).any())

    print("\n===== THONG TIN INDEX =====")
    print(index_df.head())

    print("\nKIEM TRA HOAN TAT!")

if __name__ == "__main__":
    main()
