
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

METHODS = {
    "mean": "mean_pooling.npy",
    "cls": "cls_pooling.npy"
}


def run_pca(method, filename):

    print(f"\nDANG PCA: {method.upper()}")

    vectors = np.load(OUTPUT_DIR / filename)

    print("Du lieu goc:", vectors.shape)

    # PCA tren Vector goc
    pca = PCA(n_components=2)

    reduced = pca.fit_transform(vectors)

    explained_variance = pca.explained_variance_ratio_

    print("Du lieu sau PCA:", reduced.shape)
    print("PC1:", explained_variance[0])
    print("PC2:", explained_variance[1])

    print(
        "Tong phuong sai giai thich:",
        explained_variance.sum()
    )

    # Luu du lieu PCA
    pca_df = pd.DataFrame({
        "PC1": reduced[:, 0],
        "PC2": reduced[:, 1]
    })

    pca_df.to_csv(
        OUTPUT_DIR / f"{method}_pca_2d.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Ve bieu do
    plt.figure(figsize=(10, 7))

    plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        s=8,
        alpha=0.5
    )

    plt.title(f"PCA Visualization - {method.upper()} Pooling")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{method}_pca_2d.png",
        dpi=300
    )

    plt.close()

    return explained_variance.sum()


def main():

    results = {}

    for method, filename in METHODS.items():
        results[method] = run_pca(method, filename)

    print("\n===== PCA SUMMARY =====")

    for method, variance in results.items():
        print(
            f"{method.upper()}: "
            f"{variance:.4f} ({variance * 100:.2f}%)"
        )


if __name__ == "__main__":
    main()
