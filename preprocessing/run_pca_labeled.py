
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

# 1. Doc PCA
pca_df = pd.read_csv(
    OUTPUT_DIR / "cls_pca_2d.csv"
)

# 2. Doc index Vector
index_df = pd.read_csv(
    OUTPUT_DIR / "pooled_embeddings_index.csv"
)

# 3. Doc du lieu phan doan
segment_df = pd.read_csv(
    OUTPUT_DIR / "segmented_dataset.csv"
)

# 4. Doc nhan SMS goc
original_df = pd.read_csv(
    OUTPUT_DIR / "cleaned_dataset.csv"
)

# Kiem tra cot du lieu
if "message_id" not in original_df.columns:
    raise ValueError("Khong co cot message_id")

if "label" not in original_df.columns:
    raise ValueError("Khong co cot label")

# Lien ket segment voi message_id
data = index_df.merge(
    segment_df[["segment_id", "message_id"]],
    on="segment_id",
    how="left",
    validate="one_to_one"
)

# Lien ket message_id voi label
data = data.merge(
    original_df[["message_id", "label"]],
    on="message_id",
    how="left",
    validate="many_to_one"
)

# Kiem tra du lieu
if data["label"].isna().any():
    raise ValueError("Co doan SMS chua duoc gan nhan")

# Ghep toa do PCA
data = pd.concat(
    [
        pca_df.reset_index(drop=True),
        data[["message_id", "segment_id", "label"]].reset_index(drop=True)
    ],
    axis=1
)

# Thong ke nhan
print("\n===== PHAN BO NHAN =====")
print(data["label"].value_counts().sort_index())

# Ve bieu do
plt.figure(figsize=(12, 8))

for label, group in data.groupby("label"):
    plt.scatter(
        group["PC1"],
        group["PC2"],
        s=9,
        alpha=0.5,
        label=f"Label {label}"
    )

plt.title("PCA Visualization - CLS Pooling by Label")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "cls_pca_labeled.png",
    dpi=300
)

plt.close()

# Luu du lieu phan tich
data.to_csv(
    OUTPUT_DIR / "cls_pca_labeled.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nDa luu cls_pca_labeled.png")
print("Da luu cls_pca_labeled.csv")
