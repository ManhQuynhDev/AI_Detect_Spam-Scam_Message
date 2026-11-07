
from pathlib import Path
import numpy as np
import pandas as pd

# =====================================
# 1. CONFIG
# =====================================

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "output"

DATA_FILE = OUTPUT_DIR / "final_feature_dataset.npz"
METADATA_FILE = OUTPUT_DIR / "final_feature_metadata.csv"

# =====================================
# 2. LOAD DATA
# =====================================

print("=" * 55)
print("FINAL FEATURE DATASET VALIDATION")
print("=" * 55)

data = np.load(DATA_FILE, allow_pickle=True)

X = data["X"]
y = data["y"]
embedding = data["embedding"]
numeric_features = data["numeric_features"]
feature_columns = data["feature_columns"]
message_ids = data["message_ids"]

metadata = pd.read_csv(METADATA_FILE)

# =====================================
# 3. BASIC SHAPE CHECK
# =====================================

print("\n[1] DATASET SHAPE")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Embedding shape:", embedding.shape)
print("Numeric features shape:", numeric_features.shape)
print("Metadata rows:", len(metadata))

assert X.shape[0] == 2991
assert X.shape[1] == 808
assert embedding.shape == (2991, 768)
assert numeric_features.shape == (2991, 40)
assert len(y) == len(message_ids) == len(metadata)

print("Shape validation: PASSED")

# =====================================
# 4. MISSING AND INFINITE VALUES
# =====================================

print("\n[2] DATA QUALITY")

print("NaN values:", np.isnan(X).sum())
print("Infinite values:", np.isinf(X).sum())

print("Zero embedding rows:", np.sum(
    np.linalg.norm(embedding, axis=1) == 0
))

print("Zero feature rows:", np.sum(
    np.linalg.norm(X, axis=1) == 0
))

assert np.isfinite(X).all()

print("Finite value validation: PASSED")

# =====================================
# 5. LABEL DISTRIBUTION
# =====================================

print("\n[3] LABEL DISTRIBUTION")

label_counts = pd.Series(y).value_counts().sort_index()

print(label_counts)

for label, count in label_counts.items():
    print(
        f"Label {label}: {count} "
        f"({count / len(y) * 100:.2f}%)"
    )

# =====================================
# 6. FEATURE VARIANCE
# =====================================

print("\n[4] FEATURE VARIANCE")

variance = np.var(X, axis=0)

zero_variance = np.sum(variance == 0)
low_variance = np.sum(variance < 1e-8)

print("Zero variance features:", zero_variance)
print("Near-zero variance features:", low_variance)

# Numeric feature statistics
numeric_variance = np.var(numeric_features, axis=0)

variance_report = pd.DataFrame({
    "feature": feature_columns,
    "variance": numeric_variance
}).sort_values("variance")

variance_report.to_csv(
    OUTPUT_DIR / "feature_variance_report.csv",
    index=False,
    encoding="utf-8-sig"
)

# =====================================
# 7. MESSAGE ID CONSISTENCY
# =====================================

print("\n[5] MESSAGE ID CONSISTENCY")

metadata_ids = metadata["message_id"].astype(str).to_numpy()
message_ids = message_ids.astype(str)

print(
    "Message IDs match:",
    np.array_equal(metadata_ids, message_ids)
)

assert np.array_equal(metadata_ids, message_ids)

print(
    "Labels match:",
    np.array_equal(
        metadata["label"].to_numpy(),
        y
    )
)

assert np.array_equal(
    metadata["label"].to_numpy(),
    y
)

# =====================================
# 8. SUMMARY
# =====================================

print("\n" + "=" * 55)
print("VALIDATION SUMMARY")
print("=" * 55)

print("Total messages:", X.shape[0])
print("Total dimensions:", X.shape[1])
print("Embedding dimensions:", embedding.shape[1])
print("Numeric dimensions:", numeric_features.shape[1])
print("NaN:", np.isnan(X).sum())
print("Infinite:", np.isinf(X).sum())
print("Zero variance:", zero_variance)

print("\nValidation completed.")

