
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_FILE = "output/final_feature_dataset.npz"
METADATA_FILE = "output/final_feature_metadata.csv"
OUTPUT_DIR = "output/splits"

RANDOM_STATE = 42
TRAIN_SIZE = 0.60
VALIDATION_SIZE = 0.20
TEST_SIZE = 0.20

# ==========================================
# LOAD DATA
# ==========================================

print("=" * 55)
print("SPLITTING FINAL FEATURE DATASET")
print("=" * 55)

os.makedirs(OUTPUT_DIR, exist_ok=True)

data = np.load(INPUT_FILE, allow_pickle=False)

X = data["X"]
y = data["y"]
embedding = data["embedding"]
numeric_features = data["numeric_features"]
message_ids = data["message_ids"].astype(str)
feature_columns = data["feature_columns"].astype(str)

metadata = pd.read_csv(METADATA_FILE)

# ==========================================
# VALIDATE INPUT
# ==========================================

n_samples = len(y)

assert X.shape[0] == n_samples
assert embedding.shape[0] == n_samples
assert numeric_features.shape[0] == n_samples
assert len(message_ids) == n_samples
assert len(metadata) == n_samples

assert np.array_equal(
    message_ids,
    metadata["message_id"].astype(str).to_numpy()
)

assert np.array_equal(
    y,
    metadata["label"].to_numpy()
)

assert np.isclose(
    TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE,
    1.0
)

print(f"Total samples: {n_samples}")
print(f"Feature dimensions: {X.shape[1]}")

# ==========================================
# SPLIT TRAIN AND TEMPORARY DATA
# ==========================================

all_indices = np.arange(n_samples)

train_indices, temp_indices = train_test_split(
    all_indices,
    test_size=VALIDATION_SIZE + TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# ==========================================
# SPLIT VALIDATION AND TEST
# ==========================================

relative_test_size = TEST_SIZE / (
    VALIDATION_SIZE + TEST_SIZE
)

validation_indices, test_indices = train_test_split(
    temp_indices,
    test_size=relative_test_size,
    random_state=RANDOM_STATE,
    stratify=y[temp_indices]
)

# ==========================================
# SAVE SPLIT DATA
# ==========================================

splits = {
    "train": train_indices,
    "validation": validation_indices,
    "test": test_indices
}

for split_name, indices in splits.items():

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{split_name}_dataset.npz"
    )

    np.savez_compressed(
        output_path,
        X=X[indices],
        y=y[indices],
        embedding=embedding[indices],
        numeric_features=numeric_features[indices],
        message_ids=message_ids[indices],
        feature_columns=feature_columns
    )

    print(
        f"\n{split_name.upper()}: "
        f"{len(indices)} samples"
    )

    unique, counts = np.unique(
        y[indices],
        return_counts=True
    )

    for label, count in zip(unique, counts):
        print(
            f"  Label {label}: {count} "
            f"({count / len(indices) * 100:.2f}%)"
        )

    print(f"  Saved: {output_path}")

# ==========================================
# SAVE SPLIT INDEX REPORT
# ==========================================

split_assignment = np.full(
    n_samples,
    "unassigned",
    dtype="<U12"
)

for split_name, indices in splits.items():
    split_assignment[indices] = split_name

split_report = metadata.copy()
split_report["split"] = split_assignment

split_report.to_csv(
    os.path.join(OUTPUT_DIR, "split_report.csv"),
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# FINAL CHECK
# ==========================================

assert len(train_indices) + len(validation_indices) + len(test_indices) == n_samples

assert len(np.unique(np.concatenate([
    train_indices,
    validation_indices,
    test_indices
]))) == n_samples

print("\n" + "=" * 55)
print("DATA SPLITTING COMPLETED!")
print("=" * 55)

print(f"Train:      {len(train_indices)}")
print(f"Validation: {len(validation_indices)}")
print(f"Test:       {len(test_indices)}")
print(f"Total:      {n_samples}")

print(f"\nOutput directory: {OUTPUT_DIR}")
