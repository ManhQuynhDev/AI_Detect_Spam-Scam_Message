
import os
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_DIR = "output/splits"
OUTPUT_DIR = "output/scaled_splits"

SCALER_PATH = os.path.join(
    OUTPUT_DIR,
    "numeric_scaler.joblib"
)

SPLIT_NAMES = ["train", "validation", "test"]

# ==========================================
# LOAD DATA
# ==========================================

print("=" * 55)
print("FEATURE SCALING")
print("=" * 55)

os.makedirs(OUTPUT_DIR, exist_ok=True)

datasets = {}

for split_name in SPLIT_NAMES:

    file_path = os.path.join(
        INPUT_DIR,
        f"{split_name}_dataset.npz"
    )

    data = np.load(file_path, allow_pickle=False)

    datasets[split_name] = {
        "X": data["X"],
        "y": data["y"],
        "embedding": data["embedding"],
        "numeric_features": data["numeric_features"],
        "message_ids": data["message_ids"],
        "feature_columns": data["feature_columns"]
    }

    print(
        f"{split_name}: "
        f"{datasets[split_name]['X'].shape}"
    )

# ==========================================
# FIT SCALER ON TRAIN ONLY
# ==========================================

X_train_numeric = datasets["train"]["numeric_features"]

scaler = StandardScaler()

scaler.fit(X_train_numeric)

print("\nScaler fitted on TRAIN only.")

# ==========================================
# TRANSFORM ALL SPLITS
# ==========================================

for split_name in SPLIT_NAMES:

    data = datasets[split_name]

    embedding = data["embedding"]
    numeric = data["numeric_features"]

    numeric_scaled = scaler.transform(numeric)

    # Keep mBERT embeddings unchanged.
    X_combined = np.concatenate(
        [embedding, numeric_scaled],
        axis=1
    ).astype(np.float32)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{split_name}_dataset_scaled.npz"
    )

    np.savez_compressed(
        output_path,
        X=X_combined,
        y=data["y"],
        embedding=embedding,
        numeric_features=numeric_scaled,
        message_ids=data["message_ids"],
        feature_columns=data["feature_columns"]
    )

    print(f"\n{split_name.upper()}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Numeric shape: {numeric_scaled.shape}")
    print(f"Combined shape: {X_combined.shape}")
    print(f"Saved: {output_path}")

# ==========================================
# SAVE SCALER
# ==========================================

joblib.dump(scaler, SCALER_PATH)

print("\nScaler saved:", SCALER_PATH)

# ==========================================
# VALIDATE RESULTS
# ==========================================

train_scaled = datasets["train"]["numeric_features"]
train_transformed = scaler.transform(train_scaled)

print("\nTRAIN SCALING CHECK")
print("Mean (approx. 0):", np.mean(train_transformed))
print("Std (approx. 1):", np.std(train_transformed))

for split_name in SPLIT_NAMES:

    file_path = os.path.join(
        OUTPUT_DIR,
        f"{split_name}_dataset_scaled.npz"
    )

    data = np.load(file_path, allow_pickle=False)

    X = data["X"]

    assert X.shape[1] == 808
    assert np.isfinite(X).all()

print("\n" + "=" * 55)
print("FEATURE SCALING COMPLETED!")
print("=" * 55)
