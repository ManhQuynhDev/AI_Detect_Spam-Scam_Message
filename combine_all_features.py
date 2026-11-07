
from pathlib import Path
import numpy as np
import pandas as pd

# =====================================
# 1. CONFIG
# =====================================

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "output"

FEATURE_FILE = OUTPUT_DIR / "combined_message_features.csv"
EMBEDDING_FILE = OUTPUT_DIR / "contextual_embeddings.npz"
INDEX_FILE = OUTPUT_DIR / "embedding_index.csv"

OUTPUT_FILE = OUTPUT_DIR / "final_feature_dataset.npz"
METADATA_FILE = OUTPUT_DIR / "final_feature_metadata.csv"

# =====================================
# 2. LOAD FEATURES
# =====================================

print("Loading message features...")

df = pd.read_csv(FEATURE_FILE)

required = {"message_id", "label", "text_clean"}

if not required.issubset(df.columns):
    raise ValueError("Missing required feature columns")

if df["message_id"].duplicated().any():
    raise ValueError("message_id is not unique in feature dataset")

print("Messages:", len(df))

# =====================================
# 3. LOAD EMBEDDINGS
# =====================================

print("\nLoading embeddings...")

embedding_data = np.load(EMBEDDING_FILE)

embedding_index = pd.read_csv(INDEX_FILE)

print("Embedding index rows:", len(embedding_index))

# =====================================
# 4. POOL TOKEN EMBEDDINGS
# =====================================

print("\nPooling embeddings by message...")

# Map segment_id -> message_id
segment_info = pd.read_csv(
    OUTPUT_DIR / "segmented_dataset.csv"
)

segment_to_message = segment_info[
    ["segment_id", "message_id"]
].drop_duplicates()

index = embedding_index.merge(
    segment_to_message,
    on="segment_id",
    how="left",
    validate="many_to_one"
)

if index["message_id"].isna().any():
    raise ValueError("Some embeddings cannot be mapped to message_id")

# Check expected embedding keys
if "embedding_key" not in index.columns:
    raise ValueError("Missing embedding_key in embedding_index.csv")

message_vectors = {}

for message_id, group in index.groupby("message_id"):
    token_vectors = []

    for key in group["embedding_key"]:
        if key not in embedding_data.files:
            raise KeyError(f"Missing embedding key: {key}")

        vectors = embedding_data[key]

        if vectors.ndim != 2 or vectors.shape[1] != 768:
            raise ValueError(f"Invalid embedding shape: {key}")

        token_vectors.append(vectors)

    if token_vectors:
        all_tokens = np.concatenate(token_vectors, axis=0)
        message_vectors[message_id] = all_tokens.mean(axis=0)

# =====================================
# 5. CHECK MISSING EMBEDDINGS
# =====================================

missing_ids = [
    mid for mid in df["message_id"]
    if mid not in message_vectors
]

if missing_ids:
    raise ValueError(
        f"Missing embeddings for {len(missing_ids)} messages. "
        f"Examples: {missing_ids[:5]}"
    )

embedding_matrix = np.vstack([
    message_vectors[mid]
    for mid in df["message_id"]
]).astype(np.float32)

print("Embedding matrix:", embedding_matrix.shape)

# =====================================
# 6. SELECT NUMERIC FEATURES
# =====================================

excluded_columns = {
    "message_id",
    "label",
    "text_clean"
}

feature_columns = [
    col for col in df.columns
    if col not in excluded_columns
    and pd.api.types.is_numeric_dtype(df[col])
]

numeric_matrix = (
    df[feature_columns]
    .fillna(0)
    .to_numpy(dtype=np.float32)
)

print("Numeric feature matrix:", numeric_matrix.shape)

# =====================================
# 7. COMBINE FEATURES
# =====================================

X = np.concatenate(
    [embedding_matrix, numeric_matrix],
    axis=1
)

y = df["label"].to_numpy(dtype=np.int64)

if not np.isfinite(X).all():
    raise ValueError("Combined matrix contains NaN or infinity")

# =====================================
# 8. SAVE
# =====================================

np.savez_compressed(
    OUTPUT_FILE,
    X=X,
    y=y,
    embedding=embedding_matrix,
    numeric_features=numeric_matrix,
    feature_columns=np.array(feature_columns),
    message_ids=df["message_id"].astype(str).to_numpy(dtype=str)
)

df[["message_id", "label"]].to_csv(
    METADATA_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nFeature combination completed!")
print("Messages:", X.shape[0])
print("Total dimensions:", X.shape[1])
print("Embedding dimensions:", embedding_matrix.shape[1])
print("Numeric dimensions:", numeric_matrix.shape[1])
print("Output:", OUTPUT_FILE)
print("Metadata:", METADATA_FILE)
