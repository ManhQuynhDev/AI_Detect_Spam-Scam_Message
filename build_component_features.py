
from pathlib import Path
import pandas as pd

# =========================
# 1. CONFIG
# =========================

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "output"

CLEANED_FILE = OUTPUT_DIR / "cleaned_dataset.csv"
ANALYSIS_FILE = OUTPUT_DIR / "number_occurrence_analysis.csv"

OUTPUT_FILE = OUTPUT_DIR / "number_component_features.csv"

# =========================
# 2. LOAD DATA
# =========================

print("Loading datasets...")

df_messages = pd.read_csv(CLEANED_FILE)
df_occurrences = pd.read_csv(ANALYSIS_FILE)

required_message_cols = {"message_id", "label", "text_clean"}
required_occurrence_cols = {
    "message_id",
    "predicted_component"
}

if not required_message_cols.issubset(df_messages.columns):
    raise ValueError("Missing columns in cleaned_dataset.csv")

if not required_occurrence_cols.issubset(df_occurrences.columns):
    raise ValueError("Missing columns in number_occurrence_analysis.csv")

print(f"Messages: {len(df_messages)}")
print(f"Occurrences: {len(df_occurrences)}")

# =========================
# 3. COUNT COMPONENTS
# =========================

component_counts = pd.crosstab(
    df_occurrences["message_id"],
    df_occurrences["predicted_component"]
)

component_counts = component_counts.add_prefix("count_")

# =========================
# 4. CREATE MESSAGE FEATURES
# =========================

features = df_messages[
    ["message_id", "label", "text_clean"]
].copy()

features = features.merge(
    component_counts,
    on="message_id",
    how="left"
)

count_columns = [
    col for col in features.columns
    if col.startswith("count_")
]

features[count_columns] = (
    features[count_columns]
    .fillna(0)
    .astype(int)
)

# =========================
# 5. CREATE PRESENCE FEATURES
# =========================

for col in count_columns:
    component_name = col.replace("count_", "")

    features[f"has_{component_name}"] = (
        features[col] > 0
    ).astype(int)

# =========================
# 6. AGGREGATE FEATURES
# =========================

features["total_components"] = (
    features[count_columns].sum(axis=1)
)

if "count_NUMBER_UNKNOWN" in features.columns:
    features["unknown_components"] = (
        features["count_NUMBER_UNKNOWN"]
    )
else:
    features["unknown_components"] = 0

features["known_components"] = (
    features["total_components"]
    - features["unknown_components"]
)

# =========================
# 7. SAVE
# =========================

features.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nFeature engineering completed!")
print(f"Output: {OUTPUT_FILE}")
print(f"Rows: {len(features)}")
print(f"Columns: {len(features.columns)}")

print("\nFeature columns:")
for col in features.columns:
    print("-", col)

print("\nPreview:")
print(features.head())
