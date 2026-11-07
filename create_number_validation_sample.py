from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "output"

INPUT_FILE = OUTPUT_DIR / "number_occurrence_analysis.csv"
OUTPUT_FILE = OUTPUT_DIR / "number_occurrence_validation_sample.csv"

RANDOM_STATE = 42

# Số dòng lấy mẫu cho mỗi label trong từng nhóm.
SAMPLE_PER_LABEL = {
    "NUMBER_UNKNOWN": 30,
    "CONTACT": 20,
    "YEAR": 20,
    "SERVICE_QUOTA": 20,
    "SHORTCODE": 20,
}

# Các component cần kiểm định trước vì vừa có thay đổi hoặc có nguy cơ
# nhầm lẫn ngữ cảnh.
FOCUS_COMPONENTS = list(SAMPLE_PER_LABEL.keys())


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {INPUT_FILE}. "
            "Hãy chạy analyze_number_occurrences.py trước."
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "message_id", "label", "predicted_component",
        "context_before", "context_after", "text_clean"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"File đầu vào thiếu cột: {sorted(missing)}")

    samples = []

    for component in FOCUS_COMPONENTS:
        group = df[df["predicted_component"] == component]

        for label_value in sorted(group["label"].dropna().unique()):
            subset = group[group["label"] == label_value]
            n = min(SAMPLE_PER_LABEL[component], len(subset))

            if n == 0:
                continue

            samples.append(
                subset.sample(n=n, random_state=RANDOM_STATE)
            )

    if not samples:
        raise ValueError("Không tìm thấy dữ liệu phù hợp để lấy mẫu.")

    audit = pd.concat(samples, ignore_index=True)
    audit = audit.drop_duplicates(
        subset=["message_id", "occurrence_index"]
    )

    # Cột để người kiểm định nhập nhãn thực tế.
    audit["actual_component"] = ""
    audit["is_correct"] = ""
    audit["validation_note"] = ""

    preferred_columns = [
        "message_id",
        "label",
        "occurrence_index",
        "predicted_component",
        "actual_component",
        "is_correct",
        "validation_note",
        "context_before",
        "number_placeholder",
        "context_after",
        "text_clean",
    ]
    available_columns = [c for c in preferred_columns if c in audit.columns]
    audit = audit[available_columns]

    audit = audit.sort_values(
        ["predicted_component", "label", "message_id"]
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("NUMBER COMPONENT VALIDATION SAMPLE")
    print("=" * 60)
    print(f"Tổng số dòng được lấy mẫu: {len(audit)}")
    print("\nSố mẫu theo component và label:")
    print(
        audit.groupby(["predicted_component", "label"])
        .size()
        .to_string()
    )
    print(f"\nĐã lưu file: {OUTPUT_FILE}")
    print("\nHướng dẫn:")
    print("- actual_component: nhập component đúng sau khi đọc ngữ cảnh.")
    print("- is_correct: nhập YES nếu predicted_component đúng, NO nếu sai.")
    print("- validation_note: ghi lý do nếu dự đoán sai hoặc còn mơ hồ.")
    print("- Không chắc chắn thì ghi NEED_REVIEW, không đoán nhãn.")


if __name__ == "__main__":
    main()
