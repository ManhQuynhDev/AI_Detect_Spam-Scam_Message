from pathlib import Path
import re
import unicodedata
import pandas as pd

# =========================================================
# 1. CONFIGURATION
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

INPUT_FILE = OUTPUT_DIR / "cleaned_dataset.csv"
TAXONOMY_FILE = OUTPUT_DIR / "component_taxonomy.csv"

OUTPUT_FILE = OUTPUT_DIR / "number_occurrence_analysis.csv"
SUMMARY_FILE = OUTPUT_DIR / "number_occurrence_summary.csv"

CONTEXT_SIZE = 120
NUMBER_PATTERN = re.compile(r"\[NUMBER\]", re.IGNORECASE)


# =========================================================
# 2. TEXT NORMALIZATION
# =========================================================

def normalize_text(text):
    """Lowercase, bỏ dấu tiếng Việt và chuẩn hóa khoảng trắng."""
    text = str(text).lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"\s+", " ", text).strip()


def has_any(patterns, text):
    return any(re.search(pattern, text) for pattern in patterns)


# =========================================================
# 3. CLASSIFICATION RULES
# =========================================================

def classify_number(text, start, end):
    """
    Phân loại một [NUMBER] theo ngữ cảnh cục bộ và toàn bộ tin nhắn.
    Chỉ gán nhãn khi có dấu hiệu tương đối rõ; còn lại là NUMBER_UNKNOWN.
    """
    before = text[max(0, start - CONTEXT_SIZE):start]
    after = text[end:min(len(text), end + CONTEXT_SIZE)]

    before_n = normalize_text(before)
    after_n = normalize_text(after)
    full_n = normalize_text(text)

    # Ngữ cảnh trước/sau điểm [NUMBER], dùng cho mẫu nằm hai phía.
    local = before_n + " [number] " + after_n

    # 1. URL_COMPONENT
    # Số nằm trong URL hoặc sau dấu / của đường dẫn.
    if re.search(r"(https?://|www\.)[^\s]*$", before_n):
        return "URL_COMPONENT"
    if re.search(r"(https?://|www\.)[^\s]*$", before_n) and re.match(
        r"^[^\s]*(\.|/|%[0-9a-f]{2})", after_n
    ):
        return "URL_COMPONENT"
    if re.search(r"https?://[^\s]*/$", before_n):
        return "URL_COMPONENT"

    # 2. OTP
    otp_before = [
        r"\bma\s+otp\b",
        r"\bma\s+xac\s+thuc\b",
        r"\bma\s+xac\s+minh\b",
        r"\bma\s+xac\s+nhan\b",
        r"\bma\s+bao\s+mat\b",
        r"\bverification\s+code\b",
        r"\bsecurity\s+code\b",
        r"\bauthenticator\s*[:=]?\s*$",
        # Các tin nhắn yêu cầu đọc mã cho nhân viên ngân hàng.
        r"\b(doc|doc lai)\s+ma\s*$",
    ]
    otp_after = [
        r"^\s*(la\s+ma\s+otp|la\s+ma\s+xac\s+thuc|la\s+ma\s+xac\s+minh|la\s+ma\s+xac\s+nhan)\b",
        r"^\s*(verification\s+code|security\s+code)\b",
        r"\bma\s+otp\b",
        r"\bma\s+xac\s+thuc\b",
        r"\bma\s+xac\s+minh\b",
        r"\bma\s+xac\s+nhan\b",
    ]
    if has_any(otp_before, before_n) or has_any(otp_after, after_n):
        return "OTP"

    # 3. DOCUMENT_NUMBER
    document_patterns = [
        r"\bso\s+giay\s+to\s*$",
        r"\bso\s+cmnd\s*$",
        r"\bso\s+cccd\s*$",
        r"\bso\s+ho\s+so\s*$",
        r"\bma\s+ho\s+so\s*$",
        r"\bso\s+hop\s+dong\s*$",
        r"\bso\s+van\s+ban\s*$",
        r"\b(thong\s+tu|nghi\s+dinh|quyet\s+dinh|van\s+ban)\s+\d+/\s*$",
    ]
    if has_any(document_patterns, before_n):
        return "DOCUMENT_NUMBER"

    # 4. YEAR
    year_patterns = [
        r"\bthang\s+(0?[1-9]|1[0-2])/\s*$",
        r"\b(nam|giap\s+thin|tet\s+nguyen\s+dan)\s*$",
        r"\b(ngay\s+chuyen\s+doi\s+so\s+quoc\s+gia|khuyen\s+mai\s+tap\s+trung\s+quoc\s+gia)\s*$",
        r"\b(euro|world\s+cup)\s*$",
    ]
    if has_any(year_patterns, before_n):
        return "YEAR"

    # 5. CARD_REFERENCE — ưu tiên trước ACCOUNT_NUMBER.
    card_patterns = [
        r"\bso\s+the\s*$",
        r"\bthe\s+ngan\s+hang\s*$",
        r"\bcard\s+number\s*$",
        r"\bthe\s+(visa|mastercard|atm)\s*$",
        r"\blien\s+ket\s+the\s*$",
        r"\bthe\s+[a-z0-9]+\s*$",
    ]
    if has_any(card_patterns, before_n):
        return "CARD_REFERENCE"

    # 6. ACCOUNT_NUMBER
    account_patterns = [
        r"\bso\s+tai\s+khoan\s*$",
        r"\bstk\s*$",
        r"\bsdtk\s*$",
        r"\btai\s+khoan\s+so\s*$",
        r"\baccount\s+number\s*$",
        r"\bchuyen\s+(tien\s+)?vao\s+(tk|tai\s+khoan)\s*$",
        r"\bvao\s+(tk|tai\s+khoan)\s+(ngan\s+hang\s*)?$",
    ]
    if has_any(account_patterns, before_n):
        return "ACCOUNT_NUMBER"

    # 7. TRANSACTION_ID
    transaction_patterns = [
        r"\bma\s+giao\s+dich\s*$",
        r"\bma\s+gd\s*$",
        r"\btransaction\s+id\s*$",
        r"\bma\s+tham\s+chieu\s*$",
    ]
    if has_any(transaction_patterns, before_n):
        return "TRANSACTION_ID"

    # 8. ORDER_ID
    order_patterns = [
        r"\bma\s+don\s+hang\s*$",
        r"\bso\s+don\s+hang\s*$",
        r"\bdon\s+hang\s+so\s*$",
        r"\bdh\s+cua\s+ban\s+la\s*$",
        r"\border\s+id\s*$",
    ]
    if has_any(order_patterns, before_n):
        return "ORDER_ID"

    # 9. SUBSCRIBER_ID
    subscriber_patterns = [
        r"\bma\s+thue\s+bao\s*$",
        r"\bso\s+thue\s+bao\s*$",
        r"\bsubscriber\s+id\s*$",
        r"\bma\s+khach\s+hang\s*$",
        r"\bcustomer\s+id\s*$",
        r"\bthue\s+bao\s*$",
    ]
    if has_any(subscriber_patterns, before_n):
        return "SUBSCRIBER_ID"

    # 10. SHORTCODE
    # Chỉ nhận diện khi số đứng sau "gửi" trong cú pháp gửi SMS.
    shortcode_patterns = [
        r"\bsoan\b.{0,50}\bgui\s*[,.:;]?\s*$",
        r"\b(gui|send)\s*[,.:;]?\s*$",
        r"\b[1-9]\s+gui\s*$",
    ]
    if has_any(shortcode_patterns, before_n):
        return "SHORTCODE"

    # 11. CONTACT
    contact_patterns = [
        r"\blh\s*[,.:;]?\s*$",
        r"\blien\s+he\s*[,.:;]?\s*$",
        r"\bchi\s+tiet\s+lh\s*[,.:;]?\s*$",
        r"\bhotline\s*[,.:;]?\s*$",
        r"\btong\s+dai\s*[,.:;]?\s*$",
        r"\bgoi\s*[,.:;]?\s*$",
        r"\bso\s+dien\s+thoai\s*[,.:;]?\s*$",
        r"\bdien\s+thoai\s*[,.:;]?\s*$",
        r"\bcskh\s*[,.:;]?\s*$",
        r"\bhotro\s*[,.:;]?\s*$",
        r"\bcall\s*[,.:;]?\s*$",
        r"\bzalo\s*[,.:;]?\s*$",
        r"\btelegram\s*[,.:;]?\s*$",
        r"\bsdt\s*[,.:;]?\s*$",
        r"\bso dt\s*[,.:;]?\s*$",
        r"\bnhan\s+tin\s+qua\s+zalo\s*[,.:;]?\s*$",
    ]
    if has_any(contact_patterns, before_n):
        return "CONTACT"

    # Nếu đây là số thứ hai trong cụm "gọi [NUMBER] hoặc [NUMBER]".
    if re.search(r"\bhoac\s*$", before_n) and re.search(
        r"\bgoi\s+\[number\]", full_n
    ):
        return "CONTACT"

    # Ngữ cảnh toàn tin nhắn: có nhiều số liên hệ được nối bằng "hoặc".
    if re.search(r"\b(goi|lien he|hotline)\b", full_n) and re.search(
        r"\[number\].{0,80}\bhoac\b.{0,40}\[number\]", full_n
    ):
        return "CONTACT"

    # 12. RECIPIENT_INFO
    recipient_patterns = [
        r"\bnguoi\s+nhan\s*[:=]?\s*$",
        r"\bnguoi\s+thu\s+huong\s*[:=]?\s*$",
        r"\bchuyen\s+cho\s*$",
        r"\bnhan\s+vao\s+tai\s+khoan\s*$",
    ]
    if has_any(recipient_patterns, before_n):
        return "RECIPIENT_INFO"

    # 13. SERVICE_QUOTA
    # Chỉ nhận dạng đơn vị/dung lượng rõ ràng; không gán theo từ SMS chung.
    service_patterns = [
        r"\b(mb|gb|kb)\s*$",
        r"\b(phut\s+goi|dung\s+luong|luu\s+luong|data)\s*$",
        r"\b(tin\s+nhan|sms)\s+(noi\s+mang|ngoai\s+mang)\s+va\s*$",
    ]
    if has_any(service_patterns, before_n):
        return "SERVICE_QUOTA"

    # 14. DEVICE_ID
    device_patterns = [
        r"\bthiet\s+bi\s+co\s+ten\s*[:=]?\s*$",
        r"\bdevice\s+id\s*$",
        r"\bma\s+thiet\s+bi\s*$",
        r"\bserial\s+number\s*$",
    ]
    if has_any(device_patterns, before_n):
        return "DEVICE_ID"

    return "NUMBER_UNKNOWN"


# =========================================================
# 4. LOAD DATA
# =========================================================

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {INPUT_FILE}")
    if not TAXONOMY_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy taxonomy: {TAXONOMY_FILE}. "
            "Hãy chạy component_taxonomy.py trước."
        )

    df = pd.read_csv(INPUT_FILE)
    required_columns = {"message_id", "label", "text_clean"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset thiếu các cột: {missing}")

    taxonomy_df = pd.read_csv(TAXONOMY_FILE)
    if "component" not in taxonomy_df.columns:
        raise ValueError("Taxonomy phải có cột 'component'.")

    allowed_components = set(
        taxonomy_df["component"].dropna().astype(str).str.strip()
    )
    return df, allowed_components


# =========================================================
# 5. EXTRACT NUMBER OCCURRENCES
# =========================================================

def extract_occurrences(df, allowed_components):
    records = []

    for _, row in df.iterrows():
        text = "" if pd.isna(row["text_clean"]) else str(row["text_clean"])
        matches = list(NUMBER_PATTERN.finditer(text))

        for occurrence_index, match in enumerate(matches, start=1):
            start, end = match.span()
            before = text[max(0, start - CONTEXT_SIZE):start]
            after = text[end:min(len(text), end + CONTEXT_SIZE)]

            component = classify_number(text, start, end)
            if component not in allowed_components:
                component = "NUMBER_UNKNOWN"

            records.append({
                "message_id": row["message_id"],
                "label": row["label"],
                "occurrence_index": occurrence_index,
                "number_placeholder": match.group(),
                "start_position": start,
                "end_position": end,
                "context_before": before,
                "context_after": after,
                "predicted_component": component,
                "text_clean": text,
            })

    return pd.DataFrame(records)


# =========================================================
# 6. SUMMARY
# =========================================================

def create_summary(occurrences):
    columns = [
        "label", "predicted_component",
        "total_occurrences", "unique_messages"
    ]
    if occurrences.empty:
        return pd.DataFrame(columns=columns)

    return (
        occurrences
        .groupby(["label", "predicted_component"], dropna=False)
        .agg(
            total_occurrences=("message_id", "size"),
            unique_messages=("message_id", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["label", "total_occurrences"],
            ascending=[True, False],
        )
    )


# =========================================================
# 7. MAIN
# =========================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("NUMBER OCCURRENCE ANALYSIS")
    print("=" * 65)

    df, allowed_components = load_data()
    print(f"Tổng số tin nhắn: {len(df)}")
    print(f"Số component trong taxonomy: {len(allowed_components)}")

    occurrences = extract_occurrences(df, allowed_components)
    summary = create_summary(occurrences)

    occurrences.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")

    print("\nKẾT QUẢ")
    print("-" * 65)
    print(f"Tổng số lần xuất hiện [NUMBER]: {len(occurrences)}")
    message_count = (
        occurrences["message_id"].nunique()
        if not occurrences.empty else 0
    )
    print(f"Số tin nhắn có [NUMBER]: {message_count}")

    if not occurrences.empty:
        unknown_count = (
            occurrences["predicted_component"] == "NUMBER_UNKNOWN"
        ).sum()
        unknown_ratio = unknown_count / len(occurrences) * 100
        classified_count = len(occurrences) - unknown_count

        print(f"Số NUMBER_UNKNOWN: {unknown_count}")
        print(f"Tỷ lệ NUMBER_UNKNOWN: {unknown_ratio:.2f}%")
        print(f"Số occurrence đã phân loại: {classified_count}")
        print(f"Tỷ lệ đã phân loại: {100 - unknown_ratio:.2f}%")
        print("\nTHỐNG KÊ THEO COMPONENT")
        print(summary.to_string(index=False))

    print("\nĐã lưu kết quả:")
    print(f"- {OUTPUT_FILE}")
    print(f"- {SUMMARY_FILE}")
    print("\nLưu ý: predicted_component là nhãn dự đoán theo quy tắc,")
    print("không phải nhãn đã được kiểm chứng thủ công hay độ chính xác mô hình.")


if __name__ == "__main__":
    main()
