
import re
import unicodedata
import pandas as pd



def normalize_unicode(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Chuẩn hóa Unicode NFC
    text = unicodedata.normalize("NFC", text)

    return text

test_text = "Thông báo giao dịch"

result = normalize_unicode(test_text)

print("Trước:", test_text)
print("Sau:", result)



def clean_whitespace(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Xóa ký tự Unicode vô hình
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    # Chuẩn hóa các loại khoảng trắng
    text = text.replace("\xa0", " ")
    text = text.replace("\t", " ")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    # Gộp nhiều khoảng trắng liên tiếp thành một
    text = re.sub(r"\s+", " ", text)

    return text.strip()


test_text = "  Ma OTP   xac thuc \n GD   la 066595  "

print(clean_whitespace(test_text))


PLACEHOLDERS = {
    "time": "[TIME]",
    "money": "[MONEY]",
    "otp": "[OTP]",
    "url": "[URL]",
    "phone": "[PHONE]"
}


def normalize_placeholders(text):
    if not text:
        return ""

    # Chuẩn hóa tên placeholder không phân biệt hoa thường
    for name, placeholder in PLACEHOLDERS.items():
        pattern = rf"\[{name}\]"
        text = re.sub(
            pattern,
            placeholder,
            text,
            flags=re.IGNORECASE
        )

    return text

test_text = "Giao dich thanh cong [time], so tien [money]"

print(normalize_placeholders(test_text))


def clean_sms_text(text):
    text = normalize_unicode(text)

    text = clean_whitespace(text)

    text = normalize_placeholders(text)

    return text

sample = """
    Ma OTP   xac thuc GD la 066595,
    hieu luc [time].
"""

cleaned = clean_sms_text(sample)

print("KẾT QUẢ SAU LÀM SẠCH:")
print(cleaned)