
import re
import pandas as pd


# =====================================================
# 1. CÁC BIỂU THỨC NHẬN DIỆN
# =====================================================

PATTERNS = {
    "url": re.compile(
        r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
        re.IGNORECASE
    ),

    "date": re.compile(
        r"\[DATE\]|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        re.IGNORECASE
    ),

    "time": re.compile(
        r"\[TIME\]|\b\d{1,2}:\d{2}(?::\d{2})?\b",
        re.IGNORECASE
    ),

    "money": re.compile(
        r"\[MONEY\]|"
        r"\b\d{1,3}(?:[.,]\d{3})+(?:\s?(?:đ|đồng|vnd))?\b|"
        r"\b\d+\s?(?:đ|đồng|vnd)\b",
        re.IGNORECASE
    ),

    "phone": re.compile(
        r"(?<!\d)(?:\+84|0)(?:\d[\s.-]?){8,10}(?!\d)"
    )
}


# =====================================================
# 2. NHẬN DIỆN OTP THEO NGỮ CẢNH
# =====================================================

OTP_PATTERN = re.compile(
    r"(?:mã\s*)?otp"
    r"(?:\s+xác\s+thực)?"
    r"(?:\s+(?:là|:))?"
    r"\s*[:\-]?\s*(\d{4,8})",
    re.IGNORECASE
)


# =====================================================
# 3. NHẬN DIỆN BRAND / SENDER
# =====================================================

BRANDS = [
    "Viettel",
    "Vinaphone",
    "Vietnamobile",
    "Mobifone",
    "VCB",
    "Vietcombank",
    "BIDV",
    "Techcombank",
    "Agribank",
    "MB Bank",
    "VPBank",
    "TPBank",
    "Shopee",
    "Lazada",
    "MoMo",
    "ZaloPay"
]


# =====================================================
# 4. NHẬN DIỆN HÀNH ĐỘNG
# =====================================================

ACTIONS = [
    "xác thực",
    "xác minh",
    "cập nhật",
    "đăng nhập",
    "chuyển khoản",
    "giao dịch",
    "thanh toán",
    "nhận tiền",
    "trúng thưởng",
    "nạp tiền",
    "rút tiền",
    "khóa tài khoản",
    "mở tài khoản",
    "xác nhận",
    "truy cập",
    "cung cấp"
]


# =====================================================
# 5. HÀM TRÍCH XUẤT THÀNH PHẦN
# =====================================================

def extract_components(text):

    if pd.isna(text):
        text = ""

    text = str(text)

    result = {}

    # Trích xuất URL
    result["url"] = PATTERNS["url"].findall(text)

    # Trích xuất ngày tháng
    result["date"] = PATTERNS["date"].findall(text)

    # Trích xuất thời gian
    result["time"] = PATTERNS["time"].findall(text)

    # Trích xuất số tiền
    result["money"] = PATTERNS["money"].findall(text)

    # Trích xuất số điện thoại
    result["phone"] = PATTERNS["phone"].findall(text)

    # Trích xuất OTP
    otp_matches = OTP_PATTERN.findall(text)

    result["otp"] = otp_matches

    # Nhận diện thương hiệu
    result["brand"] = [
        brand for brand in BRANDS
        if re.search(
            rf"(?<!\w){re.escape(brand)}(?!\w)",
            text,
            re.IGNORECASE
        )
    ]

    # Nhận diện hành động
    result["action"] = [
        action for action in ACTIONS
        if re.search(
            re.escape(action),
            text,
            re.IGNORECASE
        )
    ]

    return result
