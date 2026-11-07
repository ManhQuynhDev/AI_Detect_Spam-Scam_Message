
from pprint import pprint

from .component_extraction import extract_components


samples = [
    """
    Ma OTP xac thuc GD la 066595,
    hieu luc [TIME].
    Chi tiet giao dich VCB Digibank.
    """,

    """
    Quý khách vui lòng truy cập
    https://viettel.vn/doigiayto
    để cập nhật thông tin.
    Tặng [MONEY] trước [DATE].
    """,

    """
    Xác thực NGAY để không bị TẠM NGỪNG dịch vụ!
    Truy cập https://viettel.vn/xacthucTB
    hoặc liên hệ 0987654321.
    """
]


for index, message in enumerate(samples, start=1):

    print("\n" + "=" * 60)
    print(f"TIN NHẮN {index}")
    print("=" * 60)

    result = extract_components(message)

    pprint(result, sort_dicts=False)
