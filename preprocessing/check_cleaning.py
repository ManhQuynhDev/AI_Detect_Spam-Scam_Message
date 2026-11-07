
import pandas as pd

df = pd.read_csv("output/cleaned_dataset.csv")

print("===== KIỂM TRA DỮ LIỆU =====")

print("Số lượng bản ghi:", len(df))

print("\nSố dữ liệu trống:")
print(df["text_clean"].isna().sum())

print("\nSố tin nhắn rỗng:")
print((df["text_clean"].astype(str).str.strip() == "").sum())

print("\nSố nội dung trùng lặp:")
print(df["text_clean"].duplicated().sum())

print("\nĐộ dài trung bình tin nhắn:")
print(df["text_clean"].str.len().mean())

print("\n===== 10 MẪU DỮ LIỆU =====")

print(
    df[["text_raw", "text_clean"]]
    .head(10)
    .to_string(index=False)
)
