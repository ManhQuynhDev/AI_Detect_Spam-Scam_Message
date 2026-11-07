
import sys
import torch
import pandas as pd
import transformers

print("=" * 50)
print("KIỂM TRA MÔI TRƯỜNG DỰ ÁN SMS ANALYSIS")
print("=" * 50)

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Pandas:", pd.__version__)
print("Transformers:", transformers.__version__)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Đang sử dụng CPU")

print("=" * 50)
print("MÔI TRƯỜNG ĐÃ SẴN SÀNG")
