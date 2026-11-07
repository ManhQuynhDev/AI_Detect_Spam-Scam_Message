# README -- Hệ thống AI phát hiện và phân loại tin nhắn Spam/Scam

## 1. Giới thiệu

Dự án xây dựng bộ dữ liệu đặc trưng phục vụ huấn luyện các mô hình
Machine Learning phân loại tin nhắn SMS. Pipeline hiện tại kết hợp:

-   **Embedding ngữ cảnh mBERT:** 768 chiều cho mỗi tin nhắn.
-   **Đặc trưng thủ công (feature engineering):** 40 chiều, gồm đặc
    trưng thành phần và đặc trưng số/ngữ cảnh.
-   **Tổng cộng:** 808 đặc trưng cho mỗi tin nhắn.

Dataset sau khi xử lý có **2.991 tin nhắn**. Phân bố nhãn hiện có:

  Nhãn          Số lượng      Tỷ lệ
  ---------- ----------- ----------
  0                2.193     73,32%
  1                  798     26,68%
  **Tổng**     **2.991**   **100%**

> **Lưu ý về nhãn:** Trước khi huấn luyện và diễn giải các chỉ số
> Spam/Ham, cần đối chiếu tài liệu hoặc mã nguồn dataset gốc để xác nhận
> chính xác `0` và `1` đại diện cho lớp nào. Không suy luận ý nghĩa lớp
> chỉ từ số lượng mẫu.

## 2. Cấu trúc thư mục

Cấu trúc dưới đây phản ánh các file đầu ra đã được tạo trong pipeline.
Một số file script trung gian có thể khác tên nếu người phát triển đã
chỉnh sửa trong quá trình thực hiện.

``` text
AI_Detect_Spam-Scam_Message/
│
├── preprocessing/
│   └── component_extraction.py
│
├── output/
│   ├── cleaned_dataset.csv
│   ├── extracted_dataset.csv
│   ├── segmented_dataset.csv
│   ├── contextual_embeddings.npz
│   ├── embedding_index.csv
│   ├── number_occurrence_analysis.csv
│   ├── number_component_features.csv
│   ├── combined_message_features.csv
│   ├── final_feature_dataset.npz
│   ├── final_feature_metadata.csv
│   │
│   ├── splits/
│   │   ├── train_dataset.npz
│   │   ├── validation_dataset.npz
│   │   ├── test_dataset.npz
│   │   └── split_report.csv
│   │
│   └── scaled_splits/
│       ├── train_dataset_scaled.npz
│       ├── validation_dataset_scaled.npz
│       ├── test_dataset_scaled.npz
│       └── numeric_scaler.joblib
│
├── run_extraction.py
├── run_segmentation.py
├── run_tokenization.py
├── run_embeddings.py
├── run_pooling.py
├── run_pca.py
├── run_component_analysis.py
├── build_component_features.py
├── build_numeric_features.py
├── combine_all_features.py
├── validate_final_dataset.py
├── split_dataset.py
├── scale_features.py
│
└── README.md
```

Tên file script ở trên là các tên đã dùng trong quá trình triển khai.
Nếu cấu trúc thực tế có thay đổi, hãy ưu tiên kiểm tra file đang tồn tại
trong project.

## 3. Quy trình xử lý dữ liệu

### Bước 1. Làm sạch dữ liệu (Data Cleaning)

**Đầu vào:** Dataset SMS gốc.

**Đầu ra:** `output/cleaned_dataset.csv`

Các cột chính:

-   `message_id`: mã định danh tin nhắn.
-   `label`: nhãn của tin nhắn.
-   `text_clean`: nội dung sau làm sạch.

Mục tiêu là chuẩn hóa văn bản nhưng vẫn giữ lại thông tin có ích cho
phát hiện Spam/Scam như số điện thoại, OTP, số tiền, URL, thời gian và
các placeholder. Không nên xóa tùy tiện các chuỗi số vì chúng có thể là
dấu hiệu quan trọng.

### Bước 2. Trích xuất thành phần (Component Extraction)

**Đầu vào:** `output/cleaned_dataset.csv`

**Đầu ra:** `output/extracted_dataset.csv`

Các luật/regex được sử dụng để tìm những thành phần có cấu trúc, ví dụ:

-   URL.
-   Ngày tháng và thời gian.
-   Số tiền.
-   Số điện thoại.
-   OTP.
-   Thương hiệu.
-   Hành động hoặc cụm từ hành động.

Đây là bước trích xuất dựa trên luật. Kết quả phát hiện phụ thuộc vào
regex và từ khóa đã định nghĩa; không đồng nghĩa với độ chính xác ngữ
nghĩa của một mô hình NER.

### Bước 3. Phân đoạn tin nhắn (Segmentation)

**Đầu vào:** Dữ liệu đã trích xuất thành phần.

**Đầu ra:** `output/segmented_dataset.csv`

Tin nhắn được tách thành các đoạn nhỏ hơn dựa trên dấu câu và quy tắc
phân đoạn. Pipeline hiện tại tạo ra khoảng **10.023 segments** từ 2.991
tin nhắn.

Lưu ý: phân đoạn theo dấu câu là cách tách đơn giản, không phải phân
tích cú pháp tiếng Việt hoàn chỉnh. Cần chú ý URL, chữ viết tắt và các
chuỗi đặc biệt khi thay đổi quy tắc tách.

### Bước 4. Tokenization bằng mBERT

**Script:** `run_tokenization.py`

Văn bản được chuyển thành token theo tokenizer của
`bert-base-multilingual-cased`, với cấu hình đã sử dụng:

-   `MAX_LENGTH = 128`
-   `truncation = True`
-   `padding = False`

WordPiece của mBERT chia văn bản thành subword/token; đây không phải
công cụ tách từ tiếng Việt theo nghĩa ngôn ngữ học.

### Bước 5. Trích xuất contextual embeddings

**Script:** `run_embeddings.py`

Sử dụng mô hình `bert-base-multilingual-cased` để lấy biểu diễn ngữ cảnh
từ các token. Hidden size của mô hình là **768**.

**Các file đầu ra:**

-   `output/contextual_embeddings.npz`: ma trận embedding theo
    token/segment, lưu dưới dạng các mảng NumPy.
-   `output/embedding_index.csv`: bảng ánh xạ các hàng embedding với
    `segment_id`, `row_index`, `token_count` và `embedding_key`.

Embedding là biểu diễn số của văn bản trong ngữ cảnh. Bản thân embedding
không tự đưa ra nhãn Spam/Ham và cũng không tự động xác định thực thể
OTP, số tiền hay tài khoản.

### Bước 6. Pooling và trực quan hóa

**Script:** `run_pooling.py`

Pooling tổng hợp embedding token thành vector đại diện cho segment, gồm
các cách đã thử như:

-   Mean pooling.
-   CLS pooling.

**Script:** `run_pca.py`

PCA được dùng để giảm chiều phục vụ trực quan hóa/phân tích. PCA 2D
không thay thế vector embedding gốc dùng trong bộ đặc trưng cuối cùng.

### Bước 7. Phân tích và tạo đặc trưng thành phần

**Script:** `run_component_analysis.py`, `build_component_features.py`

Các thành phần số được phân loại theo nhóm ngữ cảnh, ví dụ:

1.  OTP
2.  YEAR
3.  DOCUMENT_NUMBER
4.  SHORTCODE
5.  CONTACT
6.  SERVICE_QUOTA
7.  RECIPIENT_INFO
8.  ACCOUNT_NUMBER
9.  CARD_REFERENCE
10. TRANSACTION_ID
11. ORDER_ID
12. SUBSCRIBER_ID
13. DEVICE_ID
14. URL_COMPONENT
15. NUMBER_UNKNOWN

Các nhóm này là kết quả phân loại theo luật và ngữ cảnh từ khóa.
`NUMBER_UNKNOWN` thể hiện trường hợp chưa được quy tắc hiện tại phân
loại, không có nghĩa chắc chắn là dữ liệu sai.

**Đầu ra:**

-   `output/number_occurrence_analysis.csv`: kết quả phân tích các lần
    xuất hiện thành phần số.
-   `output/number_component_features.csv`: các đặc trưng thành phần
    theo từng tin nhắn.

Bộ đặc trưng thành phần có 31 chiều:

-   15 đặc trưng đếm số lần xuất hiện (`count_*`).
-   13 đặc trưng hiện diện/không hiện diện (`has_*`).
-   3 đặc trưng tổng hợp: `total_components`, `unknown_components`,
    `known_components`.

### Bước 8. Tạo đặc trưng số và ngữ cảnh

**Script:** `build_numeric_features.py`

**Đầu ra:** `output/combined_message_features.csv`

Bổ sung 9 đặc trưng:

-   `numeric_count`
-   `total_numeric_digits`
-   `has_4_to_8_digit_number`
-   `has_6_digit_number`
-   `has_otp_context`
-   `has_money_context`
-   `has_phone_context`
-   `has_account_context`
-   `has_url`

Các đặc trưng này phản ánh dấu hiệu hình thức hoặc từ khóa/ngữ cảnh đơn
giản. Ví dụ, một chuỗi có 6 chữ số không mặc nhiên là OTP; cần xem thêm
ngữ cảnh. Các luật từ khóa có thể bỏ sót cách viết có dấu hoặc biến thể
mới.

### Bước 9. Kết hợp embedding và đặc trưng thủ công

**Script:** `combine_all_features.py`

**Đầu vào:**

-   `output/combined_message_features.csv`
-   `output/contextual_embeddings.npz`
-   `output/embedding_index.csv`
-   `output/segmented_dataset.csv`

Embedding của các segment thuộc cùng một tin nhắn được tổng hợp thành
vector đại diện 768 chiều. Vector này được nối với 40 đặc trưng thủ
công.

**Đầu ra:**

-   `output/final_feature_dataset.npz`
-   `output/final_feature_metadata.csv`

Kích thước đã xác nhận:

  Thành phần                   Số chiều
  -------------------------- ----------
  mBERT embedding                   768
  Component features                 31
  Numeric/context features            9
  **Tổng**                      **808**

### Bước 10. Kiểm tra bộ dữ liệu cuối

**Script:** `validate_final_dataset.py`

Kết quả kiểm tra đã ghi nhận:

-   2.991 mẫu.
-   Ma trận `X`: `(2991, 808)`.
-   `y`: `(2991,)`.
-   Embedding: `(2991, 768)`.
-   Numeric features: `(2991, 40)`.
-   Không có NaN hoặc Inf.
-   Không có hàng embedding bằng 0.
-   Không có đặc trưng phương sai bằng 0.
-   `message_id` và `label` khớp metadata.

### Bước 11. Chia tập dữ liệu

**Script:** `split_dataset.py`

Chia theo tỷ lệ 60% / 20% / 20%, dùng `stratify` để giữ tỷ lệ nhãn tương
tự giữa các tập.

**Đầu ra:** `output/splits/`

  -----------------------------------------------------------------------------
  File                                             Số mẫu Mục đích
  -------------------------- ---------------------------- ---------------------
  `train_dataset.npz`                               1.794 Huấn luyện mô hình

  `validation_dataset.npz`                            598 Chọn mô hình, điều
                                                          chỉnh tham số/ngưỡng

  `test_dataset.npz`                                  599 Đánh giá cuối cùng
  -----------------------------------------------------------------------------

`split_report.csv` lưu metadata cùng thông tin tập dữ liệu được gán
(`train`, `validation`, `test`).

### Bước 12. Chuẩn hóa đặc trưng số

**Script:** `scale_features.py`

Sử dụng `StandardScaler` trên **40 đặc trưng thủ công**:

-   `fit` scaler chỉ trên tập Train.
-   Dùng scaler đã fit để `transform` Validation và Test.
-   Giữ nguyên 768 chiều embedding mBERT.
-   Nối embedding 768 chiều với 40 đặc trưng đã chuẩn hóa để tạo ma trận
    808 chiều.

**Đầu ra cuối cùng để training:** `output/scaled_splits/`

  -----------------------------------------------------------------------
  File                                Nội dung
  ----------------------------------- -----------------------------------
  `train_dataset_scaled.npz`          X và y cho training

  `validation_dataset_scaled.npz`     X và y cho validation

  `test_dataset_scaled.npz`           X và y cho testing

  `numeric_scaler.joblib`             Scaler đã fit trên Train; dùng lại
                                      khi xử lý dữ liệu mới
  -----------------------------------------------------------------------

Kết quả kích thước đã xác nhận:

-   Train: `(1794, 808)`
-   Validation: `(598, 808)`
-   Test: `(599, 808)`

## 4. Ý nghĩa các định dạng file

### CSV

CSV lưu dữ liệu dạng bảng, có thể mở bằng Excel hoặc đọc bằng pandas.
Phù hợp để kiểm tra nội dung, nhãn, mã tin nhắn và các đặc trưng.

### NPZ

NPZ là file lưu nhiều mảng NumPy trong cùng một file nén. Các key hiện
dùng trong bộ dữ liệu đã chia gồm:

-   `X`: ma trận đặc trưng đầu vào, kích thước `(số_mẫu, 808)`.
-   `y`: nhãn, kích thước `(số_mẫu,)`.
-   `embedding`: embedding mBERT 768 chiều.
-   `numeric_features`: 40 đặc trưng thủ công; ở thư mục
    `scaled_splits`, đây là các đặc trưng đã chuẩn hóa.
-   `message_ids`: mã tin nhắn tương ứng.
-   `feature_columns`: tên các cột đặc trưng.

### JOBLIB

`numeric_scaler.joblib` lưu đối tượng `StandardScaler`. Cần dùng đúng
scaler này để biến đổi đặc trưng của dữ liệu mới; không fit một scaler
mới trên dữ liệu production hoặc dữ liệu test.

## 5. Cách đọc dữ liệu để training

Ví dụ đọc ba tập dữ liệu đã chuẩn hóa:

``` python
import numpy as np

DATA_DIR = "output/scaled_splits"

train = np.load(
    f"{DATA_DIR}/train_dataset_scaled.npz",
    allow_pickle=False
)

validation = np.load(
    f"{DATA_DIR}/validation_dataset_scaled.npz",
    allow_pickle=False
)

test = np.load(
    f"{DATA_DIR}/test_dataset_scaled.npz",
    allow_pickle=False
)

X_train = train["X"]
y_train = train["y"]

X_val = validation["X"]
y_val = validation["y"]

X_test = test["X"]
y_test = test["y"]

print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)
```

Kết quả dự kiến:

``` text
Train:      (1794, 808) (1794,)
Validation: (598, 808)  (598,)
Test:       (599, 808)  (599,)
```

## 6. Sử dụng với các mô hình Machine Learning

Các file NPZ cung cấp ma trận đặc trưng `X` và nhãn `y`. Có thể dùng
trực tiếp với các mô hình scikit-learn nhận đầu vào dạng ma trận số.

### 6.1 Logistic Regression

Phù hợp làm mô hình baseline, có khả năng cung cấp xác suất dự đoán khi
cấu hình phù hợp.

``` python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

val_predictions = model.predict(X_val)

print(classification_report(y_val, val_predictions))
```

### 6.2 Linear SVM

Phù hợp với dữ liệu nhiều chiều. `LinearSVC` không cung cấp
`predict_proba` trực tiếp; có thể dùng `decision_function` để lấy điểm
quyết định.

``` python
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

model = LinearSVC(
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

val_predictions = model.predict(X_val)

print(classification_report(y_val, val_predictions))
```

### 6.3 Random Forest

Có thể dùng để so sánh với mô hình tuyến tính và đánh giá khả năng học
quan hệ phi tuyến. Với dữ liệu 808 chiều và số mẫu hiện tại, cần kiểm
soát độ phức tạp để tránh overfitting.

``` python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

val_predictions = model.predict(X_val)

print(classification_report(y_val, val_predictions))
```

Các đoạn code trên là ví dụ baseline, chưa phải kết quả thực nghiệm của
dự án. Cần chạy trên môi trường thực tế, lưu mô hình và so sánh trên
cùng validation set.

## 7. Nguyên tắc sử dụng Train, Validation và Test

1.  **Train:** dùng để học tham số mô hình.
2.  **Validation:** dùng để so sánh mô hình, điều chỉnh hyperparameter
    và ngưỡng phân loại.
3.  **Test:** chỉ dùng đánh giá cuối cùng sau khi đã chốt mô hình và cấu
    hình.

Không dùng kết quả Test để quay lại điều chỉnh mô hình nhiều lần. Không
gộp Test vào Train trong quá trình lựa chọn mô hình.

Do nhãn đang mất cân bằng (73,32% so với 26,68%), không nên chỉ nhìn
Accuracy. Nên theo dõi thêm:

-   Precision của lớp Spam.
-   Recall của lớp Spam.
-   F1-score của lớp Spam.
-   Confusion Matrix.
-   ROC-AUC và PR-AUC.

Chỉ số nào là ưu tiên phụ thuộc mục tiêu vận hành: giảm bỏ sót tin nhắn
lừa đảo hay giảm cảnh báo nhầm tin nhắn hợp lệ.

## 8. Lưu ý kỹ thuật và giới hạn hiện tại

-   **Xác nhận ý nghĩa label:** phải xác minh mapping `0`/`1` từ nguồn
    dataset gốc trước khi gọi một lớp là Spam hoặc Ham.
-   **Đặc trưng theo luật:** regex và từ khóa có thể bỏ sót mẫu mới,
    cách viết biến thể, văn bản có dấu hoặc dữ liệu bị ẩn danh.
-   **Số điện thoại và OTP:** thống kê phát hiện bằng luật hiện tại có
    thể thấp hơn thực tế; không coi số lần phát hiện là độ chính xác đã
    được kiểm chứng.
-   **Embedding:** mBERT embedding là vector ngữ cảnh, không phải xác
    suất hoặc nhãn phân loại.
-   **Pooling:** vector tin nhắn được tổng hợp từ embedding các segment.
    Nếu thay đổi segmentation hoặc cách pooling, cần tạo lại
    embedding/bộ đặc trưng và các tập chia tương ứng.
-   **Tái lập:** giữ cố định random seed, phiên bản thư viện, model
    tokenizer và model mBERT khi tái tạo pipeline.
-   **Dữ liệu mới:** phải áp dụng cùng quy trình làm sạch, trích xuất,
    tokenization, embedding, feature engineering và đúng
    `numeric_scaler.joblib`. Không tự ý thay đổi thứ tự 808 đặc trưng.

## 9. Lệnh chạy pipeline đã sử dụng

Chạy từ thư mục gốc dự án, trong môi trường Python/venv đã cài các thư
viện cần thiết:

``` powershell
python run_extraction.py
python run_segmentation.py
python run_tokenization.py
python run_embeddings.py
python run_pooling.py
python run_pca.py
python run_component_analysis.py
python build_component_features.py
python build_numeric_features.py
python combine_all_features.py
python validate_final_dataset.py
python split_dataset.py
python scale_features.py
```

Chỉ chạy lại bước cần thiết khi đầu vào tương ứng thay đổi. Nếu thay đổi
dữ liệu hoặc quy tắc tiền xử lý ở bước trước, cần chạy lại các bước phụ
thuộc phía sau để bảo đảm các file đầu ra đồng bộ.

------------------------------------------------------------------------

**Thư mục dữ liệu cuối cùng dùng cho training:** `output/scaled_splits/`

**Ma trận đầu vào:** `X` -- 808 chiều.

**Nhãn mục tiêu:** `y`.

**Scaler để tái sử dụng:** `output/scaled_splits/numeric_scaler.joblib`.
