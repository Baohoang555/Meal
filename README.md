# Dự đoán tiền ăn hàng tháng của nhân viên

Đồ án xây dựng mô hình dự đoán chi tiêu ăn uống tại căn-tin của từng nhân viên trong tháng tới, dựa trên log check-in lịch sử. Tiền ăn do nhân viên tự trả, đơn giá cố định: **Việt = 44.000đ**, **Chay = 44.000đ**, **Âu = 70.000đ**.

## 1. Mục tiêu

- Ước tính khoản chi ăn uống tháng tới cho từng nhân viên (phục vụ trừ lương / nạp thẻ căn-tin).
- Dự báo số suất từng loại món cho toàn công ty / theo phòng ban (phục vụ lên kế hoạch nguyên liệu, doanh thu căn-tin).
- So sánh hiệu quả giữa các họ mô hình ML, DL, và các cách kết hợp (ensemble) chúng.

## 2. Định nghĩa bài toán

Thay vì dự đoán thẳng số tiền, bài toán được phân rã thành dự đoán số bữa ăn từng loại trong tháng tới, sau đó nhân đơn giá:

```
Tổng tiền = N44 × 44.000 + N70 × 70.000
```

Trong đó:
- `N44`: số bữa ăn món Việt + Chay trong tháng
- `N70`: số bữa ăn món Âu trong tháng

Cách này chính xác và dễ diễn giải hơn so với ép mô hình đoán thẳng một con số tiền.

**Giới hạn của đề tài:**
- Dữ liệu chỉ phản ánh chi tiêu tại căn-tin qua thẻ check-in; không bao gồm ăn ngoài, mang cơm, hay các khoản ăn uống khác. Biến mục tiêu cần gọi đúng là "chi tiêu tại căn-tin", không phải tổng tiền ăn thực tế của nhân viên.
- Đơn giá cố định trong toàn bộ giai đoạn dữ liệu, nên mô hình không phản ánh được hành vi khi giá thay đổi.
- Dữ liệu cá nhân (tên/mã nhân viên) cần được ẩn danh khi trình bày và lưu trữ.

## 3. Dữ liệu

- **Nguồn:** log check-in nhà ăn (thời gian check-in, mã/tên nhân viên, phòng ban, loại món).
- **Khoảng thời gian:** từ tháng 6 năm trước đến hiện tại (~15 tháng).
- **Quy mô:** ~2.500 nhân viên → sau khi tổng hợp theo tháng, ~30.000–35.000 dòng (nhân viên × tháng).

> Cột dữ liệu cụ thể và số dòng thực tế cần điền lại sau khi khảo sát file log gốc.

## 4. Pipeline tổng thể

```
1. Làm sạch dữ liệu & dựng lưới nhân viên × ngày
2. EDA (phân phối, tỷ lệ món, số 0, giờ ăn, theo phòng ban)
3. Chốt biến mục tiêu (N44, N70 theo tháng) & cách chia dữ liệu
4. Baseline (Naive Lag-1, Moving Average 3 tháng, Group-Mean, Ridge, Decision Tree nông)
5. Feature engineering
6. Huấn luyện 3 mô hình ML
7. Huấn luyện 2 mô hình DL
8. Ensemble (10 cặp đôi + mở rộng nhiều mô hình)
9. Đánh giá, phân tích lỗi, viết giới hạn
```

### 4.1. Vấn đề điển hình cần xử lý ở mỗi bước

| Nhóm | Vấn đề |
|---|---|
| Dữ liệu gốc | Log chỉ ghi ngày có ăn (cần điền 0 cho ngày không ăn); dữ liệu trùng/nhiễu; không phân biệt được lý do vắng (nghỉ phép, đi ngoài, nghỉ việc); tên/phòng ban thay đổi theo thời gian |
| Định nghĩa bài toán | Chọn đơn vị dự đoán (tiền / số bữa / theo ngày); phân phối lệch, nhiều số 0; chỉ có 1 chu kỳ Tết trong dữ liệu |
| Đặc trưng | Data leakage (dùng thông tin của tháng cần dự đoán); cold-start cho nhân viên mới; mất 3 tháng đầu do lag |
| Mô hình | Số mẫu độc lập thực tế ít hơn số dòng (2.500 người lặp lại nhiều tháng); DL khó thắng GBDT trên dữ liệu bảng; so sánh không công bằng nếu ngân sách tuning khác nhau |
| Đánh giá | Chỉ test 1 tháng dễ do may rủi; MAPE hỏng khi giá trị thật = 0; baseline đơn giản có thể đã rất mạnh |

## 5. Chia dữ liệu

Chia theo trục thời gian (**không** random split) để tránh rò rỉ thông tin tương lai:

- **Train:** ~10–11 tháng đầu
- **Validation:** 2–3 tháng tiếp theo (tuning siêu tham số, early stopping)
- **Test:** backtest kiểu **expanding window** trên nhiều tháng cuối (không chỉ 1 tháng), để kết luận không phụ thuộc vào may rủi của một tháng đơn lẻ

Target encoding / lag features chỉ được tính từ dữ liệu quá khứ so với tháng cần dự đoán.

## 6. Baseline

| Cấp độ | Mô hình | Công thức / cơ chế |
|---|---|---|
| Level 0 | Naive Lag-1 | Tiền tháng M = Tiền tháng M−1 |
| Level 0 | Group-Mean | Chi phí TB/ngày của phòng ban × số ngày làm việc tháng M |
| Level 1 | Moving Average 3M | Trung bình tiền 3 tháng gần nhất |
| Level 2 | Ridge / Lasso | Hồi quy tuyến tính có điều chuẩn trên bộ đặc trưng |
| Level 2 | Decision Tree nông (max_depth ≈ 4–5) | Cây quyết định đơn, dễ trực quan hóa |

Baseline là mốc bắt buộc: nếu ML/DL không vượt qua được nhóm này, mô hình đó không có giá trị triển khai.

## 7. Bộ đặc trưng (Feature Engineering)

| Nhóm | Đặc trưng |
|---|---|
| Lịch sử tần suất | Số bữa ăn tháng M−1, M−2, M−3 |
| Lịch sử chọn món | Tỷ lệ % chọn món Âu (70k) trong 3 tháng gần nhất — biến quan trọng nhất |
| Hành vi check-in | Giờ check-in trung bình, độ lệch chuẩn giờ ăn |
| Tổ chức | Mã nhân viên, phòng ban |
| Lịch làm việc | Số ngày làm việc thực tế của tháng cần dự đoán (trừ T7, CN, ngày lễ) — thông tin **biết trước**, rất quan trọng |

## 8. Danh sách mô hình (đã chốt)

| Ký hiệu | Mô hình | Họ |
|---|---|---|
| M1 | Random Forest Regressor | ML |
| M2 | LightGBM Regressor | ML |
| M3 | CatBoost Regressor | ML |
| M4 | GRU / LSTM (chuỗi check-in theo ngày) | DL |
| M5 | MLP với entity embedding (hoặc TabNet) | DL |

### 8.1. Ablation từng mô hình (v0 = cơ bản → v* = bản tốt nhất đưa vào ensemble)

**M1 — Random Forest:** tuning `max_features`/`min_samples_leaf`/`max_depth` → thử Extra Trees → loại đặc trưng bằng permutation importance.

**M2 — LightGBM:** early stopping theo thời gian → Optuna tuning → đổi loss sang Poisson cho số bữa → monotone constraints.

**M3 — CatBoost:** early stopping → tuning `depth`/`l2_leaf_reg`/`learning_rate` → MultiRMSE (dự đoán đồng thời N44, N70) → kiểm tra đóng góp của Ma_NV.

**M4 — GRU/LSTM:** chuẩn hóa + AdamW + gradient clipping + dropout → nối đặc trưng lịch tháng dự đoán → nối embedding nhân viên/phòng ban → multi-task head + Huber loss → attention trên chuỗi (tùy chọn) → **trung bình 5 seed**.

**M5 — MLP/TabNet:** chuẩn hóa + AdamW + dropout → entity embedding (kèm embedding dropout, weight decay) → mã hóa số kiểu quantile/piecewise-linear → BatchNorm + residual → multi-task head + Huber loss → thử TabNet → **trung bình 5 seed**.

**Quy tắc chung:** cùng một bộ đặc trưng, cùng ngân sách tuning (50 lần thử/mô hình), cùng cách backtest, để so sánh công bằng giữa các họ mô hình.

## 9. Ensemble

### 9.1. 10 cặp đôi

| # | Cặp | Loại |
|---|---|---|
| 1 | M1 + M2 | ML + ML |
| 2 | M1 + M3 | ML + ML |
| 3 | M2 + M3 | ML + ML |
| 4 | M4 + M5 | DL + DL |
| 5 | M1 + M4 | ML + DL |
| 6 | M1 + M5 | ML + DL |
| 7 | M2 + M4 | ML + DL |
| 8 | M2 + M5 | ML + DL |
| 9 | M3 + M4 | ML + DL |
| 10 | M3 + M5 | ML + DL |

### 9.2. Kỹ thuật kết hợp cho mỗi cặp (E0 → E*)

| Phiên bản | Kỹ thuật |
|---|---|
| E0 | Trung bình đơn giản (A+B)/2 |
| E1 | Trung bình có trọng số w·A + (1−w)·B, w chọn trên validation |
| E2 | Trọng số riêng cho N44 và N70 |
| E3 | Walk-forward weights: ước lượng lại w mỗi fold backtest, chỉ dùng dự đoán các tháng trước |
| E4 | Hiệu chỉnh sau khi trộn: cắt N trong [0, số ngày làm việc], sửa bias |

### 9.3. Mở rộng — kết hợp nhiều mô hình (S0 → S4)

Trung bình đều 5 mô hình → chọn theo độ đa dạng (tương quan sai số) → Non-negative least squares → Greedy forward selection (Caruana) → Stacking (Ridge/LightGBM nhỏ làm meta-learner).

### 9.4. Rủi ro cần lưu ý khi ensemble

- Dữ liệu để học trọng số rất ít (~12 tháng dùng được) → ưu tiên E1–E3 (ít tham số) hơn stacking phức tạp.
- Trọng số/stacking phải học trên dữ liệu **chưa** dùng để tuning mô hình gốc, tránh rò rỉ hai lớp.
- Cải thiện nhỏ có thể chỉ là nhiễu → báo cáo số tháng ensemble thắng mô hình đơn tốt nhất, không chỉ MAE trung bình.
- Ensemble không đảm bảo luôn thắng — nếu trọng số tối ưu dồn hết về một mô hình, đó vẫn là kết luận hợp lệ.

## 10. Đánh giá

| Cấp độ | Metric | Mục đích |
|---|---|---|
| Cá nhân | MAE, RMSE (VNĐ) | Độ chính xác từng nhân viên |
| Toàn công ty / phòng ban | WAPE, sai số tổng | Phù hợp cho dự báo ngân sách/nguyên liệu |
| Theo loại món | MAE số suất Việt/Chay/Âu | Phục vụ kế hoạch căn-tin |

MAPE không dùng làm chỉ số chính vì bị méo khi giá trị thật bằng 0.

## 11. Bảng tổng hợp số cấu hình cần chạy

| Nhóm | Số cấu hình |
|---|---|
| Baseline | 5 |
| Ablation 5 mô hình (v0 → v*) | ~25 |
| Ensemble 10 cặp × (E0, E1, E3 tối thiểu) | 30 |
| Mở rộng (S0–S4) | tối đa 5 |

Nếu thời gian hạn chế: chỉ cần bản v* của mỗi mô hình, cặp ensemble dùng E0/E1, và 1–2 cấu hình mở rộng (S2 hoặc S4) cho nhóm đáng chú ý nhất.

## 12. Cấu trúc thư mục đề xuất

```
project/
├── data/
│   ├── raw/                # log check-in gốc
│   └── processed/          # bảng đặc trưng theo nhân viên/tháng
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_error_analysis.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── baselines.py
│   ├── models/
│   │   ├── ml_models.py      # M1, M2, M3
│   │   └── dl_models.py      # M4, M5
│   ├── ensemble.py
│   └── evaluate.py
├── results/
│   ├── metrics/
│   └── figures/
├── README.md
└── requirements.txt
```
"""
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

Set-Location "C:\Users\cris\CS\Meal"

python "src\data_cleaning.py" `
    --input "data\raw\MealHistory_UTF8.csv" `
    --output "data\processed\daily_grid.parquet" `
    --removed-log "data\processed\removed_rows.csv"

python "src\monthly_aggregation.py" `
    --grid "data\processed\daily_grid.parquet" `
    --output "data\processed\monthly_dataset.parquet"

python "src\feature_engineering.py"

python "src\check_monthly_history.py"

python "src\eda.py" `
    --raw-log "data\raw\MealHistory_UTF8.csv" `
    --grid "data\processed\daily_grid.parquet" `
    --output-dir "results"

python "src\baselines.py"

python "src\models\ml_models.py"

python "src\models\dl_models.py"

python "src\ensemble.py"

python "src\evaluate.py"
"""