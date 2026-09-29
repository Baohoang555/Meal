# Báo cáo kết quả dự báo chi tiêu căn-tin

## 1. Tóm tắt điều hành

Project xây dựng hệ thống dự báo số bữa ăn tháng kế tiếp của nhân viên, sau đó
quy đổi thành chi phí:

```text
Chi phí = N44 × 44.000 + N70 × 70.000
```

Trong đó `N44` là số bữa Việt + Chay và `N70` là số bữa Âu.

Kết quả hiện tại sau khi loại bỏ feature rò rỉ target:

- ML tốt nhất theo MAE là **CatBoost**: MAE chi phí **189.586 đ**,
  RMSE **234.600 đ**, WAPE **29,803%**.
- DL tốt nhất theo MAE là **MLP**: MAE chi phí **195.767 đ**,
  RMSE **241.624 đ**, WAPE **30,509%**.
- Baseline tốt nhất theo MAE là **Ridge**: MAE chi phí **201.360 đ**,
  RMSE **241.608 đ**, WAPE **31,309%**.
- Ensemble S2 đạt MAE chi phí **193.146 đ**, RMSE **236.388 đ**,
  WAPE **30,212%**.
- Sai lệch tổng ngân sách trên evaluation cohort hiện tại là **-9,147%**
  (2026-07), **+4,263%** (2026-08) và **+54,515%** (2026-09).

Kết quả trước đây có MAE chỉ vài nghìn đồng được tạo ra khi mô hình còn nhận
feature `expected_amount`/`amount_diff` tính từ target tháng hiện tại. Đó là
target leakage, không phải hiệu năng dự báo hợp lệ; các kết quả cũ không được
dùng làm căn cứ triển khai.

> Lưu ý: số liệu trong báo cáo được đọc từ các file kết quả hiện có trong
> `results/metrics`. Sau khi chạy lại toàn bộ pipeline, các con số có thể thay
> đổi theo dữ liệu và môi trường.

## 2. Dữ liệu và EDA

Nguồn EDA: [`results/eda_report.md`](results/eda_report.md).

| Chỉ tiêu | Kết quả |
|---|---:|
| Số nhân viên | 1.787 |
| Khoảng thời gian | 2025-07-03 đến 2026-09-21 |
| Số dòng lưới nhân viên-ngày | 633.406 |
| Tỷ lệ ngày có ăn | 57,5% |
| Tổng bữa hợp lệ | 363.962 |
| Tổng chi tiêu ghi nhận | 16.162.710.000 đ |
| Tenure ngắn dưới 90 ngày | 179 / 1.787 = 10,0% |
| Bản ghi lệch bảng giá | 0 |

### 2.1. Hành vi theo ngày

| Ngày | Tỷ lệ có ăn |
|---|---:|
| Thứ 2 | 72,1% |
| Thứ 3 | 73,1% |
| Thứ 4 | 74,3% |
| Thứ 5 | 72,5% |
| Thứ 6 | 73,7% |
| Thứ 7 | 36,2% |
| Chủ nhật | 0,0% |

Ngày làm việc có tỷ lệ ăn khoảng 72-74%, thứ 7 giảm mạnh và chủ nhật không có
bữa. Vì vậy các biến lịch làm việc, cuối tuần và ngày lễ là các biến quan trọng.

### 2.2. Cơ cấu món ăn

| Loại món | Tỷ lệ |
|---|---:|
| Việt | 86,0% |
| Chay | 12,4% |
| Âu | 1,6% |

Món Âu là nhóm hiếm, nên MAE của `N70` cần được phân tích riêng. Tỷ lệ món Âu
giữa các phòng ban dao động từ 0% đến 100%, nhưng các phòng ban rất nhỏ có thể
làm tỷ lệ này biến động mạnh.

### 2.3. Phân phối chi phí

| Thống kê | Giá trị |
|---|---:|
| Trung bình employee-month | 723.455 đ |
| Trung vị | 836.000 đ |
| Độ lệch chuẩn | 302.205 đ |
| Nhỏ nhất | 0 đ |
| Phân vị 25% | 528.000 đ |
| Phân vị 75% | 968.000 đ |
| Lớn nhất | 1.750.000 đ |

### 2.4. Các biểu đồ EDA

- [Tỷ lệ ăn theo thứ](results/figures/ty_le_an_theo_thu.png)
- [Tỷ lệ món tổng thể](results/figures/ty_le_mon_tong_the.png)
- [Tỷ lệ món theo phòng ban](results/figures/ty_le_mon_theo_phong_ban.png)
- [Giờ ăn theo ca](results/figures/gio_an_theo_ca.png)
- [Phân phối tiền ăn theo tháng](results/figures/phan_phoi_tien_an_thang.png)
- [Tổng tiền theo tháng](results/figures/tong_tien_theo_thang.png)

## 3. Cách chia dữ liệu và mục tiêu

Pipeline sử dụng chia theo thời gian, không random split:

- Train: các tháng trước tháng test.
- Test/backtest artifacts hiện có: tháng 2026-07, 2026-08 và 2026-09.
- Feature lag và target encoding chỉ dùng dữ liệu trước tháng cần dự báo.

Feature engineering đã được điều chỉnh để:

- Không dùng `au_share` của chính tháng mục tiêu.
- Không dùng `expected_amount`/`amount_diff`, là các giá trị dẫn xuất trực tiếp
  từ `N44`/`N70` của tháng mục tiêu.
- Chỉ giữ dòng có đủ lag 1, 2 và 3 tháng.
- Tính ngày làm việc có loại ngày lễ cấu hình.
- Dùng target encoding phòng ban dạng expanding theo thời gian.

### Cảnh báo về tháng 2026-09 chưa hoàn chỉnh

Daily grid hiện chỉ có dữ liệu đến **2026-09-21**, nhưng prediction/evaluation
đang so sánh dự báo tổng của cả tháng với tổng thực tế tính đến ngày 21. Do đó
tháng 9 chưa phải một fold test hợp lệ cho dự báo cả tháng. Sai lệch +54,515%
không thể diễn giải là sai số forecast tháng đầy đủ: thực tế tháng đang bị
right-censored (thiếu 9 ngày lịch cuối tháng). Cần nạp đủ log tháng 9 rồi chạy
lại evaluation, hoặc loại tháng này khỏi thống kê backtest.

Các summary ML/DL/ensemble hiện trung bình kết quả các tháng test sẵn có, gồm
cả tháng 9 chưa đủ dữ liệu; vì vậy các chỉ số tổng hợp dưới đây chỉ là kết quả
tham khảo cho artifact hiện tại, chưa phải kết luận hiệu năng production.

## 4. Kết quả baseline

Nguồn: [`results/metrics/baselines_summary.csv`](results/metrics/baselines_summary.csv).

| Mô hình | MAE chi phí | RMSE chi phí | WAPE | MAE N44 | MAE N70 |
|---|---:|---:|---:|---:|---:|
| Ridge | 201.360 đ | 241.608 đ | 31,309% | 4,461 | 0,279 |
| Decision Tree d=4 | 202.443 đ | 246.987 đ | 30,894% | 4,436 | 0,286 |
| MA 3M | 204.037 đ | 259.103 đ | 32,386% | 4,477 | 0,259 |
| Naive Lag-1 | 208.352 đ | 265.384 đ | 32,714% | 4,557 | 0,257 |
| Group/Global Mean | 239.551 đ | 292.842 đ | 35,328% | 5,389 | 0,448 |

Ridge là baseline có MAE chi phí thấp nhất. Đây là mốc tối thiểu cần vượt qua
khi đánh giá ML/DL.

## 5. Kết quả ML

Nguồn: [`results/metrics/ml_models_summary.csv`](results/metrics/ml_models_summary.csv).

| Mô hình | MAE chi phí | RMSE chi phí | WAPE | MAE N44 | MAE N70 |
|---|---:|---:|---:|---:|---:|
| CatBoost | **189.586 đ** | 234.600 đ | 29,803% | 4,213 | 0,211 |
| LightGBM | 191.133 đ | **234.239 đ** | **29,699%** | 4,256 | **0,209** |
| Random Forest | 194.189 đ | 235.671 đ | 30,116% | 4,306 | 0,255 |

CatBoost có MAE thấp nhất trong nhóm ML; LightGBM có RMSE và WAPE thấp hơn một
chút. Chênh lệch nhỏ và summary có bao gồm fold tháng 9 chưa hoàn chỉnh.

## 6. Kết quả DL

Nguồn: [`results/metrics/dl_models_summary.csv`](results/metrics/dl_models_summary.csv).

| Mô hình | MAE chi phí | RMSE chi phí | WAPE | MAE N44 | MAE N70 |
|---|---:|---:|---:|---:|---:|
| MLP | **195.767 đ** | 241.624 đ | **30,509%** | **4,349** | 0,236 |
| GRU | 217.041 đ | **258.494 đ** | 34,157% | 4,824 | **0,228** |

MLP tốt hơn GRU theo MAE chi phí và WAPE; GRU có MAE N70 thấp hơn. Cả hai mô hình
DL hiện không vượt baseline Ridge theo MAE chi phí. Kết luận trước đây rằng GRU
là mô hình tốt nhất là không còn đúng sau khi loại bỏ leakage.

## 7. Kết quả ensemble

### 7.1. Cặp đôi tốt nhất

Nguồn: [`results/metrics/ensemble_pairs_summary.csv`](results/metrics/ensemble_pairs_summary.csv).

| Cấu hình | MAE chi phí | RMSE chi phí | WAPE |
|---|---:|---:|---:|
| LightGBM + CatBoost, E0 | **189.521 đ** | **233.254 đ** | **29,628%** |
| Random Forest + CatBoost, E0 | 190.457 đ | 233.356 đ | 29,752% |
| CatBoost + MLP, E1 | 191.395 đ | 236.001 đ | 30,011% |

Trong bảng hiện tại, E0 là trung bình đơn giản. E1 là weighted average; trọng số
cần được học từ các tháng trước, không dùng target của chính tháng đang đánh giá.

### 7.2. Ensemble năm mô hình

Nguồn: [`results/metrics/ensemble_multi_summary.csv`](results/metrics/ensemble_multi_summary.csv).

| Phương pháp | MAE chi phí | RMSE chi phí | WAPE |
|---|---:|---:|---:|
| S0 - trung bình đều | 194.939 đ | 237.184 đ | 30,498% |
| S2 - convex optimization | **193.146 đ** | **236.388 đ** | **30,212%** |

S2 chỉ cải thiện nhẹ so với S0 và vẫn kém baseline Ridge theo MAE chi phí.
Không nên mặc định rằng ensemble nhiều mô hình sẽ luôn cải thiện kết quả.

## 8. Độ chính xác ngân sách toàn công ty

Nguồn: [`results/metrics/company_budget_eval.csv`](results/metrics/company_budget_eval.csv).

| Tháng | Chi phí thực tế | Dự báo | Sai lệch | Sai lệch % |
|---|---:|---:|---:|---:|
| 2026-07 | 1.299.814.000 đ | 1.180.916.555 đ | -118.897.445 đ | -9,147% |
| 2026-08 | 1.194.892.000 đ | 1.245.831.339 đ | +50.939.339 đ | +4,263% |
| 2026-09* | 778.952.000 đ | 1.203.597.983 đ | +424.645.983 đ | +54,515% |

*Tháng 9 chỉ có dữ liệu đến 21/09, không phải tổng thực tế cả tháng. Vì vậy
không dùng fold này để kết luận độ chính xác forecast tháng. Tháng 7 và 8 trong
evaluation cohort lần lượt lệch -9,147% và +4,263%; các kết quả này cũng cần
được so với baseline trên đúng cùng cohort.

## 9. Sai số theo nhóm hành vi

Nguồn: [`results/metrics/segment_eval.csv`](results/metrics/segment_eval.csv).

| Nhóm | Số lượng | MAE chi phí | RMSE chi phí | WAPE |
|---|---:|---:|---:|---:|
| Không ăn | 79 | 110.572 đ | 229.219 đ | Không xác định* |
| Ăn ít dưới 300k | 313 | 301.550 đ | 400.644 đ | 219,083% |
| Ăn vừa 300k-800k | 2.062 | 251.467 đ | 284.431 đ | 42,304% |
| Ăn nhiều từ 800k | 2.030 | 119.694 đ | 154.754 đ | 12,119% |

WAPE của nhóm không ăn không xác định vì tổng chi phí thực tế bằng 0; giá trị
0 trong CSV là quy ước khi mẫu số bằng 0, không phải dự báo hoàn hảo. Tổng dự
báo dư của nhóm này là khoảng **8.735.186 đ**.

Các sai số theo segment lớn hơn đáng kể so với báo cáo cũ. Nhóm ăn ít có WAPE
cao một phần vì mẫu số thực tế nhỏ; tháng test chưa hoàn chỉnh cũng ảnh hưởng
phân đoạn theo chi tiêu thực tế.

## 10. Ngoại lệ lớn

Nguồn: [`results/metrics/top_outliers.csv`](results/metrics/top_outliers.csv).

Các ngoại lệ lớn thường rơi vào một trong các trường hợp:

- Dự báo thừa số bữa Âu cho nhân viên thực tế ăn ít hoặc không ăn Âu.
- Dự báo thấp ở nhân viên có biến động mạnh giữa các tháng.
- Nhân viên có hành vi đột biến so với lịch sử gần nhất.

Các dòng chi tiết được giữ trong file `top_outliers.csv`; cần ẩn danh mã nhân
viên trước khi đưa báo cáo ra ngoài project.
