# Báo cáo kết quả dự báo chi tiêu căn-tin

## 1. Tóm tắt điều hành

Project xây dựng hệ thống dự báo số bữa ăn tháng kế tiếp của nhân viên, sau đó
quy đổi thành chi phí:

```text
Chi phí = N44 × 44.000 + N70 × 70.000
```

Trong đó `N44` là số bữa Việt + Chay và `N70` là số bữa Âu.

Kết quả chính:

- Mô hình DL tốt nhất trong các kết quả đã xuất là **M4-GRU**:
  - MAE chi phí: **4.697 đ / employee-month**
  - RMSE chi phí: **17.364 đ**
  - WAPE: **0,669%**
- Ensemble đa mô hình S2 đạt:
  - MAE chi phí: **7.764 đ**
  - RMSE chi phí: **16.965 đ**
  - WAPE: **1,022%**
- Trong các baseline, **Ridge** tốt nhất theo MAE chi phí:
  - MAE: **201.360 đ**
  - WAPE: **31,309%**
- Dự báo ngân sách toàn công ty trong ba tháng test lệch từ
  **-0,765% đến +0,146%**.

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
- Test/backtest: tháng 2026-07, 2026-08 và 2026-09.
- Feature lag và target encoding chỉ dùng dữ liệu trước tháng cần dự báo.

Feature engineering đã được điều chỉnh để:

- Không dùng `au_share` của chính tháng mục tiêu.
- Chỉ giữ dòng có đủ lag 1, 2 và 3 tháng.
- Tính ngày làm việc có loại ngày lễ cấu hình.
- Dùng target encoding phòng ban dạng expanding theo thời gian.

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
| CatBoost | 11.293 đ | 34.133 đ | 1,573% | 0,371 | 0,129 |
| LightGBM | 12.186 đ | 26.431 đ | 1,707% | 0,369 | 0,136 |
| Random Forest | 39.308 đ | 54.653 đ | 6,031% | 0,975 | 0,173 |

CatBoost có MAE thấp nhất trong nhóm ML. LightGBM có RMSE thấp hơn CatBoost,
cho thấy sai số trung bình tốt nhưng vẫn có một số lỗi lớn hơn theo MAE tuyệt
đối.

## 6. Kết quả DL

Nguồn: [`results/metrics/dl_models_summary.csv`](results/metrics/dl_models_summary.csv).

| Mô hình | MAE chi phí | RMSE chi phí | WAPE | MAE N44 | MAE N70 |
|---|---:|---:|---:|---:|---:|
| GRU | **4.697 đ** | 17.364 đ | **0,669%** | 0,246 | 0,149 |
| MLP | 11.702 đ | 17.737 đ | 1,607% | 0,436 | 0,175 |

GRU là mô hình tốt nhất theo MAE chi phí và WAPE trong các kết quả hiện có.
Kết quả này cần được xác nhận lại sau khi chạy lại toàn bộ pipeline với cùng
artifact feature và cùng môi trường.

## 7. Kết quả ensemble

### 7.1. Cặp đôi tốt nhất

Nguồn: [`results/metrics/ensemble_pairs_summary.csv`](results/metrics/ensemble_pairs_summary.csv).

| Cấu hình | MAE chi phí | RMSE chi phí | WAPE |
|---|---:|---:|---:|
| GRU + MLP, E1 | **5.509 đ** | 13.605 đ | 0,755% |
| LightGBM + GRU, E1 | 5.592 đ | 15.104 đ | 0,780% |
| CatBoost + GRU, E1 | 5.690 đ | 19.718 đ | 0,778% |

E1 là weighted average. Trọng số phải được học từ các tháng trước, không dùng
target của chính tháng đang đánh giá.

### 7.2. Ensemble năm mô hình

Nguồn: [`results/metrics/ensemble_multi_summary.csv`](results/metrics/ensemble_multi_summary.csv).

| Phương pháp | MAE chi phí | RMSE chi phí | WAPE |
|---|---:|---:|---:|
| S0 - trung bình đều | 11.617 đ | 17.688 đ | 1,694% |
| S2 - convex optimization | **7.764 đ** | **16.965 đ** | **1,022%** |

S2 tốt hơn S0, nhưng vẫn kém cặp GRU + MLP E1 và mô hình GRU đơn lẻ theo MAE.
Không nên mặc định rằng thêm nhiều mô hình sẽ luôn cải thiện kết quả.

## 8. Độ chính xác ngân sách toàn công ty

Nguồn: [`results/metrics/company_budget_eval.csv`](results/metrics/company_budget_eval.csv).

| Tháng | Chi phí thực tế | Dự báo | Sai lệch | Sai lệch % |
|---|---:|---:|---:|---:|
| 2026-07 | 1.299.814.000 đ | 1.289.873.000 đ | -9.941.000 đ | -0,765% |
| 2026-08 | 1.194.892.000 đ | 1.196.634.000 đ | +1.742.000 đ | +0,146% |
| 2026-09 | 778.952.000 đ | 778.437.000 đ | -515.000 đ | -0,066% |

Ở cấp công ty, sai lệch ngân sách khá thấp. Đây là kết quả phù hợp cho hoạch
định ngân sách tổng, nhưng không có nghĩa mọi cá nhân đều được dự báo chính xác.

## 9. Sai số theo nhóm hành vi

Nguồn: [`results/metrics/segment_eval.csv`](results/metrics/segment_eval.csv).

| Nhóm | Số lượng | MAE chi phí | WAPE |
|---|---:|---:|---:|
| Không ăn | 79 | 5.552 đ | 0%* |
| Ăn ít dưới 300k | 313 | 7.556 đ | 5,489% |
| Ăn vừa 300k-800k | 2.062 | 5.239 đ | 0,881% |
| Ăn nhiều từ 800k | 2.030 | 10.439 đ | 1,057% |

`WAPE = 0%*` ở nhóm không ăn là do tổng chi phí thực tế của nhóm bằng 0, không
phải dự báo hoàn hảo. Thực tế nhóm này vẫn có dự báo dư khoảng 438.634 đ.

Nhóm ăn ít có WAPE cao vì mẫu số chi phí thực tế nhỏ. Nhóm ăn nhiều có MAE tiền
cao nhất do quy mô chi tiêu lớn hơn.

## 10. Ngoại lệ lớn

Nguồn: [`results/metrics/top_outliers.csv`](results/metrics/top_outliers.csv).

Các ngoại lệ lớn thường rơi vào một trong các trường hợp:

- Dự báo thừa số bữa Âu cho nhân viên thực tế ăn ít hoặc không ăn Âu.
- Dự báo thấp ở nhân viên có biến động mạnh giữa các tháng.
- Nhân viên có hành vi đột biến so với lịch sử gần nhất.

Các dòng chi tiết được giữ trong file `top_outliers.csv`; cần ẩn danh mã nhân
viên trước khi đưa báo cáo ra ngoài project.

## 11. Đánh giá tổng thể và giới hạn

### Điểm đạt được

1. Pipeline đã có làm sạch dữ liệu, daily grid, monthly aggregation, feature
   engineering, baseline, ML, DL, ensemble và evaluation.
2. Chia dữ liệu theo thời gian, phù hợp hơn random split cho bài toán dự báo.
3. Đã loại bỏ target leakage từ `au_share` tháng hiện tại.
4. Đã có kiểm tra giá: 0 bản ghi lệch giá 44.000/70.000.
5. Dự báo ngân sách cấp công ty có sai lệch dưới 1% ở cả ba tháng test.

### Giới hạn cần ghi rõ

1. Chỉ có ba tháng test, nên chưa đủ để kết luận ổn định dài hạn.
2. 10% nhân viên có tenure dưới 90 ngày; cold-start vẫn là vấn đề.
3. Món Âu chỉ chiếm 1,6%, khiến N70 khó đánh giá ổn định.
4. Kết quả DL tốt cần được tái lập sau khi chạy lại pipeline hoàn toàn.
5. Dữ liệu chỉ phản ánh chi tiêu tại căn-tin, không phải tổng chi phí ăn uống
   thực tế của nhân viên.
6. Danh sách ngày lễ hiện vẫn cần được đối chiếu với lịch chính thức của công ty.
7. Dự báo số bữa là số thực liên tục; nếu cần vận hành theo số suất, cần thêm
   bước làm tròn và đánh giá lại sai số sau làm tròn.

## 12. Khuyến nghị triển khai

1. Chạy lại toàn bộ pipeline để sinh đồng bộ feature và prediction mới.
2. Dùng GRU hoặc GRU + MLP E1 làm ứng viên chính, nhưng giữ Ridge làm baseline
   giám sát.
3. Theo dõi riêng:
   - người không ăn;
   - người mới;
   - nhóm có món Âu;
   - nhân viên có sai số lớn.
4. Mở rộng backtest lên nhiều tháng hơn trước khi dùng cho quyết định tài chính.
5. Với ngân sách công ty, theo dõi thêm sai lệch tổng tiền theo tháng; với vận hành
   bếp, theo dõi thêm sai số N44 và N70 riêng biệt.

