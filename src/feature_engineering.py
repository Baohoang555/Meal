import numpy as np
import pandas as pd

from data_cleaning import PUBLIC_HOLIDAYS


def get_working_days_in_month(period: pd.Period) -> int:
    """Đếm ngày làm việc, không tính cuối tuần và ngày lễ đã cấu hình."""
    start_date = period.start_time.date()
    end_date = period.end_time.date()
    bus_days = pd.bdate_range(start=start_date, end=end_date)
    holidays = pd.DatetimeIndex(PUBLIC_HOLIDAYS).normalize()
    return int((~bus_days.normalize().isin(holidays)).sum())


def build_features(
    input_path="data/processed/monthly_dataset.parquet",
    output_path="data/processed/features_monthly.parquet",
):
    print("Đang đọc dữ liệu monthly...")
    df = pd.read_parquet(input_path)
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    df = df.sort_values(["employee_id", "month"]).reset_index(drop=True)

    # 1. LỊCH LÀM VIỆC CỦA THÁNG DỰ ĐOÁN (Biết trước)
    month_unique = df["month"].unique()
    work_days_map = {m: get_working_days_in_month(m) for m in month_unique}
    df["working_days"] = df["month"].map(work_days_map)

    # 2. ĐẶC TRƯNG LAGS (1, 2, 3 THÁNG)
    lag_cols = ["N44", "N70", "total_meals", "days_with_meal", "total_amount"]
    for lag in [1, 2, 3]:
        for col in lag_cols:
            df[f"{col}_lag_{lag}"] = df.groupby("employee_id")[col].shift(lag)

    # 3. ROLLING WINDOWS & MOMENTUM
    for col in ["N44", "N70", "total_meals"]:
        # Trung bình 3 tháng gần nhất
        df[f"{col}_mean_3m"] = (
            df.groupby("employee_id")[col]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )
        # Độ lệch chuẩn (tính biến thiên lịch sử)
        df[f"{col}_std_3m"] = (
            df.groupby("employee_id")[col]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=2).std())
            .fillna(0)
        )

    # Động lượng (tháng vừa rồi ăn nhiều hơn hay ít hơn 2 tháng trước)
    df["meal_momentum_1_2"] = (df["total_meals_lag_1"] + 1) / (df["total_meals_lag_2"] + 1)

    # Tỷ lệ tận dụng ngày làm việc ở tháng M-1
    # Giả sử tháng M-1 có working_days tương ứng
    df["prev_month"] = df["month"] - 1
    df["prev_working_days"] = df["prev_month"].map(work_days_map)
    df["attendance_rate_lag1"] = (
        df["days_with_meal_lag_1"] / df["prev_working_days"]
    ).clip(0, 1.5).fillna(0)
    df.drop(columns=["prev_month", "prev_working_days"], inplace=True)

    # Tỷ lệ chọn món Âu lịch sử
    df["au_ratio_lag1"] = (
        df["N70_lag_1"] / (df["total_meals_lag_1"] + 1e-5)
    ).clip(0, 1).fillna(0)

    # Không dùng au_share của chính tháng hiện tại làm feature: nó được tính
    # từ N70 của target và sẽ làm rò rỉ nhãn vào mô hình.
    #
    # QUAN TRỌNG: "expected_amount" và "amount_diff" (tạo ở monthly_aggregation.py)
    # được tính trực tiếp từ N44/N70 CỦA CHÍNH THÁNG ĐANG DỰ ĐOÁN
    # (expected_amount = N44*44000 + N70*70000), tức gần như CHÍNH LÀ đáp án
    # (amount_diff luôn bằng 0 vì không có bản ghi lệch giá). Nếu không loại,
    # đây là rò rỉ nhãn nghiêm trọng nhất trong toàn bộ pipeline — kiểm chứng
    # thực nghiệm cho thấy feature này chiếm >99% gain importance của LightGBM
    # và làm MAE thấp giả tạo gấp ~10 lần so với khi loại bỏ đúng cách.
    df.drop(columns=["au_share", "expected_amount", "amount_diff"], inplace=True, errors="ignore")

    # 4. TARGET ENCODING EXPANDING (Theo phòng ban - Tránh rò rỉ dữ liệu)
    if "department" in df.columns:
        df["dept_enc_n44"] = 0.0
        df["dept_enc_n70"] = 0.0
        months = sorted(df["month"].unique())
        for m in months:
            past_data = df[df["month"] < m]
            if len(past_data) > 0:
                dept_means = past_data.groupby("department")[["N44", "N70"]].mean()
                cur_idx = df[df["month"] == m].index
                m_dept = df.loc[cur_idx, "department"]
                df.loc[cur_idx, "dept_enc_n44"] = m_dept.map(dept_means["N44"]).fillna(past_data["N44"].mean())
                df.loc[cur_idx, "dept_enc_n70"] = m_dept.map(dept_means["N70"]).fillna(past_data["N70"].mean())

    # 5. Chỉ giữ các dòng có đủ 3 tháng lịch sử; không hard-code ngày tháng
    # để pipeline vẫn đúng khi dữ liệu được cập nhật hoặc đổi khoảng thời gian.
    required_history = [
        f"{col}_lag_{lag}"
        for col in ["N44", "N70", "total_meals", "days_with_meal", "total_amount"]
        for lag in [1, 2, 3]
    ]
    df_features = df[df[required_history].notna().all(axis=1)].copy()
    df_features = df_features.reset_index(drop=True)

    print(f"Bảng đặc trưng hoàn chỉnh: {df_features.shape[0]:,} dòng × {df_features.shape[1]} cột")
    df_features.to_parquet(output_path, index=False)
    print(f"Đã lưu tại: {output_path}")
    return df_features


if __name__ == "__main__":
    build_features()