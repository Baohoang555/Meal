import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

PRICE_44 = 44000
PRICE_70 = 70000


def run_evaluation():
    print("Đang đọc dữ liệu dự đoán out-of-fold và thông tin gốc...")
    ml_preds = pd.read_parquet("data/processed/ml_test_predictions.parquet")
    dl_preds = pd.read_parquet("data/processed/dl_test_predictions.parquet")
    features = pd.read_parquet("data/processed/features_monthly.parquet")

    # Ghép bảng
    df = ml_preds.merge(dl_preds, on=["employee_id", "month"], how="inner")
    
    # Bổ sung thông tin department nếu có
    if "department" in features.columns:
        dept_map = features[["employee_id", "month", "department"]].drop_duplicates()
        df = df.merge(dept_map, on=["employee_id", "month"], how="left")
    else:
        df["department"] = "Toàn công ty"

    models = ["m1", "m2", "m3", "m4", "m5"]
    months = sorted(df["month"].unique())

    # Tính toán dự đoán của S2 (Convex Ensemble) cho từng tháng test
    df["pred_n44_s2"] = 0.0
    df["pred_n70_s2"] = 0.0

    for m in months:
        idx = df[df["month"] == m].index
        sub = df.loc[idx]
        y_cost = sub["N44"].values * PRICE_44 + sub["N70"].values * PRICE_70

        cost_matrix = np.column_stack([
            sub[f"{mod}_p44"].values * PRICE_44 + sub[f"{mod}_p70"].values * PRICE_70
            for mod in models
        ])

        def loss_multi(w):
            return np.mean(np.abs(y_cost - cost_matrix @ w))

        cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0) for _ in range(5)]
        opt_s2 = minimize(loss_multi, [0.2] * 5, method="SLSQP", bounds=bounds, constraints=cons)
        w = opt_s2.x

        df.loc[idx, "pred_n44_s2"] = np.clip(np.sum([w[i] * sub[f"{models[i]}_p44"].values for i in range(5)], axis=0), 0, None)
        df.loc[idx, "pred_n70_s2"] = np.clip(np.sum([w[i] * sub[f"{models[i]}_p70"].values for i in range(5)], axis=0), 0, None)

    # Tính tiền thực tế và dự đoán
    df["actual_cost"] = df["N44"] * PRICE_44 + df["N70"] * PRICE_70
    df["pred_cost"] = df["pred_n44_s2"] * PRICE_44 + df["pred_n70_s2"] * PRICE_70
    df["error_cost"] = df["pred_cost"] - df["actual_cost"]
    df["abs_error_cost"] = np.abs(df["error_cost"])

    print("\n" + "=" * 75)
    print("1. ĐÁNH GIÁ CẤP ĐỘ TOÀN CÔNG TY (BUDGET LEVEL)")
    print("=" * 75)
    company_eval = (
        df.groupby("month")
        .agg(
            Tong_Nhan_Vien=("employee_id", "count"),
            Tien_Thuc_Te=("actual_cost", "sum"),
            Tien_Du_Doan=("pred_cost", "sum"),
            Chenh_Lech=("error_cost", "sum"),
        )
        .reset_index()
    )
    company_eval["Ti_Le_Lech_%"] = (company_eval["Chenh_Lech"] / company_eval["Tien_Thuc_Te"]) * 100

    print(company_eval.to_string(index=False, formatters={
        "Tien_Thuc_Te": "{:,.0f} đ".format,
        "Tien_Du_Doan": "{:,.0f} đ".format,
        "Chenh_Lech": "{:+,.0f} đ".format,
        "Ti_Le_Lech_%": "{:+.2f}%".format,
    }))

    print("\n" + "=" * 75)
    print("2. ĐÁNH GIÁ THEO PHÂN KHÚC HÀNH VI (BEHAVIOR SEGMENTS)")
    print("=" * 75)
    # Phân nhóm hành vi
    conditions = [
        (df["actual_cost"] == 0),
        (df["actual_cost"] > 0) & (df["actual_cost"] < 300000),
        (df["actual_cost"] >= 300000) & (df["actual_cost"] < 800000),
        (df["actual_cost"] >= 800000),
    ]
    labels = ["Không ăn (0 đ)", "Ăn ít (<300k)", "Ăn vừa (300k - 800k)", "Ăn nhiều (>=800k)"]
    df["segment"] = np.select(conditions, labels, default="Khác")

    segment_eval = (
        df.groupby("segment")
        .agg(
            So_Luong=("employee_id", "count"),
            MAE_Cost=("abs_error_cost", "mean"),
            RMSE_Cost=("error_cost", lambda x: np.sqrt(np.mean(x**2))),
            Tong_Thuc=("actual_cost", "sum"),
            Tong_Doan=("pred_cost", "sum"),
        )
        .reset_index()
    )
    segment_eval["WAPE_%"] = (
        df.groupby("segment").apply(lambda g: np.sum(g["abs_error_cost"]) / (np.sum(g["actual_cost"]) + 1e-5) * 100).values
    )

    print(segment_eval.to_string(index=False, formatters={
        "So_Luong": "{:,}".format,
        "MAE_Cost": "{:,.0f} đ".format,
        "RMSE_Cost": "{:,.0f} đ".format,
        "WAPE_%": "{:.2f}%".format,
        "Tong_Thuc": "{:,.0f} đ".format,
        "Tong_Doan": "{:,.0f} đ".format,
    }))

    print("\n" + "=" * 75)
    print("3. PHÂN TÍCH NHÓM KHẨU VỊ: MÓN ÂU (N70 > 0)")
    print("=" * 75)
    df["has_au"] = df["N70"] > 0
    au_eval = (
        df.groupby("has_au")
        .agg(
            So_Luong=("employee_id", "count"),
            MAE_N70=("N70", lambda x: np.mean(np.abs(x - df.loc[x.index, "pred_n70_s2"]))),
            MAE_Cost=("abs_error_cost", "mean"),
        )
        .reset_index()
    )
    au_eval["Nhom"] = au_eval["has_au"].map({True: "Có ăn món Âu", False: "Không ăn món Âu"})
    print(au_eval[["Nhom", "So_Luong", "MAE_N70", "MAE_Cost"]].to_string(index=False, formatters={
        "So_Luong": "{:,}".format,
        "MAE_N70": "{:.2f}".format,
        "MAE_Cost": "{:,.0f} đ".format,
    }))

    print("\n" + "=" * 75)
    print("4. TOP 10 TRƯỜNG HỢP NGOẠI LỆ SAI SỐ LỚN NHẤT (OUTLIERS)")
    print("=" * 75)
    top_outliers = df.sort_values("abs_error_cost", ascending=False).head(10)
    print(
        top_outliers[[
            "employee_id", "month", "N44", "pred_n44_s2", "N70", "pred_n70_s2", "actual_cost", "pred_cost", "error_cost"
        ]].to_string(
            index=False,
            formatters={
                "pred_n44_s2": "{:.1f}".format,
                "pred_n70_s2": "{:.1f}".format,
                "actual_cost": "{:,.0f} đ".format,
                "pred_cost": "{:,.0f} đ".format,
                "error_cost": "{:+,.0f} đ".format,
            },
        )
    )

    # Lưu báo cáo chi tiết
    os.makedirs("results/metrics", exist_ok=True)
    df.to_parquet("results/metrics/final_predictions_evaluated.parquet", index=False)
    company_eval.to_csv("results/metrics/company_budget_eval.csv", index=False)
    segment_eval.to_csv("results/metrics/segment_eval.csv", index=False)
    top_outliers.to_csv("results/metrics/top_outliers.csv", index=False)
    print("\nĐã lưu toàn bộ báo cáo chi tiết trong thư mục results/metrics/")


if __name__ == "__main__":
    run_evaluation()