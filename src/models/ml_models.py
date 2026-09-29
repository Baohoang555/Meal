import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

PRICE_44 = 44000
PRICE_70 = 70000


def calculate_metrics(y_true_n44, y_true_n70, y_pred_n44, y_pred_n70):
    """Tính các metric: MAE tiền, RMSE tiền, WAPE tiền, MAE từng món."""
    y_pred_n44 = np.clip(y_pred_n44, 0, None)
    y_pred_n70 = np.clip(y_pred_n70, 0, None)

    cost_true = y_true_n44 * PRICE_44 + y_true_n70 * PRICE_70
    cost_pred = y_pred_n44 * PRICE_44 + y_pred_n70 * PRICE_70

    mae_cost = np.mean(np.abs(cost_true - cost_pred))
    rmse_cost = np.sqrt(np.mean((cost_true - cost_pred) ** 2))
    total_cost = np.sum(cost_true)
    wape_cost = (
        np.sum(np.abs(cost_true - cost_pred)) / total_cost * 100
        if total_cost > 0
        else 0.0
    )

    mae_n44 = np.mean(np.abs(y_true_n44 - y_pred_n44))
    mae_n70 = np.mean(np.abs(y_true_n70 - y_pred_n70))

    return {
        "MAE_Cost": mae_cost,
        "RMSE_Cost": rmse_cost,
        "WAPE_Cost_%": wape_cost,
        "MAE_N44": mae_n44,
        "MAE_N70": mae_n70,
    }


def train_predict_m1_rf(X_tr, y_tr_44, y_tr_70, X_te):
    """M1: Random Forest via LightGBM Native API (Không cần scikit-learn)."""
    dtrain_44 = lgb.Dataset(X_tr, label=y_tr_44, free_raw_data=False)
    dtrain_70 = lgb.Dataset(X_tr, label=y_tr_70, free_raw_data=False)

    params_rf = {
        "boosting_type": "rf",
        "objective": "regression",
        "learning_rate": 0.1,
        "bagging_freq": 1,
        "bagging_fraction": 0.8,
        "feature_fraction": 0.8,
        "verbose": -1,
        "num_threads": -1,
        "seed": 42
    }

    bst_44 = lgb.train(params_rf, dtrain_44, num_boost_round=100)
    bst_70 = lgb.train(params_rf, dtrain_70, num_boost_round=100)

    p44 = np.asarray(bst_44.predict(X_te), dtype=np.float64)
    p70 = np.asarray(bst_70.predict(X_te), dtype=np.float64)
    return p44, p70


def train_predict_m2_lgb(X_tr, y_tr_44, y_tr_70, X_te):
    """M2: GBDT LightGBM Native API (Poisson cho N44, L1 cho N70)."""
    dtrain_44 = lgb.Dataset(X_tr, label=y_tr_44, free_raw_data=False)
    dtrain_70 = lgb.Dataset(X_tr, label=y_tr_70, free_raw_data=False)

    params_44 = {
        "objective": "poisson",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "num_threads": -1,
        "seed": 42
    }

    params_70 = {
        "objective": "regression_l1",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 20,
        "verbose": -1,
        "num_threads": -1,
        "seed": 42
    }

    bst_44 = lgb.train(params_44, dtrain_44, num_boost_round=250)
    bst_70 = lgb.train(params_70, dtrain_70, num_boost_round=250)

    p44 = np.asarray(bst_44.predict(X_te), dtype=np.float64)
    p70 = np.asarray(bst_70.predict(X_te), dtype=np.float64)
    return p44, p70


def train_predict_m3_catboost(X_tr, y_tr_44, y_tr_70, X_te):
    """M3: CatBoost Regressor độc lập."""
    cb_44 = CatBoostRegressor(
        iterations=250,
        learning_rate=0.05,
        depth=6,
        loss_function="MAE",
        random_seed=42,
        verbose=0,
        thread_count=-1
    )
    cb_70 = CatBoostRegressor(
        iterations=250,
        learning_rate=0.03,
        depth=5,
        loss_function="MAE",
        random_seed=42,
        verbose=0,
        thread_count=-1
    )
    cb_44.fit(X_tr, y_tr_44)
    cb_70.fit(X_tr, y_tr_70)

    p44 = np.asarray(cb_44.predict(X_te), dtype=np.float64)
    p70 = np.asarray(cb_70.predict(X_te), dtype=np.float64)
    return p44, p70


def run_ml_pipeline(features_path="data/processed/features_monthly.parquet"):
    print("Đang đọc dữ liệu đặc trưng...")
    df = pd.read_parquet(features_path)
    df["month"] = pd.PeriodIndex(df["month"], freq="M")

    # Loại bỏ các cột định danh, nhãn và văn bản.
    # "expected_amount"/"amount_diff" được liệt kê phòng trường hợp chạy trên
    # một file features_monthly.parquet cũ (sinh ra trước khi sửa
    # feature_engineering.py) còn sót 2 cột rò rỉ nhãn này.
    drop_cols = [
        "employee_id", "month", "department", "N44", "N70",
        "total_meals", "total_amount", "days_with_meal", "calendar_days",
        "expected_amount", "amount_diff",
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    # Chỉ giữ lại các cột số (loại trừ cột object/string nếu có)
    numeric_feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    print(f"Tổng số đặc trưng số đưa vào huấn luyện: {len(numeric_feature_cols)}")

    test_months = [pd.Period("2026-07", "M"), pd.Period("2026-08", "M"), pd.Period("2026-09", "M")]

    print("\n" + "=" * 70)
    print("BẮT ĐẦU BACKTESTING ML: M1 (RF), M2 (LightGBM), M3 (CatBoost)")
    print("=" * 70)

    results = []
    predictions_dump = []

    for test_m in test_months:
        train_mask = df["month"] < test_m
        test_mask = df["month"] == test_m

        X_train = df.loc[train_mask, numeric_feature_cols].fillna(0)
        y44_train = df.loc[train_mask, "N44"].values
        y70_train = df.loc[train_mask, "N70"].values

        X_test = df.loc[test_mask, numeric_feature_cols].fillna(0)
        y44_test = df.loc[test_mask, "N44"].values
        y70_test = df.loc[test_mask, "N70"].values

        print(f"\n--> Đang huấn luyện và dự đoán tháng test: {test_m} (Train size: {len(X_train):,}, Test size: {len(X_test):,})")

        # 1. Chạy M1 - Random Forest
        p44_m1, p70_m1 = train_predict_m1_rf(X_train, y44_train, y70_train, X_test)
        res_m1 = calculate_metrics(y44_test, y70_test, p44_m1, p70_m1)
        res_m1.update({"Model": "M1_RandomForest", "Test_Month": str(test_m)})
        results.append(res_m1)

        # 2. Chạy M2 - LightGBM
        p44_m2, p70_m2 = train_predict_m2_lgb(X_train, y44_train, y70_train, X_test)
        res_m2 = calculate_metrics(y44_test, y70_test, p44_m2, p70_m2)
        res_m2.update({"Model": "M2_LightGBM", "Test_Month": str(test_m)})
        results.append(res_m2)

        # 3. Chạy M3 - CatBoost
        p44_m3, p70_m3 = train_predict_m3_catboost(X_train, y44_train, y70_train, X_test)
        res_m3 = calculate_metrics(y44_test, y70_test, p44_m3, p70_m3)
        res_m3.update({"Model": "M3_CatBoost", "Test_Month": str(test_m)})
        results.append(res_m3)

        # Lưu dự đoán cho bước Ensemble sau này
        fold_preds = df.loc[test_mask, ["employee_id", "month", "N44", "N70"]].copy()
        fold_preds["m1_p44"], fold_preds["m1_p70"] = p44_m1, p70_m1
        fold_preds["m2_p44"], fold_preds["m2_p70"] = p44_m2, p70_m2
        fold_preds["m3_p44"], fold_preds["m3_p70"] = p44_m3, p70_m3
        predictions_dump.append(fold_preds)

    summary_df = pd.DataFrame(results)
    avg_df = (
        summary_df.groupby("Model")[["MAE_Cost", "RMSE_Cost", "WAPE_Cost_%", "MAE_N44", "MAE_N70"]]
        .mean()
        .reset_index()
        .sort_values("MAE_Cost")
    )

    print("\n" + "=" * 70)
    print("KẾT QUẢ SO SÁNH CÁC MÔ HÌNH ML TRUNG BÌNH QUA 3 THÁNG TEST")
    print("=" * 70)
    print(
        avg_df.to_string(
            index=False,
            formatters={
                "MAE_Cost": "{:,.0f} đ".format,
                "RMSE_Cost": "{:,.0f} đ".format,
                "WAPE_Cost_%": "{:.2f}%".format,
                "MAE_N44": "{:.2f}".format,
                "MAE_N70": "{:.2f}".format,
            },
        )
    )

    baseline_file = "results/metrics/baselines_summary.csv"
    if os.path.exists(baseline_file):
        base_df = pd.read_csv(baseline_file)
        print("\n--- SO SÁNH VỚI BASELINE TỐT NHẤT ---")
        best_base = base_df.sort_values("MAE_Cost").iloc[0]
        best_ml = avg_df.iloc[0]
        diff_mae = best_base["MAE_Cost"] - best_ml["MAE_Cost"]
        pct_imp = (diff_mae / best_base["MAE_Cost"]) * 100
        print(f"Baseline tốt nhất ({best_base['Model']}): {best_base['MAE_Cost']:,.0f} đ")
        print(f"ML tốt nhất ({best_ml['Model']}): {best_ml['MAE_Cost']:,.0f} đ")
        print(f"Mức độ cải thiện: Giảm {diff_mae:,.0f} đ ({pct_imp:.2f}%)")

    os.makedirs("results/metrics", exist_ok=True)
    avg_df.to_csv("results/metrics/ml_models_summary.csv", index=False)

    all_preds_df = pd.concat(predictions_dump, ignore_index=True)
    os.makedirs("data/processed", exist_ok=True)
    all_preds_df.to_parquet("data/processed/ml_test_predictions.parquet", index=False)
    print("\nĐã lưu dự đoán chi tiết tại: data/processed/ml_test_predictions.parquet")


if __name__ == "__main__":
    run_ml_pipeline()
    