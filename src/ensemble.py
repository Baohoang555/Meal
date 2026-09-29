import os
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import minimize

PRICE_44 = 44000
PRICE_70 = 70000


def calculate_metrics(y_true_n44, y_true_n70, y_pred_n44, y_pred_n70):
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


def optimize_e1_weight(p_a_44, p_a_70, p_b_44, p_b_70, y44, y70):
    """Tìm trọng số w tối ưu MAE chi phí: w*A + (1-w)*B"""
    cost_true = y44 * PRICE_44 + y70 * PRICE_70
    cost_a = p_a_44 * PRICE_44 + p_a_70 * PRICE_70
    cost_b = p_b_44 * PRICE_44 + p_b_70 * PRICE_70

    def loss(w):
        pred_cost = w[0] * cost_a + (1 - w[0]) * cost_b
        return np.mean(np.abs(cost_true - pred_cost))

    res = minimize(loss, [0.5], bounds=[(0.0, 1.0)], method="L-BFGS-B")
    return float(res.x[0])


def fit_convex_weights(cost_matrix, y_cost):
    """Fit non-negative weights using only predictions from prior periods."""
    if len(y_cost) == 0:
        return np.full(cost_matrix.shape[1], 1.0 / cost_matrix.shape[1])

    def loss(weights):
        return np.mean(np.abs(y_cost - cost_matrix @ weights))

    result = minimize(
        loss,
        np.full(cost_matrix.shape[1], 1.0 / cost_matrix.shape[1]),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * cost_matrix.shape[1],
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
    )
    if not result.success:
        return np.full(cost_matrix.shape[1], 1.0 / cost_matrix.shape[1])
    return result.x


def run_ensemble():
    print("Đang đọc dữ liệu dự đoán out-of-fold...")
    ml_preds = pd.read_parquet("data/processed/ml_test_predictions.parquet")
    dl_preds = pd.read_parquet("data/processed/dl_test_predictions.parquet")

    # Ghép dự đoán của ML và DL theo employee_id và month
    df = ml_preds.merge(dl_preds, on=["employee_id", "month"], how="inner")
    df["month"] = pd.PeriodIndex(df["month"], freq="M")

    models = ["m1", "m2", "m3", "m4", "m5"]
    model_names = {
        "m1": "M1(RF)",
        "m2": "M2(LGB)",
        "m3": "M3(CatB)",
        "m4": "M4(GRU)",
        "m5": "M5(MLP)",
    }

    pairs = list(itertools.combinations(models, 2))
    months = sorted(df["month"].unique())

    print("\n" + "=" * 75)
    print("ĐÁNH GIÁ 10 CẶP ENSEMBLE (E0: Simple Avg & E1: Optimal Weighted)")
    print("=" * 75)

    pair_results = []

    for m_a, m_b in pairs:
        pair_label = f"{model_names[m_a]} + {model_names[m_b]}"
        
        # Đánh giá qua từng tháng test
        e0_metrics_list = []
        e1_metrics_list = []

        for m in months:
            sub = df[df["month"] == m]
            y44, y70 = sub["N44"].values, sub["N70"].values

            pa_44, pa_70 = sub[f"{m_a}_p44"].values, sub[f"{m_a}_p70"].values
            pb_44, pb_70 = sub[f"{m_b}_p44"].values, sub[f"{m_b}_p70"].values

            # --- E0: Simple Average ---
            p_e0_44 = (pa_44 + pb_44) / 2.0
            p_e0_70 = (pa_70 + pb_70) / 2.0
            res_e0 = calculate_metrics(y44, y70, p_e0_44, p_e0_70)
            e0_metrics_list.append(res_e0)

            # --- E1: Weighted Average ---
            prior = df[df["month"] < m]
            if prior.empty:
                w_opt = 0.5
            else:
                w_opt = optimize_e1_weight(
                    prior[f"{m_a}_p44"].values,
                    prior[f"{m_a}_p70"].values,
                    prior[f"{m_b}_p44"].values,
                    prior[f"{m_b}_p70"].values,
                    prior["N44"].values,
                    prior["N70"].values,
                )
            p_e1_44 = w_opt * pa_44 + (1 - w_opt) * pb_44
            p_e1_70 = w_opt * pa_70 + (1 - w_opt) * pb_70
            res_e1 = calculate_metrics(y44, y70, p_e1_44, p_e1_70)
            res_e1["Weight_A"] = w_opt
            e1_metrics_list.append(res_e1)

        # Trung bình qua 3 tháng
        avg_e0 = pd.DataFrame(e0_metrics_list).mean().to_dict()
        avg_e0.update({"Pair": pair_label, "Technique": "E0 (Simple Avg)"})
        pair_results.append(avg_e0)

        avg_e1 = pd.DataFrame(e1_metrics_list).mean().to_dict()
        avg_e1.update({"Pair": pair_label, "Technique": "E1 (Optimal Weight)"})
        pair_results.append(avg_e1)

    df_pairs = pd.DataFrame(pair_results)
    df_sorted = df_pairs.sort_values("MAE_Cost").reset_index(drop=True)

    print("\n--- TOP 10 CẤU HÌNH CẶP ĐÔI HIỆU QUẢ NHẤT ---")
    print(
        df_sorted.head(10)[["Pair", "Technique", "MAE_Cost", "RMSE_Cost", "WAPE_Cost_%", "MAE_N44", "MAE_N70"]].to_string(
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

    # ============================================================
    # MỞ RỘNG: ENSEMBLE NHIỀU MÔ HÌNH (S0 & S2)
    # ============================================================
    print("\n" + "=" * 75)
    print("MỞ RỘNG: ENSEMBLE TỔ HỢP ĐA MÔ HÌNH (S0 & S2)")
    print("=" * 75)

    multi_results = []
    for m in months:
        sub = df[df["month"] == m]
        y44, y70 = sub["N44"].values, sub["N70"].values
        y_cost = y44 * PRICE_44 + y70 * PRICE_70

        # S0: Trung bình cộng 5 mô hình
        p_s0_44 = np.mean([sub[f"{mod}_p44"].values for mod in models], axis=0)
        p_s0_70 = np.mean([sub[f"{mod}_p70"].values for mod in models], axis=0)
        res_s0 = calculate_metrics(y44, y70, p_s0_44, p_s0_70)
        res_s0["Method"] = "S0 (All 5 Simple Avg)"
        multi_results.append(res_s0)

        # S2: trọng số học từ các tháng trước, không dùng nhãn của tháng đang test.
        prior = df[df["month"] < m]
        prior_cost_matrix = np.column_stack([
            prior[f"{mod}_p44"].values * PRICE_44 + prior[f"{mod}_p70"].values * PRICE_70
            for mod in models
        ]) if not prior.empty else np.empty((0, len(models)))
        prior_cost = (
            prior["N44"].values * PRICE_44 + prior["N70"].values * PRICE_70
            if not prior.empty else np.array([])
        )
        weights = fit_convex_weights(prior_cost_matrix, prior_cost)

        p_s2_44 = np.sum([weights[i] * sub[f"{models[i]}_p44"].values for i in range(5)], axis=0)
        p_s2_70 = np.sum([weights[i] * sub[f"{models[i]}_p70"].values for i in range(5)], axis=0)
        res_s2 = calculate_metrics(y44, y70, p_s2_44, p_s2_70)
        res_s2["Method"] = "S2 (5-Model Convex Opt)"
        multi_results.append(res_s2)

    df_multi = pd.DataFrame(multi_results).groupby("Method")[["MAE_Cost", "RMSE_Cost", "WAPE_Cost_%", "MAE_N44", "MAE_N70"]].mean().reset_index()

    print(
        df_multi.to_string(
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

    # Lưu kết quả
    os.makedirs("results/metrics", exist_ok=True)
    df_sorted.to_csv("results/metrics/ensemble_pairs_summary.csv", index=False)
    df_multi.to_csv("results/metrics/ensemble_multi_summary.csv", index=False)
    print("\nĐã lưu kết quả tại results/metrics/ensemble_pairs_summary.csv và ensemble_multi_summary.csv")


if __name__ == "__main__":
    run_ensemble()