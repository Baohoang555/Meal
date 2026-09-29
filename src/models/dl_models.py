import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

PRICE_44 = 44000
PRICE_70 = 70000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


# ============================================================
# KIẾN TRÚC M4: GRU MULTI-TASK REGRESSOR
# ============================================================
class GRUMultiTask(nn.Module):
    def __init__(self, seq_features=5, hidden_dim=64, num_layers=2, static_dim=10):
        super().__init__()
        self.gru = nn.GRU(
            input_size=seq_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc_static = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.ReLU(),
        )
        self.head_44 = nn.Sequential(
            nn.Linear(hidden_dim + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.head_70 = nn.Sequential(
            nn.Linear(hidden_dim + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_seq, x_static):
        out_gru, _ = self.gru(x_seq)
        last_hidden = out_gru[:, -1, :]
        stat_out = self.fc_static(x_static)
        combined = torch.cat([last_hidden, stat_out], dim=1)
        pred_44 = self.head_44(combined)
        pred_70 = self.head_70(combined)
        return pred_44.squeeze(-1), pred_70.squeeze(-1)


# ============================================================
# KIẾN TRÚC M5: RESIDUAL TABULAR MLP
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.head_44 = nn.Linear(hidden_dim, 1)
        self.head_70 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_layer(x)
        h = self.res1(h)
        h = self.res2(h)
        return self.head_44(h).squeeze(-1), self.head_70(h).squeeze(-1)


# ============================================================
# TRAINING LOOPS VỚI HUBER LOSS
# ============================================================
def train_m4_gru(train_seq, train_stat, y44_tr, y70_tr, test_seq, test_stat, epochs=30, batch_size=256):
    dataset = TensorDataset(
        torch.tensor(train_seq, dtype=torch.float32),
        torch.tensor(train_stat, dtype=torch.float32),
        torch.tensor(y44_tr, dtype=torch.float32),
        torch.tensor(y70_tr, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GRUMultiTask(
        seq_features=train_seq.shape[2],
        static_dim=train_stat.shape[1],
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    model.train()
    for _ in range(epochs):
        for b_seq, b_stat, b_y44, b_y70 in loader:
            b_seq, b_stat = b_seq.to(DEVICE), b_stat.to(DEVICE)
            b_y44, b_y70 = b_y44.to(DEVICE), b_y70.to(DEVICE)

            optimizer.zero_grad()
            p44, p70 = model(b_seq, b_stat)
            loss = loss_fn(p44, b_y44) + 1.5 * loss_fn(p70, b_y70)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        t_seq = torch.tensor(test_seq, dtype=torch.float32).to(DEVICE)
        t_stat = torch.tensor(test_stat, dtype=torch.float32).to(DEVICE)
        p44, p70 = model(t_seq, t_stat)
        p44 = p44.cpu().numpy()
        p70 = p70.cpu().numpy()

    return p44, p70


def train_m5_mlp(train_x, y44_tr, y70_tr, test_x, epochs=35, batch_size=256):
    dataset = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(y44_tr, dtype=torch.float32),
        torch.tensor(y70_tr, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TabularMLP(input_dim=train_x.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    model.train()
    for _ in range(epochs):
        for bx, b_y44, b_y70 in loader:
            bx = bx.to(DEVICE)
            b_y44, b_y70 = b_y44.to(DEVICE), b_y70.to(DEVICE)

            optimizer.zero_grad()
            p44, p70 = model(bx)
            loss = loss_fn(p44, b_y44) + 1.5 * loss_fn(p70, b_y70)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        tx = torch.tensor(test_x, dtype=torch.float32).to(DEVICE)
        p44, p70 = model(tx)
        p44 = p44.cpu().numpy()
        p70 = p70.cpu().numpy()

    return p44, p70


# ============================================================
# PIPELINE BACKTEST DL
# ============================================================
def run_dl_pipeline(features_path="data/processed/features_monthly.parquet"):
    print(f"Khởi chạy Deep Learning trên thiết bị: {DEVICE}")
    df = pd.read_parquet(features_path)
    df["month"] = pd.PeriodIndex(df["month"], freq="M")

    drop_cols = [
        "employee_id", "month", "department", "N44", "N70",
        "total_meals", "total_amount", "days_with_meal", "calendar_days"
    ]
    numeric_feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in drop_cols
    ]

    test_months = [pd.Period("2026-07", "M"), pd.Period("2026-08", "M"), pd.Period("2026-09", "M")]

    print("\n" + "=" * 70)
    print("BẮT ĐẦU BACKTESTING DL: M4 (GRU) & M5 (Tabular MLP)")
    print("=" * 70)

    results = []
    dl_predictions = []

    for test_m in test_months:
        train_mask = df["month"] < test_m
        test_mask = df["month"] == test_m

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        y44_tr, y70_tr = train_df["N44"].values, train_df["N70"].values
        y44_te, y70_te = test_df["N44"].values, test_df["N70"].values

        # Chuẩn hóa Z-score tĩnh dựa trên Train Set
        X_train_raw = train_df[numeric_feature_cols].fillna(0).values
        X_test_raw = test_df[numeric_feature_cols].fillna(0).values

        mean = np.mean(X_train_raw, axis=0)
        std = np.std(X_train_raw, axis=0) + 1e-6
        X_tr_scaled = (X_train_raw - mean) / std
        X_te_scaled = (X_test_raw - mean) / std

        # Trích xuất dạng Sequence 3 tháng cho GRU: [lag_3, lag_2, lag_1]
        step_features = ["N44_lag_", "N70_lag_", "total_meals_lag_", "days_with_meal_lag_", "total_amount_lag_"]
        seq_train = np.stack(
            [
                train_df[[f"{f}{step}" for f in step_features]].fillna(0).values
                for step in [3, 2, 1]
            ],
            axis=1,
        )
        seq_test = np.stack(
            [
                test_df[[f"{f}{step}" for f in step_features]].fillna(0).values
                for step in [3, 2, 1]
            ],
            axis=1,
        )

        # Chuẩn hóa Sequence
        seq_mean = np.mean(seq_train, axis=(0, 1), keepdims=True)
        seq_std = np.std(seq_train, axis=(0, 1), keepdims=True) + 1e-6
        seq_train_norm = (seq_train - seq_mean) / seq_std
        seq_test_norm = (seq_test - seq_mean) / seq_std

        static_train_norm = X_tr_scaled[:, :10]
        static_test_norm = X_te_scaled[:, :10]

        print(f"\n--> Đang huấn luyện Fold: {test_m}")

        # Huấn luyện M4 (GRU) qua 3 seeds
        preds_m4_44, preds_m4_70 = [], []
        for s in [42, 100, 2024]:
            seed_everything(s)
            p44, p70 = train_m4_gru(
                seq_train_norm, static_train_norm, y44_tr, y70_tr,
                seq_test_norm, static_test_norm, epochs=25
            )
            preds_m4_44.append(p44)
            preds_m4_70.append(p70)
        p44_m4 = np.mean(preds_m4_44, axis=0)
        p70_m4 = np.mean(preds_m4_70, axis=0)

        res_m4 = calculate_metrics(y44_te, y70_te, p44_m4, p70_m4)
        res_m4.update({"Model": "M4_GRU", "Test_Month": str(test_m)})
        results.append(res_m4)

        # Huấn luyện M5 (MLP) qua 3 seeds
        preds_m5_44, preds_m5_70 = [], []
        for s in [42, 100, 2024]:
            seed_everything(s)
            p44, p70 = train_m5_mlp(X_tr_scaled, y44_tr, y70_tr, X_te_scaled, epochs=30)
            preds_m5_44.append(p44)
            preds_m5_70.append(p70)
        p44_m5 = np.mean(preds_m5_44, axis=0)
        p70_m5 = np.mean(preds_m5_70, axis=0)

        res_m5 = calculate_metrics(y44_te, y70_te, p44_m5, p70_m5)
        res_m5.update({"Model": "M5_MLP", "Test_Month": str(test_m)})
        results.append(res_m5)

        fold_preds = test_df[["employee_id", "month"]].copy()
        fold_preds["m4_p44"], fold_preds["m4_p70"] = p44_m4, p70_m4
        fold_preds["m5_p44"], fold_preds["m5_p70"] = p44_m5, p70_m5
        dl_predictions.append(fold_preds)

    summary_df = pd.DataFrame(results)
    avg_df = (
        summary_df.groupby("Model")[["MAE_Cost", "RMSE_Cost", "WAPE_Cost_%", "MAE_N44", "MAE_N70"]]
        .mean()
        .reset_index()
        .sort_values("MAE_Cost")
    )

    print("\n" + "=" * 70)
    print("KẾT QUẢ SO SÁNH CÁC MÔ HÌNH DL TRUNG BÌNH QUA 3 THÁNG TEST")
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

    os.makedirs("results/metrics", exist_ok=True)
    avg_df.to_csv("results/metrics/dl_models_summary.csv", index=False)

    all_dl_preds = pd.concat(dl_predictions, ignore_index=True)
    all_dl_preds.to_parquet("data/processed/dl_test_predictions.parquet", index=False)
    print("\nĐã lưu dự đoán DL tại: data/processed/dl_test_predictions.parquet")


if __name__ == "__main__":
    run_dl_pipeline()