"""# Dự đoán tiền ăn hàng tháng của nhân viên

Kiểm tra nhanh toàn bộ artifact kết quả của project.

Chạy từ thư mục gốc:
    python scripts/test_results.py
"""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Thiếu file kết quả: {path}")


def check_metrics(path: Path, required_columns: set[str]) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path)
    missing = required_columns - set(df.columns)
    if missing:
        raise AssertionError(f"{path.name} thiếu cột: {sorted(missing)}")
    if df.empty:
        raise AssertionError(f"{path.name} không có dữ liệu")
    if df["MAE_Cost"].isna().any() or (df["MAE_Cost"] < 0).any():
        raise AssertionError(f"{path.name} có MAE_Cost không hợp lệ")
    return df


def main() -> int:
    processed = ROOT / "data" / "processed"
    metrics = ROOT / "results" / "metrics"

    for path in [
        processed / "daily_grid.parquet",
        processed / "monthly_dataset.parquet",
        processed / "features_monthly.parquet",
        processed / "ml_test_predictions.parquet",
        processed / "dl_test_predictions.parquet",
        ROOT / "results" / "eda_report.md",
    ]:
        require_file(path)

    monthly = pd.read_parquet(processed / "monthly_dataset.parquet")
    features = pd.read_parquet(processed / "features_monthly.parquet")
    ml_preds = pd.read_parquet(processed / "ml_test_predictions.parquet")
    dl_preds = pd.read_parquet(processed / "dl_test_predictions.parquet")

    for name, df in [
        ("monthly_dataset", monthly),
        ("features_monthly", features),
        ("ml_predictions", ml_preds),
        ("dl_predictions", dl_preds),
    ]:
        if df.empty:
            raise AssertionError(f"{name} không có dữ liệu")

    if "au_share" in features.columns:
        raise AssertionError("features_monthly còn au_share của tháng mục tiêu")

    lag_columns = [column for column in features.columns if "_lag_" in column]
    if not lag_columns or features[lag_columns].isna().any().any():
        raise AssertionError("Feature lag bị thiếu hoặc không tồn tại")

    for name, df in [
        ("monthly_dataset", monthly),
        ("features_monthly", features),
        ("ml_predictions", ml_preds),
        ("dl_predictions", dl_preds),
    ]:
        if df.duplicated(["employee_id", "month"]).any():
            raise AssertionError(f"{name} có employee-month trùng")

    required_metric_columns = {
        "Model",
        "MAE_Cost",
        "RMSE_Cost",
        "WAPE_Cost_%",
        "MAE_N44",
        "MAE_N70",
    }
    baseline = check_metrics(metrics / "baselines_summary.csv", required_metric_columns)
    ml = check_metrics(metrics / "ml_models_summary.csv", required_metric_columns)
    dl = check_metrics(metrics / "dl_models_summary.csv", required_metric_columns)
    ensemble = check_metrics(metrics / "ensemble_multi_summary.csv", {
        "Method",
        "MAE_Cost",
        "RMSE_Cost",
        "WAPE_Cost_%",
        "MAE_N44",
        "MAE_N70",
    })

    budget = pd.read_csv(metrics / "company_budget_eval.csv")
    if budget.empty or budget["Tien_Thuc_Te"].le(0).any():
        raise AssertionError("Company budget evaluation không hợp lệ")
    if budget["Ti_Le_Lech_%"].abs().max() > 10:
        raise AssertionError("Sai lệch ngân sách vượt 10%, cần kiểm tra lại")

    print("PASS: artifacts tồn tại và đọc được")
    print(f"PASS: monthly rows={len(monthly):,}, features rows={len(features):,}")
    print(f"PASS: prediction rows ML={len(ml_preds):,}, DL={len(dl_preds):,}")
    best_baseline = str(
        baseline.sort_values("MAE_Cost", kind="stable").iloc[0]["Model"]
    )
    best_ml = str(ml.sort_values("MAE_Cost", kind="stable").iloc[0]["Model"])
    best_dl = str(dl.sort_values("MAE_Cost", kind="stable").iloc[0]["Model"])
    best_ensemble = str(
        ensemble.sort_values("MAE_Cost", kind="stable").iloc[0]["Method"]
    )
    print(f"BEST BASELINE: {best_baseline}")
    print(f"BEST ML: {best_ml}")
    print(f"BEST DL: {best_dl}")
    print(f"BEST ENSEMBLE: {best_ensemble}")
    print(
        "BUDGET MAX ABS ERROR: "
        f"{budget['Ti_Le_Lech_%'].abs().max():.3f}%"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError, KeyError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
