import pandas as pd
from pathlib import Path


PRICE_N44 = 44_000
PRICE_N70 = 70_000


def create_monthly_dataset(grid_path, output_path):
    print("Đọc daily grid...")
    df = pd.read_parquet(grid_path)

    df["date"] = pd.to_datetime(df["date"])

    # Tháng
    df["month"] = df["date"].dt.to_period("M")

    # N44 = Việt + Chay
    df["n44"] = df["n_viet"] + df["n_chay"]

    # N70 = Âu
    df["n70"] = df["n_au"]

    # Aggregate employee-month
    monthly = (
        df.groupby(
            ["employee_id", "department", "month"],
            as_index=False
        )
        .agg(
            N44=("n44", "sum"),
            N70=("n70", "sum"),
            total_meals=("n_meals", "sum"),
            total_amount=("total_amount", "sum"),
            days_with_meal=("has_checkin", "sum"),
            calendar_days=("date", "count"),
        )
    )

    # Kiểm tra số tiền theo giá cố định
    monthly["expected_amount"] = (
        monthly["N44"] * PRICE_N44
        + monthly["N70"] * PRICE_N70
    )

    monthly["amount_diff"] = (
        monthly["total_amount"]
        - monthly["expected_amount"]
    )

    print("\n=== MONTHLY DATASET ===")
    print(f"Employee-month: {len(monthly):,}")
    print(f"Employees: {monthly['employee_id'].nunique():,}")
    print(
        f"Months: "
        f"{monthly['month'].min()} -> {monthly['month'].max()}"
    )

    print("\n=== TARGET CHECK ===")

    mismatch = monthly[
        monthly["amount_diff"].abs() > 0.01
    ]

    print(f"Amount mismatch: {len(mismatch):,}")

    if len(mismatch) > 0:
        print(mismatch.head(10))

    # Sắp xếp
    monthly = monthly.sort_values(
        ["employee_id", "month"]
    ).reset_index(drop=True)

    # ------------------------------------------------
    # LAG FEATURES
    # ------------------------------------------------

    monthly["N44_lag1"] = (
        monthly.groupby("employee_id")["N44"]
        .shift(1)
    )

    monthly["N44_lag2"] = (
        monthly.groupby("employee_id")["N44"]
        .shift(2)
    )

    monthly["N44_lag3"] = (
        monthly.groupby("employee_id")["N44"]
        .shift(3)
    )

    monthly["N70_lag1"] = (
        monthly.groupby("employee_id")["N70"]
        .shift(1)
    )

    monthly["N70_lag2"] = (
        monthly.groupby("employee_id")["N70"]
        .shift(2)
    )

    monthly["N70_lag3"] = (
        monthly.groupby("employee_id")["N70"]
        .shift(3)
    )

    # ------------------------------------------------
    # ROLLING FEATURES
    # ------------------------------------------------

    monthly["N44_mean_3m"] = (
        monthly.groupby("employee_id")["N44"]
        .transform(
            lambda x: x.shift(1).rolling(3).mean()
        )
    )

    monthly["N70_mean_3m"] = (
        monthly.groupby("employee_id")["N70"]
        .transform(
            lambda x: x.shift(1).rolling(3).mean()
        )
    )

    # ------------------------------------------------
    # WESTERN SHARE
    # ------------------------------------------------
# Tỷ lệ bữa Âu
    monthly["au_share"] = 0.0

    has_meals = monthly["total_meals"] > 0

    monthly.loc[has_meals, "au_share"] = (
        monthly.loc[has_meals, "N70"]
        / monthly.loc[has_meals, "total_meals"]
    )

    # Tỷ lệ Âu trung bình của 3 tháng trước
    monthly["au_share_3m"] = (
        monthly.groupby("employee_id")["au_share"]
        .transform(
            lambda x: x.shift(1).rolling(3).mean()
        )
    )

    # ------------------------------------------------
    # TENURE
    # ------------------------------------------------

    first_month = (
        monthly.groupby("employee_id")["month"]
        .transform("min")
    )

    monthly["tenure_months"] = (
        monthly["month"].astype("int64")
        - first_month.astype("int64")
    )

    # ------------------------------------------------
    # SAVE
    # ------------------------------------------------

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    monthly.to_parquet(
        output_path,
        index=False
    )

    print(f"\nĐã lưu: {output_path}")

    return monthly


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--grid",
        default="data/processed/daily_grid.parquet"
    )

    parser.add_argument(
        "--output",
        default="data/processed/monthly_dataset.parquet"
    )

    args = parser.parse_args()

    create_monthly_dataset(
        args.grid,
        args.output
    )