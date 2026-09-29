# """"""
# # Cách dùng:
#     python data_cleaning.py --input data/raw/MealHistory_UTF8.csv \
                            #   --output data/processed/daily_grid.parquet \
                            #   --removed-log data/processed/removed_rows.csv
# """

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 1. CẤU HÌNH
# ----------------------------------------------------------------------

COLUMN_MAP = {
    "employee_id": "Mã chấm công",
    "department": "Phòng ban",
    "meal_type_raw": "Khẩu phần ăn",
    "bonus_meal_raw": "Khẩu phần ăn bồi dưỡng",   # chưa dùng tới, giữ để tham khảo
    "shift_raw": "Ca ăn",
    "amount": "Thành tiền",
    "cumulative_raw": "Tổng tích lũy",             # chưa rõ ý nghĩa, giữ để tham khảo
    "checkin_time": "Thời gian ăn",
    "status": "Trạng thái",
    "reason": "Lý do",
}

# Trạng thái được coi là một bữa ăn thật
VALID_STATUS = "Hợp lệ"

# Nhận diện loại món qua nhãn TIẾNG ANH trong ngoặc — đáng tin hơn phần
# tiếng Việt vì không bị lỗi dấu.
MEAL_TYPE_MAP = {
    "Vietnamese Meal": "Việt",
    "Vegetarian Meal": "Chay",
    "Western Meal": "Âu",
}

# Đơn giá CHỈ dùng để đối chiếu/QA với cột "Thành tiền" thật, không dùng
# để tự tính tiền.
PRICE_MAP = {"Việt": 44_000, "Chay": 44_000, "Âu": 70_000}

# Nhận diện ca ăn qua nhãn tiếng Anh trong ngoặc. "Service Shift" không có
# khung giờ cố định trong dữ liệu nguồn nên để None.
SHIFT_MAP = {
    "Shift 1":           {"start": "05:20", "end": "14:20"},
    "Office Hour Shift": {"start": "07:00", "end": "17:00"},  # giờ vào linh hoạt 07:00/07:30
    "Shift 2":           {"start": "14:40", "end": "23:40"},
    "Service Shift":     {"start": None, "end": None},
}

# Danh sách ngày lễ VN — PLACEHOLDER, cần thay bằng lịch nghỉ chính thức
# của công ty cho đúng khoảng thời gian dữ liệu thật.
PUBLIC_HOLIDAYS = pd.to_datetime([
    # --- 2025 ---
    "2025-01-01",
    "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31", "2025-02-01", "2025-02-02", "2025-02-03",
    "2025-04-30", "2025-05-01",
    "2025-09-02",
    # --- 2026 --- (theo Thông báo 9441/TB-BNV, Bộ Nội vụ, 16/10/2025 — vẫn
    # cần đối chiếu với lịch nghỉ CHÍNH THỨC của công ty, vì doanh nghiệp
    # được quyền tự quyết định lịch nghỉ Tết/Quốc khánh khác khối nhà nước)
    "2026-01-01",                                                          # Tết Dương lịch
    "2026-02-14", "2026-02-15", "2026-02-16", "2026-02-17", "2026-02-18",  # Tết Bính Ngọ
    "2026-02-19", "2026-02-20", "2026-02-21", "2026-02-22",
    "2026-04-26", "2026-04-27",                                            # Giỗ Tổ Hùng Vương (+nghỉ bù)
    "2026-04-30", "2026-05-01", "2026-05-02", "2026-05-03",                # 30/4 - 1/5 (kỳ nghỉ dài)
    "2026-08-31", "2026-09-01", "2026-09-02",                              # Quốc khánh — NGUỒN CÒN CHÊNH
    # LỆCH: một số nguồn báo nghỉ từ 29/8, cần xác nhận lại với lịch công ty.
])

_PAREN_RE = re.compile(r"\(([^)]+)\)")


# ----------------------------------------------------------------------
# 2. ĐỌC DỮ LIỆU THÔ
# ----------------------------------------------------------------------

def load_raw_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")

    missing = [c for c in COLUMN_MAP.values() if c not in df.columns]
    if missing:
        raise ValueError(
            f"Không tìm thấy các cột {missing} trong file input. "
            f"Kiểm tra lại COLUMN_MAP cho khớp với file thật."
        )

    inv_map = {v: k for k, v in COLUMN_MAP.items()}
    df = df.rename(columns=inv_map)
    return df[list(COLUMN_MAP.keys())].copy()


# ----------------------------------------------------------------------
# 3. TRÍCH NHÃN TIẾNG ANH TRONG NGOẶC (dùng chung cho loại món & ca ăn)
# ----------------------------------------------------------------------

def _extract_paren_label(text) -> str | float:
    """Lấy nội dung trong dấu ngoặc đơn ĐẦU TIÊN của chuỗi.
    Trả về NaN nếu không tìm thấy hoặc input rỗng."""
    if pd.isna(text):
        return np.nan
    m = _PAREN_RE.search(str(text))
    return m.group(1).strip() if m else np.nan


def standardize_meal_type(series: pd.Series) -> pd.Series:
    labels = series.map(_extract_paren_label)
    return labels.map(MEAL_TYPE_MAP)  # không khớp -> NaN, để lộ ra khi QA


def standardize_shift(series: pd.Series) -> pd.Series:
    labels = series.map(_extract_paren_label)
    # chuẩn hóa những nhãn nằm trong SHIFT_MAP, còn lại để NaN cho dễ phát hiện
    return labels.where(labels.isin(SHIFT_MAP.keys()), np.nan)


# ----------------------------------------------------------------------
# 4. LÀM SẠCH
# ----------------------------------------------------------------------

def clean_log(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trả về (log_sach, log_bi_loai)."""
    df = raw.copy()
    removed_reasons = []

    # --- Bước 1 (QUAN TRỌNG NHẤT): chỉ giữ lượt ăn Hợp lệ ---
    invalid_mask = df["status"] != VALID_STATUS
    removed_reasons.append(
        df[invalid_mask].assign(drop_reason="trang_thai_khong_hop_le")
    )
    df = df[~invalid_mask]

    # --- ép kiểu mã nhân viên về int, tránh lệch "3905.0" vs "3905" ---
    df["employee_id"] = pd.to_numeric(df["employee_id"], errors="coerce").astype("Int64")
    bad_emp = df["employee_id"].isna()
    removed_reasons.append(df[bad_emp].assign(drop_reason="thieu_ma_nhan_vien"))
    df = df[~bad_emp]

    # --- parse thời gian: định dạng M/D/YYYY (kiểu Mỹ), KHÔNG dayfirst ---
    df["checkin_time"] = pd.to_datetime(df["checkin_time"], errors="coerce", dayfirst=False)
    bad_time = df["checkin_time"].isna()
    removed_reasons.append(df[bad_time].assign(drop_reason="khong_parse_duoc_thoi_gian"))
    df = df[~bad_time]

    # --- chuẩn hóa loại món qua nhãn tiếng Anh ---
    df["meal_type"] = standardize_meal_type(df["meal_type_raw"])
    bad_meal = df["meal_type"].isna()
    removed_reasons.append(df[bad_meal].assign(drop_reason="khong_nhan_dien_duoc_loai_mon"))
    df = df[~bad_meal]

    # --- chuẩn hóa ca ăn qua nhãn tiếng Anh (không loại dòng nếu thiếu,
    #     chỉ đánh dấu NaN vì ca ăn không phải điều kiện bắt buộc) ---
    df["shift"] = standardize_shift(df["shift_raw"])

    # --- ép kiểu tiền, loại các dòng có amount không hợp lệ (âm, NaN) ---
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    bad_amount = df["amount"].isna() | (df["amount"] < 0)
    removed_reasons.append(df[bad_amount].assign(drop_reason="thanh_tien_khong_hop_le"))
    df = df[~bad_amount]

    # --- QA: đối chiếu Thành tiền thực tế với bảng giá đã biết ---
    expected_price = df["meal_type"].map(PRICE_MAP)
    df["price_mismatch"] = (df["amount"] != expected_price)
    n_mismatch = df["price_mismatch"].sum()
    if n_mismatch > 0:
        print(f"[CẢNH BÁO] {n_mismatch} dòng có Thành tiền KHÁC với bảng giá "
              f"44.000/70.000 đã biết — kiểm tra lại (có thể có phụ phí, "
              f"khuyến mãi, hoặc giá đã thay đổi theo thời gian).")
    dup_mask = df.duplicated(
        subset=["employee_id", "checkin_time", "meal_type", "shift"], keep="first"
    )
    removed_reasons.append(df[dup_mask].assign(drop_reason="trung_lap_ban_ghi"))
    df = df[~dup_mask]

    removed_df = pd.concat(removed_reasons, ignore_index=True) if removed_reasons else pd.DataFrame()
    return df.reset_index(drop=True), removed_df


# ----------------------------------------------------------------------
# 5. GỘP NHIỀU BỮA ĂN TRONG CÙNG 1 NGÀY -> BẢNG NGÀY
# ----------------------------------------------------------------------

def aggregate_daily(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Một nhân viên có thể ăn nhiều hơn 1 lần hợp lệ trong ngày (nhiều ca).
    Gộp lại thành 1 dòng/nhân viên/ngày: tổng tiền + số bữa từng loại."""
    df = clean_df.copy()
    df["date"] = df["checkin_time"].dt.normalize()

    daily = (
        df.groupby(["employee_id", "date"])
        .agg(
            department=("department", "last"),
            total_amount=("amount", "sum"),
            n_meals=("amount", "size"),
            n_viet=("meal_type", lambda s: (s == "Việt").sum()),
            n_chay=("meal_type", lambda s: (s == "Chay").sum()),
            n_au=("meal_type", lambda s: (s == "Âu").sum()),
        )
        .reset_index()
    )
    return daily


# ----------------------------------------------------------------------
# 6. DỰNG LƯỚI NHÂN VIÊN x NGÀY
# ----------------------------------------------------------------------

def build_employee_day_grid(daily: pd.DataFrame) -> pd.DataFrame:
    """Dựng lưới đầy đủ trong khoảng hoạt động của từng nhân viên (xấp xỉ
    theo lần check-in đầu/cuối — xem giới hạn ở README nếu không có ngày
    vào làm / nghỉ việc chính thức từ HR)."""
    global_min, global_max = daily["date"].min(), daily["date"].max()

    emp_info = (
        daily.sort_values("date")
        .groupby("employee_id")
        .agg(department=("department", "last"), first_seen=("date", "min"), last_seen=("date", "max"))
        .reset_index()
    )

    grids = []
    for _, row in emp_info.iterrows():
        start = max(row["first_seen"], global_min)
        end = min(row["last_seen"], global_max)
        dates = pd.date_range(start, end, freq="D")
        grids.append(pd.DataFrame({
            "employee_id": row["employee_id"],
            "department": row["department"],
            "date": dates,
        }))
    grid = pd.concat(grids, ignore_index=True)

    grid = grid.merge(
        daily[["employee_id", "date", "total_amount", "n_meals", "n_viet", "n_chay", "n_au"]],
        on=["employee_id", "date"], how="left",
    )
    for col in ["total_amount", "n_meals", "n_viet", "n_chay", "n_au"]:
        grid[col] = grid[col].fillna(0)
    grid["has_checkin"] = (grid["n_meals"] > 0).astype(int)

    return grid


def add_calendar_features(grid: pd.DataFrame) -> pd.DataFrame:
    grid = grid.copy()
    grid["is_weekend"] = grid["date"].dt.dayofweek.isin([5, 6])
    grid["is_public_holiday"] = grid["date"].isin(PUBLIC_HOLIDAYS)
    grid["is_workday"] = ~(grid["is_weekend"] | grid["is_public_holiday"])
    return grid


# ----------------------------------------------------------------------
# 7. HÀM CHẠY CHÍNH
# ----------------------------------------------------------------------

def run(input_path: str, output_path: str, removed_log_path: str | None = None) -> pd.DataFrame:
    raw = load_raw_log(input_path)
    clean, removed = clean_log(raw)
    daily = aggregate_daily(clean)
    grid = build_employee_day_grid(daily)
    grid = add_calendar_features(grid)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if output_path.endswith(".parquet"):
        grid.to_parquet(output_path, index=False)
    else:
        grid.to_csv(output_path, index=False)

    if removed_log_path and len(removed) > 0:
        Path(removed_log_path).parent.mkdir(parents=True, exist_ok=True)
        removed.to_csv(removed_log_path, index=False)

    print("=== TÓM TẮT LÀM SẠCH DỮ LIỆU ===")
    print(f"Số dòng log thô          : {len(raw):,}")
    print(f"Số dòng log sau làm sạch : {len(clean):,}")
    print(f"Số dòng bị loại          : {len(removed):,}")
    if len(removed) > 0:
        print(removed["drop_reason"].value_counts().to_string())
    print()
    print("=== TÓM TẮT SAU KHI GỘP THEO NGÀY ===")
    print(f"Số nhân viên          : {daily['employee_id'].nunique():,}")
    print(f"Số dòng (NV x ngày ăn): {len(daily):,}")
    print(f"Số ngày có >1 bữa     : {(daily['n_meals'] > 1).sum():,} "
          f"({(daily['n_meals'] > 1).mean():.1%})")
    print()
    print("=== TÓM TẮT LƯỚI NHÂN VIÊN x NGÀY ===")
    print(f"Số dòng lưới  : {len(grid):,}")
    print(f"Tỷ lệ có ăn   : {grid['has_checkin'].mean():.1%}")
    print(f"Khoảng ngày   : {grid['date'].min().date()} -> {grid['date'].max().date()}")

    return grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--removed-log", default=None)
    args = parser.parse_args()

    run(args.input, args.output, args.removed_log)