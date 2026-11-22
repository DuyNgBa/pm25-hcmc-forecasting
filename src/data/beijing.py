"""G1-45, G1-46 — Chuẩn bị dữ liệu Bắc Kinh.

    python -m src.data.beijing      # đọc 12 CSV -> beijing_raw.parquet -> beijing_clean.parquet + bảng % thiếu

Quy ước (docs/PIPELINE_GUIDE.md §2):
  - tên cột snake_case; khoá (city, station_id, time); time = giờ địa phương Bắc Kinh (UTC+8), KHÔNG kèm tz.
  - city = "beijing"; station_id = tên trạm gốc (vd. "Aotizhongxin").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.cleaning_common import complete_hourly_grid, interpolate_short_gaps, missing_table
from src.data.download_beijing import STATIONS, fname
from src.utils import ROOT, load_config, path_of

# Đổi tên cột gốc -> snake_case. Giữ đủ cột gốc ở beijing_raw (để TV2 kiểm chứng Open-Meteo, G1-35).
RENAME = {"PM2.5": "pm25", "PM10": "pm10", "SO2": "so2", "NO2": "no2", "CO": "co", "O3": "o3",
          "TEMP": "temp_native", "PRES": "pres_native", "DEWP": "dewp_native",
          "RAIN": "rain_native", "wd": "wd_native", "WSPM": "wspm_native"}


def load_raw(raw_dir) -> pd.DataFrame:
    """G1-45: đọc & gộp 12 file, tạo cột time (giờ địa phương) và station_id."""
    frames = []
    for st in STATIONS:
        d = pd.read_csv(raw_dir / fname(st))
        d["time"] = pd.to_datetime(d[["year", "month", "day", "hour"]])  # giờ địa phương, naive
        frames.append(d)
    df = pd.concat(frames, ignore_index=True).rename(columns=RENAME)
    df["city"] = "beijing"
    df = df.rename(columns={"station": "station_id"})
    df = df.drop(columns=["No", "year", "month", "day", "hour"])
    key = ["city", "station_id", "time"]
    df = df[key + [c for c in df.columns if c not in key]].sort_values(key).reset_index(drop=True)
    # Kiểm tra khoá duy nhất và lưới giờ đầy đủ
    assert not df.duplicated(key).any(), "Trùng khoá (city, station_id, time)"
    per = df.groupby("station_id")["time"].agg(["min", "max", "count"])
    expected = len(pd.date_range(per["min"].min(), per["max"].max(), freq="h"))
    assert (per["count"] == expected).all(), "Có trạm thiếu dòng -> cần complete_hourly_grid"
    return df


def clean(df_raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """G1-46: xử lý thiếu theo CÙNG quy tắc với HealthyAir.

    1) lưới giờ đầy đủ (Bắc Kinh vốn đã đủ, bước này chỉ để đảm bảo);
    2) PM2.5 <= 0 -> NaN (cùng quy tắc #6a của HealthyAir; Bắc Kinh thực tế không có);
    3) KHÔNG cắt outlier PM2.5 (giống TV1: PM2.5 là biến mục tiêu, giá trị cao là sự kiện thật cần dự báo;
       log1p khi chuẩn hoá đã nén đuôi). Chỉ GHI LẠI ngưỡng phân vị 99,9 trên train để tham khảo;
    4) nội suy tuyến tính khoảng trống <= max_interp_gap_hours, trong từng trạm; cờ is_imputed.
    Khí tượng gốc (temp_native...) được nội suy cùng quy tắc nhưng KHÔNG dùng trong lược đồ chung
    khi đã có Open-Meteo (xem schema.py).
    """
    max_gap = cfg["cleaning"]["max_interp_gap_hours"]
    q = cfg["cleaning"]["outlier_quantile"]
    tr_s, tr_e = (pd.Timestamp(x) for x in cfg["splits"]["beijing"]["train"])

    df = complete_hourly_grid(df_raw)
    log = {"n_rows": int(len(df))}

    nonpos = df["pm25"] <= 0
    log["pm25_nonpositive_to_nan"] = int(nonpos.sum())
    df.loc[nonpos, "pm25"] = np.nan

    in_train = df["time"].between(tr_s, tr_e)
    thr = float(df.loc[in_train, "pm25"].quantile(q))
    log.update({"pm25_train_q999_tham_khao": round(thr, 2), "pm25_gio_vuot_q999": int((df["pm25"] > thr).sum()),
                "pm25_clipped": False})

    out = []
    for st, g in df.groupby("station_id", sort=True):
        g = g.sort_values("time").reset_index(drop=True)
        g["pm25"], g["is_imputed"] = interpolate_short_gaps(g["pm25"], max_gap)
        for c in ["temp_native", "pres_native", "dewp_native", "rain_native", "wspm_native"]:
            g[c], _ = interpolate_short_gaps(g[c], max_gap)
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    log["pm25_imputed_rows"] = int(df["is_imputed"].sum())
    return df, log


def main():
    cfg = load_config()
    raw = load_raw(ROOT / cfg["paths"]["raw_beijing"])
    raw.to_parquet(path_of(cfg, "beijing_raw_parquet"), index=False)
    cols = ["pm25", "temp_native", "pres_native", "dewp_native", "rain_native", "wd_native", "wspm_native"]
    before = missing_table(raw, cols)
    clean_df, log = clean(raw, cfg)
    clean_df.to_parquet(path_of(cfg, "beijing_clean_parquet"), index=False)
    after = missing_table(clean_df, cols)
    tbl = before.add_suffix("_truoc").join(after.add_suffix("_sau"))
    tbl = tbl[[c for pair in zip(before.columns + "_truoc", after.columns + "_sau") for c in pair]]
    out = ROOT / "reports" / "beijing_missing_before_after.csv"
    tbl.to_csv(out)
    print("Log làm sạch:", log)
    print(tbl[["pm25_truoc", "pm25_sau"]])
    print(f"Đã lưu {out}")


if __name__ == "__main__":
    main()
