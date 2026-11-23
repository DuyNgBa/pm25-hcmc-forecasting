"""G1-47 — Đưa Bắc Kinh và TP.HCM về CÙNG một lược đồ (mục 4.2 của đề xuất).

    python -m src.data.schema                       # build cả 2 thành phố
    python -m src.data.schema --beijing-meteo native   # khi CHƯA có Open-Meteo Bắc Kinh (chỉ để phát triển)

Đầu vào:
  - data/interim/beijing_clean.parquet         (src/data/beijing.py, TV3)
  - data/interim/healthyair_clean.parquet      (TV1). Nếu chưa có, dùng bản tạm *_DEV.parquet
                                                (scripts/dev_hcm_minimal_clean.py) và in cảnh báo.
  - data/interim/meteo_hcm.parquet, meteo_beijing.parquet  (TV2, tên biến đúng như Open-Meteo API)
Đầu ra: data/processed/{hcm,beijing}_schema.parquet — CÙNG danh sách cột, CÙNG đơn vị.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from src.utils import ROOT, load_config, path_of

# ---- Danh sách cột của lược đồ chung (thứ tự cố định) ----
KEY = ["city", "station_id", "time"]
TARGET = ["pm25", "is_imputed"]
METEO = ["temp", "rh", "dew", "wind_speed", "wind_u", "wind_v", "precip", "precip_6h",
         "precip_24h", "pressure"]
TIME_FEATS = ["hour_sin", "hour_cos", "dow", "dow_sin", "dow_cos"]
# Cột phụ: chỉ có ý nghĩa ở TP.HCM, Bắc Kinh để NaN/0 để hai file CÙNG cột.
EXTRA = ["temp_ha", "rh_ha",          # khí tượng gốc HealthyAir (ablation b)
         "is_lockdown",               # biến chỉ báo giãn cách (TV1, G1-08)
         "meteo_source"]              # "open_meteo" | "beijing_native" | "missing"
SCHEMA_COLS = KEY + TARGET + METEO + TIME_FEATS + EXTRA
# Lưu ý: KHÔNG có "month" và các chất ô nhiễm khác (NO2, CO, O3, SO2, TSP) — đúng mục 4.2.

# Tên biến Open-Meteo (Historical Weather API) -> tên trong lược đồ
OPENMETEO_MAP = {"temperature_2m": "temp", "relative_humidity_2m": "rh", "dew_point_2m": "dew",
                 "precipitation": "precip", "surface_pressure": "pressure",
                 "wind_speed_10m": "wind_speed", "wind_direction_10m": "wind_dir"}

# 16 hướng la bàn của bộ Bắc Kinh -> độ (hướng gió THỔI TỪ, tính theo chiều kim đồng hồ từ Bắc)
COMPASS16 = {d: i * 22.5 for i, d in enumerate(
    ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"])}


def wind_uv(speed, direction_deg):
    """Tách gió thành 2 thành phần (quy ước khí tượng).

    direction_deg là hướng gió THỔI TỪ (gió Bắc = 0° thổi từ Bắc xuống Nam).
      u = -speed * sin(θ)   (dương = gió thổi về phía Đông)
      v = -speed * cos(θ)   (dương = gió thổi về phía Bắc)
    Ví dụ: gió Bắc 5 m/s -> (u, v) = (0, -5); gió Tây (270°) 5 m/s -> (u, v) = (+5, 0).
    Vì sao cần: 359° và 1° gần như cùng hướng nhưng là 2 số cách xa nhau; u, v liên tục và
    còn mang cả độ lớn gió.
    """
    th = np.deg2rad(np.asarray(direction_deg, dtype=float))
    sp = np.asarray(speed, dtype=float)
    return -sp * np.sin(th), -sp * np.cos(th)


def rh_from_temp_dew(t, td):
    """Độ ẩm tương đối (%) từ nhiệt độ và điểm sương (°C), công thức Magnus (hệ số Alduchov–Eskridge)."""
    a, b = 17.625, 243.04
    t, td = np.asarray(t, float), np.asarray(td, float)
    return np.clip(100 * np.exp(a * td / (b + td)) / np.exp(a * t / (b + t)), 0, 100)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Đặc trưng chu kỳ theo GIỜ ĐỊA PHƯƠNG (chu kỳ sinh hoạt gắn với giờ địa phương)."""
    h = df["time"].dt.hour
    d = df["time"].dt.dayofweek          # 0 = Thứ Hai
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    df["dow"] = d.astype(np.int8)
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)
    return df


def add_rain_accum(df: pd.DataFrame) -> pd.DataFrame:
    """Tổng mưa 6h, 24h QUA (gồm cả giờ t, chỉ nhìn quá khứ), tính trong từng trạm."""
    df = df.sort_values(KEY)
    if "precip" not in df.columns:
        df["precip"] = np.nan
    g = df.groupby(["city", "station_id"])["precip"]
    df["precip_6h"] = g.transform(lambda s: s.rolling(6, min_periods=6).sum())
    df["precip_24h"] = g.transform(lambda s: s.rolling(24, min_periods=24).sum())
    return df


def meteo_from_openmeteo(path) -> pd.DataFrame:
    m = pd.read_parquet(path)
    missing = [c for c in [*OPENMETEO_MAP, "station_id", "time"] if c not in m.columns]
    if missing:
        raise ValueError(f"{path} thiếu cột {missing}. TV2 vui lòng giữ tên biến đúng như API.")
    m = m.rename(columns=OPENMETEO_MAP)
    m["wind_u"], m["wind_v"] = wind_uv(m["wind_speed"], m["wind_dir"])
    m["time"] = pd.to_datetime(m["time"]).dt.tz_localize(None)
    m["station_id"] = m["station_id"].astype(str)
    m["meteo_source"] = "open_meteo"
    return m[["station_id", "time", "temp", "rh", "dew", "precip", "pressure",
              "wind_speed", "wind_u", "wind_v", "meteo_source"]]


def meteo_from_beijing_native(bj: pd.DataFrame) -> pd.DataFrame:
    """Khí tượng GỐC của bộ Bắc Kinh, quy về cùng biến (chỉ dùng khi chưa có Open-Meteo,
    và để TV2 kiểm chứng Open-Meteo — G1-35). Không có RH gốc -> tính từ TEMP và DEWP."""
    m = pd.DataFrame({"station_id": bj["station_id"], "time": bj["time"],
                      "temp": bj["temp_native"], "dew": bj["dewp_native"],
                      "precip": bj["rain_native"], "pressure": bj["pres_native"],
                      "wind_speed": bj["wspm_native"]})
    m["rh"] = rh_from_temp_dew(m["temp"], m["dew"])
    m["wind_u"], m["wind_v"] = wind_uv(m["wind_speed"], bj["wd_native"].map(COMPASS16))
    m["meteo_source"] = "beijing_native"
    return m


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = add_rain_accum(df)
    df = add_time_features(df)
    for c in SCHEMA_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[SCHEMA_COLS].sort_values(KEY).reset_index(drop=True)
    df["is_imputed"] = df["is_imputed"].fillna(0).astype(np.int8)
    df["is_lockdown"] = df["is_lockdown"].fillna(0).astype(np.int8)
    df["meteo_source"] = df["meteo_source"].fillna("missing").astype(str)
    return df


def build_beijing(cfg: dict, meteo: str = "auto") -> pd.DataFrame:
    bj = pd.read_parquet(path_of(cfg, "beijing_clean_parquet"))
    om_path = path_of(cfg, "meteo_beijing")
    if meteo == "open_meteo" or (meteo == "auto" and om_path.exists()):
        m = meteo_from_openmeteo(om_path)
    else:
        warnings.warn("Chưa có meteo_beijing.parquet (Open-Meteo) -> dùng khí tượng GỐC Bắc Kinh. "
                      "Chỉ hợp lệ cho phát triển code, KHÔNG dùng cho kết quả chuyển giao (mục 4.3).")
        m = meteo_from_beijing_native(bj)
    df = bj[["city", "station_id", "time", "pm25", "is_imputed"]].merge(
        m, on=["station_id", "time"], how="left")
    return finalize(df)


def load_healthyair_clean(cfg: dict) -> pd.DataFrame:
    """Đọc dữ liệu HealthyAir đã làm sạch của TV1, chuẩn hoá tên cột tối thiểu.

    Giao diện mong đợi từ TV1 (thống nhất trong docs/PIPELINE_GUIDE.md §2):
      time | station_id ("hcm_01".."hcm_06") | pm25 | is_imputed |
      temp_sensor, rh_sensor (khí tượng gốc, đã NaN các đoạn hỏng) | is_lockdown
    (đúng file data/interim/healthyair_clean.parquet do src/data/clean_healthyair.py của TV1 tạo ra)
    """
    p = path_of(cfg, "healthyair_clean")
    if not p.exists():
        p_dev = p.with_name(p.stem + "_DEV.parquet")
        if not p_dev.exists():
            raise FileNotFoundError(f"Chưa có {p.name}. Chạy: python src/data/clean_healthyair.py (code của TV1).")
        warnings.warn(f"Đang dùng bản TẠM {p_dev.name} (làm sạch tối thiểu). Kết quả chỉ để thử pipeline; "
                      "phải chạy lại khi TV1 xuất healthyair_clean.parquet.")
        p = p_dev
    h = pd.read_parquet(p)
    if "station_id" not in h.columns and "station_no" in h.columns:
        h["station_id"] = "hcm_" + h["station_no"].astype(int).astype(str).str.zfill(2)
    h["station_id"] = h["station_id"].astype(str)
    h = h.rename(columns={"temp_sensor": "temp_ha", "rh_sensor": "rh_ha", "temp": "temp_ha", "rh": "rh_ha"})
    h["city"] = "hcm"
    return h


def build_hcm(cfg: dict) -> pd.DataFrame:
    h = load_healthyair_clean(cfg)
    keep = [c for c in ["city", "station_id", "time", "pm25", "is_imputed", "temp_ha", "rh_ha", "is_lockdown"]
            if c in h.columns]
    df = h[keep]
    om_path = path_of(cfg, "meteo_hcm")
    if om_path.exists():
        df = df.merge(meteo_from_openmeteo(om_path), on=["station_id", "time"], how="left")
    else:
        warnings.warn("Chưa có meteo_hcm.parquet (TV2) -> các cột khí tượng = NaN; chỉ chạy được ablation (a).")
    return finalize(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beijing-meteo", default="auto", choices=["auto", "open_meteo", "native"])
    args = ap.parse_args()
    cfg = load_config()
    bj = build_beijing(cfg, args.beijing_meteo)
    hcm = build_hcm(cfg)
    assert list(bj.columns) == list(hcm.columns) == SCHEMA_COLS, "Hai lược đồ phải CÙNG cột"
    bj.to_parquet(path_of(cfg, "beijing_schema"), index=False)
    hcm.to_parquet(path_of(cfg, "hcm_schema"), index=False)
    for name, d in [("beijing", bj), ("hcm", hcm)]:
        print(f"{name}: {d.shape}, meteo_source={d['meteo_source'].unique().tolist()}, "
              f"% thiếu pm25={d['pm25'].isna().mean() * 100:.2f}")


if __name__ == "__main__":
    main()
