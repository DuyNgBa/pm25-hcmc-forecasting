"""G1-36, G1-37, G1-38, G1-41 — Tạo đặc trưng gió, mưa từ Open-Meteo và xuất meteo_{hcm,beijing}.parquet.

    python -m src.data.meteo_features      # -> data/interim/meteo_hcm.parquet, meteo_beijing.parquet

Khoá: (city, station_id, time); time là giờ địa phương, KHÔNG kèm múi giờ (quy ước G0-03).
Tên 9 biến gốc GIỮ NGUYÊN như API (temperature_2m, ...) vì src/data/schema.py của TV3 đọc đúng các tên này
và tự đổi sang tên lược đồ chung (temp, rh, ...). Các cột đặc trưng mới được thêm vào sau.

Quy ước thời gian của Open-Meteo (cần biết khi ghép với HealthyAir):
  - nhiệt độ, độ ẩm, áp suất, mây, gió: giá trị TỨC THỜI tại đúng mốc giờ.
  - precipitation tại mốc T: tổng lượng mưa của giờ TRƯỚC đó (T-1h -> T).
  - HealthyAir dòng T là trung bình T:00 -> T:59. Nên tại gốc dự báo t, dùng khí tượng có mốc <= t
    là hợp lệ (không nhìn trước tương lai).

Đặc trưng tạo thêm (tính trong TỪNG trạm, CHỈ dùng quá khứ và hiện tại; kiểm chứng ở tests/test_meteo_features.py):
  wind_u, wind_v        (G1-36) thành phần gió (m/s), quy ước khí tượng: hướng là nơi gió THỔI TỚI TỪ
                        u = -ws*sin(dir), v = -ws*cos(dir). Gió Bắc 5 m/s -> (0, -5); gió Tây 5 m/s -> (+5, 0)
  wind_dir_sin/cos      (G1-36) hướng gió mã hoá vòng tròn
  precip_6h, precip_24h (G1-37) tổng mưa 6 / 24 giờ qua (gồm giờ hiện tại)
  hours_since_rain      (G1-37) số giờ kể từ giờ gần nhất có mưa >= rain_threshold_mm (chốt 0,1 mm/h), chặn trên
  wind_speed_6h         (G1-38) tốc độ gió trung bình 6 giờ qua (gồm giờ hiện tại)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.openmeteo_common import ROOT, load_om_config, read_raw, rpath

KEY = ["city", "station_id", "time"]
DERIVED = ["wind_u", "wind_v", "wind_dir_sin", "wind_dir_cos", "precip_6h", "precip_24h",
           "hours_since_rain", "wind_speed_6h"]


def wind_components(speed, direction_deg):
    """(u, v) theo quy ước khí tượng; direction_deg là hướng gió THỔI TỚI TỪ (0 = Bắc, 90 = Đông)."""
    th = np.deg2rad(np.asarray(direction_deg, dtype=float))
    sp = np.asarray(speed, dtype=float)
    return -sp * np.sin(th), -sp * np.cos(th)


def hours_since_event(event: np.ndarray, cap: int) -> np.ndarray:
    """Số giờ kể từ lần cuối event=True (0 nếu giờ hiện tại có event). Trước event đầu tiên -> cap.
    Chỉ nhìn về quá khứ: giá trị tại i chỉ phụ thuộc event[0..i]."""
    out = np.empty(len(event), dtype=float)
    last = None
    for i, e in enumerate(event):
        if e:
            last = i
        out[i] = cap if last is None else min(i - last, cap)
    return out


def add_features(g: pd.DataFrame, fcfg: dict) -> pd.DataFrame:
    """Đặc trưng cho MỘT trạm trên lưới giờ đầy đủ (tên cột theo API)."""
    g = g.sort_values("time").copy()
    assert (g["time"].diff().dropna() == pd.Timedelta(hours=1)).all(), "cần lưới giờ đầy đủ trước khi rolling"
    ws, wd, pr = g["wind_speed_10m"], g["wind_direction_10m"], g["precipitation"]
    g["wind_u"], g["wind_v"] = wind_components(ws, wd)
    rad = np.deg2rad(wd)
    g["wind_dir_sin"], g["wind_dir_cos"] = np.sin(rad), np.cos(rad)
    # rolling mặc định của pandas là cửa sổ "lùi" (giờ t-w+1 .. t) -> không nhìn trước
    for w in fcfg["precip_windows"]:
        g[f"precip_{w}h"] = pr.rolling(w, min_periods=w).sum()
    w = fcfg["wind_mean_window"]
    g[f"wind_speed_{w}h"] = ws.rolling(w, min_periods=w).mean()
    g["hours_since_rain"] = hours_since_event((pr >= fcfg["rain_threshold_mm"]).to_numpy(),
                                              fcfg["hours_since_rain_cap"])
    return g


def build_city(cfg: dict, city: str) -> pd.DataFrame:
    frames = []
    for sid in cfg["cities"][city]["stations"]:
        d = read_raw(cfg, city, sid)
        d["city"], d["station_id"] = city, sid
        frames.append(add_features(d, cfg["features"]))
    df = pd.concat(frames, ignore_index=True)
    df = df[KEY + cfg["hourly_vars"] + DERIVED].sort_values(KEY).reset_index(drop=True)
    assert not df.duplicated(KEY).any()
    return df


def main() -> None:
    cfg = load_om_config()
    for city, key in [("hcm", "out_hcm"), ("beijing", "out_beijing")]:
        df = build_city(cfg, city)
        out = rpath(cfg, key)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"{city}: {len(df)} dòng × {df.shape[1]} cột, {df['station_id'].nunique()} trạm, "
              f"{df['time'].min()} -> {df['time'].max()} -> {out.relative_to(ROOT)}")
        print(df[DERIVED].describe().T.round(2).to_string(), "\n")


if __name__ == "__main__":
    main()
