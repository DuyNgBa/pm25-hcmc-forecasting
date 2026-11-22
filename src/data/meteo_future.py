"""G1-40 — Khí tượng TƯƠNG LAI (giá trị thực tại t+h) cho thiết lập phụ "dự báo thời tiết hoàn hảo".

Dùng chung cho cả nhóm (G2-48 của TV2 và các thiết lập tương ứng của TV1, TV3). Đây là GIỚI HẠN TRÊN:
giả sử có dự báo thời tiết đúng tuyệt đối, để đo thông tin thời tiết tương lai đáng giá đến đâu.
KHÔNG được dùng trong thiết lập chính (mô hình chính chỉ thấy khí tượng đến thời điểm t).

    from src.data.meteo_future import add_future_meteo, future_block

    # (1) Mô hình cây (một mô hình cho mỗi tầm h): thêm cột <biến>_fut{h} = giá trị tại t+h
    df = add_future_meteo(df, cols=["temp", "wind_speed", "precip"], horizons=[1, 6, 24])

    # (2) Mô hình DL dự báo cả chuỗi 1..H: khối (n_gốc, H, n_biến) khí tượng tại t+1..t+H
    X_fut, ok = future_block(df, origins, cols=["temp", "wind_u", "wind_v", "precip"], H=24)

    python -m src.data.meteo_future    # xuất data/interim/meteo_hcm_future.parquet (h = 1, 6, 24)

Mọi phép dịch thực hiện trong TỪNG trạm, trên lưới giờ đầy đủ (dịch k dòng = dịch k giờ).
Open-Meteo không thiếu giờ nào nên không phát sinh NaN, trừ khi t+h vượt quá cuối chuỗi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.openmeteo_common import ROOT, load_om_config, rpath

KEY = ["city", "station_id", "time"]
DEFAULT_COLS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "surface_pressure",
                "wind_speed_10m", "wind_u", "wind_v"]


def _check_grid(df: pd.DataFrame, group=("city", "station_id")) -> None:
    for k, g in df.groupby(list(group)):
        d = g["time"].sort_values().diff().dropna()
        if not (d == pd.Timedelta(hours=1)).all():
            raise ValueError(f"Trạm {k} không có lưới giờ đầy đủ -> dịch dòng sẽ KHÔNG phải dịch giờ")


def add_future_meteo(df: pd.DataFrame, cols=DEFAULT_COLS, horizons=(1, 6, 24),
                     group=("city", "station_id")) -> pd.DataFrame:
    """Thêm cột f"{c}_fut{h}" = giá trị của c tại thời điểm t+h (cùng trạm)."""
    _check_grid(df, group)
    out = df.sort_values(list(group) + ["time"]).copy()
    g = out.groupby(list(group), sort=False)
    for h in horizons:
        shifted = g[list(cols)].shift(-h)
        for c in cols:
            out[f"{c}_fut{h}"] = shifted[c]
    return out


def future_block(df: pd.DataFrame, origins: pd.DataFrame, cols=DEFAULT_COLS, H: int = 24,
                 group=("city", "station_id")) -> tuple[np.ndarray, np.ndarray]:
    """Khối khí tượng tương lai cho mô hình dự báo nhiều bước.

    origins: DataFrame có các cột group + "time" (= gốc t). Trả về
      X  : mảng (len(origins), H, len(cols)), X[i, k-1, :] = khí tượng tại origins.time[i] + k giờ
      ok : mảng bool (len(origins),), False nếu có bước t+k nằm ngoài dữ liệu
    """
    keys = list(group)
    tab = df.set_index(keys + ["time"])[list(cols)].sort_index()
    X = np.full((len(origins), H, len(cols)), np.nan, dtype=np.float32)
    for k in range(1, H + 1):
        idx = pd.MultiIndex.from_frame(origins[keys].assign(time=origins["time"] + pd.Timedelta(hours=k)))
        X[:, k - 1, :] = tab.reindex(idx).to_numpy(dtype=np.float32)
    ok = ~np.isnan(X).any(axis=(1, 2))
    return X, ok


def main() -> None:
    cfg = load_om_config()
    m = pd.read_parquet(rpath(cfg, "out_hcm"))
    fut = add_future_meteo(m, DEFAULT_COLS, horizons=[1, 6, 24])
    keep = KEY + [c for c in fut.columns if "_fut" in c]
    out = rpath(cfg, "out_hcm").with_name("meteo_hcm_future.parquet")
    fut[keep].to_parquet(out, index=False)
    print(f"{out.relative_to(ROOT)}: {len(fut)} dòng × {len(keep)} cột")


if __name__ == "__main__":
    main()
