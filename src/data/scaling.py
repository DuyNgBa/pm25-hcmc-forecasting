"""G1-51, G1-52, G1-53 — Chuẩn hoá PM2.5 và khí tượng; hàm đảo ngược.

PM2.5:   z = (log(1 + x) - mu_s) / sigma_s      (mu_s, sigma_s tính trên TRAIN của trạm s)
Đảo ngược: x = exp(z * sigma_s + mu_s) - 1

Vì sao log rồi z-score theo trạm?
  - log nén đuôi phải (PM2.5 lệch phải mạnh) và biến "tăng gấp đôi" thành "cộng một lượng" ->
    mô hình học tỉ lệ biến động, giống nhau hơn giữa Bắc Kinh (TB ~80) và TP.HCM (TB ~21).
  - z-score theo trạm loại bỏ mức nền riêng của trạm -> mô hình học HÌNH DẠNG biến động.
  - Chỉ fit trên train để không rò rỉ thông tin của val/test (data leakage).

Ví dụ:
    params = fit_scalers(df, train_mask, cfg)        # dict, lưu được ra JSON
    df_s   = transform(df, params)                   # thêm cột pm25_z và <meteo>_s
    y_ug   = inverse_pm25(z_pred, station_ids, params, city="hcm")
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-6


def fit_scalers(df: pd.DataFrame, train_mask: np.ndarray, cfg: dict) -> dict:
    """Tính tham số chuẩn hoá CHỈ trên các dòng train_mask=True.

    df có thể chứa nhiều thành phố; tham số PM2.5 theo (city, station), khí tượng theo city.
    Với PM2.5 chỉ dùng giá trị ĐO THẬT (is_imputed = 0) để tính mu/sigma.
    """
    sc = cfg["scaling"]["meteo"]
    tr = df.loc[np.asarray(train_mask)]
    params = {"pm25": {}, "meteo": {}, "fit_rows": int(len(tr))}
    real = tr[tr["is_imputed"] == 0] if "is_imputed" in tr else tr
    for (city, st), g in real.groupby(["city", "station_id"]):
        v = np.log1p(g["pm25"].dropna().to_numpy())
        if len(v) < 24:
            raise ValueError(f"Quá ít dữ liệu train để fit scaler cho {city}/{st}: {len(v)} giờ")
        params["pm25"].setdefault(city, {})[st] = {"mu": float(v.mean()), "sigma": float(v.std() + EPS)}
    for city, g in tr.groupby("city"):
        p = {}
        for c in sc["zscore"]:
            v = g[c].dropna().to_numpy()
            p[c] = {"mu": float(v.mean()), "sigma": float(v.std() + EPS), "transform": "zscore"} if len(v) else None
        for c in sc["scale_only"]:
            v = g[c].dropna().to_numpy()
            p[c] = {"mu": 0.0, "sigma": float(v.std() + EPS), "transform": "scale_only"} if len(v) else None
        for c in sc["log1p_scale_only"]:
            v = np.log1p(g[c].dropna().to_numpy())
            p[c] = {"mu": 0.0, "sigma": float(v.std() + EPS), "transform": "log1p_scale_only"} if len(v) else None
        params["meteo"][city] = p
    return params


def transform(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Thêm cột pm25_z và <biến>_s. Giữ nguyên cột gốc (cần µg/m³ để đánh giá)."""
    out = df.copy()
    mu = _map_station(out, params, "mu")
    sg = _map_station(out, params, "sigma")
    out["pm25_z"] = (np.log1p(out["pm25"]) - mu) / sg
    for city, p in params["meteo"].items():
        m = out["city"] == city
        for c, q in p.items():
            col = f"{c}_s"
            if col not in out:
                out[col] = np.nan
            if q is None:
                continue
            x = out.loc[m, c].astype(float)
            if q["transform"] == "log1p_scale_only":
                x = np.log1p(x)
            out.loc[m, col] = (x - q["mu"]) / q["sigma"]
    return out


def _map_station(df, params, key):
    lut = {(c, s): v[key] for c, d in params["pm25"].items() for s, v in d.items()}
    idx = pd.MultiIndex.from_frame(df[["city", "station_id"]])
    vals = pd.Series(lut).reindex(idx).to_numpy()
    if np.isnan(vals).any():
        missing = set(map(tuple, df.loc[np.isnan(vals), ["city", "station_id"]].drop_duplicates().to_numpy()))
        raise KeyError(f"Không có tham số scaler cho trạm: {missing}")
    return vals


def inverse_pm25(z, station_ids, params: dict, city: str | list) -> np.ndarray:
    """G1-52: z (dự đoán của mô hình) -> µg/m³. Hỗ trợ z dạng (N,) hoặc (N, H).

    station_ids: mảng (N,) mã trạm tương ứng từng dòng. city: 1 tên hoặc mảng (N,).
    """
    z = np.asarray(z, dtype=float)
    st = np.asarray(station_ids)
    cities = np.full(len(st), city) if isinstance(city, str) else np.asarray(city)
    mu = np.array([params["pm25"][c][s]["mu"] for c, s in zip(cities, st)])
    sg = np.array([params["pm25"][c][s]["sigma"] for c, s in zip(cities, st)])
    if z.ndim == 2:
        mu, sg = mu[:, None], sg[:, None]
    return np.expm1(z * sg + mu)


def save_params(params: dict, path) -> None:
    Path(path).write_text(json.dumps(params, indent=2, ensure_ascii=False))


def load_params(path) -> dict:
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# Chạy trực tiếp:  python -m src.data.scaling
# Fit scaler trên TRAIN của split chính cho cả 2 thành phố -> data/processed/scaler_params.json,
# rồi kiểm tra đổi đi đổi lại (x -> z -> x) có ra đúng giá trị gốc không (G1-52).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data.splits import get_split_ranges, in_range
    from src.utils import load_config, path_of

    cfg = load_config()
    all_params = {}
    for city in ["hcm", "beijing"]:
        df = pd.read_parquet(path_of(cfg, f"{city}_schema"))
        rng = get_split_ranges(cfg, city)
        params = fit_scalers(df, in_range(df["time"], rng["train"]), cfg)
        params["fit_info"] = {"city": city, "train_range": [str(x) for x in rng["train"]]}
        all_params[city] = params

        df_s = transform(df, params)
        ok = df_s["pm25"].notna()
        back = inverse_pm25(df_s.loc[ok, "pm25_z"], df_s.loc[ok, "station_id"], params, city)
        err = float(np.abs(back - df_s.loc[ok, "pm25"].to_numpy()).max())

        print(f"\n=== {city}: fit trên train {rng['train'][0].date()} -> {rng['train'][1].date()} ===")
        tbl = pd.DataFrame(params["pm25"][city]).T.round(3)
        tbl["exp(mu)-1 (µg/m³)"] = np.expm1(tbl["mu"]).round(1)   # mức "điển hình" của trạm
        print(tbl.to_string())
        z_tr = df_s.loc[in_range(df_s["time"], rng["train"]), "pm25_z"]
        print(f"pm25_z trên train: TB = {z_tr.mean():.3f}, độ lệch chuẩn = {z_tr.std():.3f}  (kỳ vọng ≈ 0 và ≈ 1)")
        print(f"Kiểm tra đổi ngược z -> µg/m³: sai số lớn nhất = {err:.2e}  ->", "OK" if err < 1e-6 else "LỖI")

    out = path_of(cfg, "scaler_params")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_params(all_params, out)
    print(f"\nĐã lưu {out}")
