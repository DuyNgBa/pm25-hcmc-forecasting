"""G1-57 — Hàm đánh giá CHUNG cho cả nhóm (mọi mô hình phải dùng hàm này).

Đầu vào là bảng dự đoán theo format chung (G0-05, xem docs/PIPELINE_GUIDE.md §6):
    city, station_id, time_origin, horizon, y_true, y_pred, model, config
  - y_true, y_pred: µg/m³ (đã đảo ngược chuẩn hoá).
  - y_true = NaN nếu giờ đó KHÔNG phải giá trị đo thật (thiếu hoặc is_imputed = 1)
    -> tự động bị loại khỏi mọi độ đo ("chỉ tính trên giờ đo thật", G1-16).

Độ đo:
  MAE  = mean |y - ŷ|                          (µg/m³, dễ hiểu nhất)
  RMSE = sqrt(mean (y - ŷ)²)                   (phạt nặng sai số lớn — các đỉnh ô nhiễm)
  R²   = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²             (tỉ lệ phương sai giải thích được; < 0 nghĩa là
                                                 tệ hơn đoán bằng trung bình)
  MAE_high = MAE chỉ trên giờ y > 35,4 µg/m³   (giờ cần cảnh báo)
  AQI_acc, AQI_macroF1: trên mức AQI quy đổi (src/eval/aqi.py)

Ví dụ:
    from src.eval.metrics import evaluate, evaluate_grouped
    evaluate(preds)                                    # dict tổng thể
    evaluate_grouped(preds, by=["horizon"])            # theo tầm
    evaluate_grouped(preds, by=["horizon", "station_id"])
    evaluate_grouped(add_area_type(preds, cfg), by=["horizon", "area_type"])
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.aqi import aqi_classification_scores

HIGH_THRESHOLD = 35.4


def regression_scores(y_true, y_pred, high_threshold: float = HIGH_THRESHOLD) -> dict:
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = ~np.isnan(yt) & ~np.isnan(yp)
    yt, yp = yt[ok], yp[ok]
    n = len(yt)
    if n == 0:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAE_high": np.nan, "n_high": 0}
    err = yp - yt
    sst = ((yt - yt.mean()) ** 2).sum()
    hi = yt > high_threshold
    return {
        "n": int(n),
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "R2": float(1 - (err ** 2).sum() / sst) if sst > 0 else np.nan,
        "MAE_high": float(np.abs(err[hi]).mean()) if hi.any() else np.nan,
        "n_high": int(hi.sum()),
    }


def evaluate(preds: pd.DataFrame, high_threshold: float = HIGH_THRESHOLD, aqi_scale: str = "us_epa_2024") -> dict:
    r = regression_scores(preds["y_true"], preds["y_pred"], high_threshold)
    r.update(aqi_classification_scores(preds["y_true"], preds["y_pred"], aqi_scale))
    return r


def evaluate_grouped(preds: pd.DataFrame, by: list[str], **kw) -> pd.DataFrame:
    rows = []
    for keys, g in preds.groupby(by, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({**dict(zip(by, keys)), **evaluate(g, **kw)})
    return pd.DataFrame(rows)


def add_area_type(preds: pd.DataFrame, cfg: dict, city: str = "hcm") -> pd.DataFrame:
    lut = {s: v["area_type"] for s, v in cfg["stations"][city].items()}
    out = preds.copy()
    out["area_type"] = out["station_id"].map(lut)
    return out


def wide_to_long(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                 horizons=(1, 6, 24), model: str = "", config: str = "") -> pd.DataFrame:
    """Chuyển ma trận (N, H) -> bảng dự đoán dài theo format chung, chỉ giữ các tầm cần báo cáo.

    horizons đánh số từ 1 (cột 0 của ma trận là t+1).
    """
    parts = []
    for h in horizons:
        d = meta[["city", "station_id", "time_origin"]].copy()
        d["horizon"] = h
        d["y_true"] = y_true[:, h - 1]
        d["y_pred"] = y_pred[:, h - 1]
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out["model"], out["config"] = model, config
    return out
