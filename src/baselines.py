"""G1-60, G1-61, G1-62 — Hai baseline bắt buộc, chạy trên test chính và 4 fold rolling-origin.

    python -m src.baselines                  # TP.HCM: test + fold1..4, ghi results/baselines.csv + results/preds/
    python -m src.baselines --city beijing   # (tham khảo) chạy trên val Bắc Kinh, ghi results/beijing_reference.csv

1) Persistence:          ŷ(t+h) = y(t)        — "ngày mai giống hôm nay"; mọi mô hình PHẢI thắng nó.
2) Trung bình theo giờ:  ŷ(t+h) = mean_train(y | trạm s, giờ-trong-ngày của t+h)
                          — chỉ nắm chu kỳ ngày, không dùng thông tin gần đây.
Cả hai tính trực tiếp trên µg/m³. Tập cửa sổ GIỐNG HỆT các mô hình khác (L lớn nhất = 72, cùng
quy tắc lọc) để so sánh công bằng; dự đoán chỉ đánh giá trên giờ ĐO THẬT.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.data.pipeline import prepare
from src.data.splits import in_range
from src.eval.metrics import wide_to_long
from src.eval.results import append_results, make_run_id, results_rows, save_predictions
from src.utils import ROOT, load_config, path_of, results_file

L_REF = 72          # dùng L lớn nhất để tập gốc t giống nhau với mọi L
FEATS = ["pm25_z"]  # baseline chỉ cần PM2.5; cửa sổ lọc giống ablation (a)


def persistence(ws) -> np.ndarray:
    y0 = ws.last_observed_ug()
    return np.repeat(y0[:, None], ws.H, axis=1)


def hourly_mean_table(df: pd.DataFrame, train_rng) -> pd.Series:
    """Trung bình PM2.5 (µg/m³, chỉ giá trị đo thật) theo (trạm, giờ) trên TRAIN."""
    tr = df[in_range(df["time"], train_rng) & (df["is_imputed"] == 0)]
    return tr.groupby(["station_id", tr["time"].dt.hour])["pm25"].mean()


def hourly_mean(ws, table: pd.Series) -> np.ndarray:
    meta = ws.meta()
    out = np.empty((len(ws), ws.H))
    for h in range(1, ws.H + 1):
        hrs = (meta["time_origin"] + pd.Timedelta(hours=h)).dt.hour
        idx = pd.MultiIndex.from_arrays([meta["station_id"], hrs])
        out[:, h - 1] = table.reindex(idx).to_numpy()
    return out


def run(city: str = "hcm", horizons=(1, 6, 24), write: bool = True):
    cfg = load_config()
    eval_sets = [("test", None)] + [(f"fold{f['fold']}", f["fold"]) for f in cfg["rolling_origin"].get(city, [])] \
        if city == "hcm" else [("val", None)]
    all_rows = []
    for eval_set, fold in eval_sets:
        P = prepare(city, fold=fold, cfg=cfg, save_scaler=(fold is None and write))
        split = "val" if eval_set == "val" else "test"
        ws = P.windows(split, FEATS, L=L_REF)   # tập đánh giá chung (common_eval)
        y_true = ws.y_true_ug()
        table = hourly_mean_table(P.df, P.rng["train"])
        for name, y_pred in [("persistence", persistence(ws)), ("hourly_mean", hourly_mean(ws, table))]:
            run_id = make_run_id(model=name, city=city, data=cfg["data_version"])
            preds = wide_to_long(ws.meta(), y_true, y_pred, horizons, model=name, config=run_id)
            rows = results_rows(preds, member="Bo bo", model=name, strategy="S0" if city == "hcm" else "-",
                                ablation="-", train_frac=1.0, L=0, seed=0, eval_set=eval_set, run_id=run_id,
                                notes=f"data={cfg['data_version']}; windows L_ref={L_REF}")
            all_rows.append(rows)
            if write and city == "hcm":
                save_predictions(preds, path_of(cfg, "preds_dir"), run_id, eval_set)
    res = pd.concat(all_rows, ignore_index=True)
    if write:
        if city == "hcm":
            append_results(res, results_file(cfg, "baselines"))
        else:
            res.to_csv(ROOT / "results" / "beijing_reference.csv", index=False, float_format="%.4f")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="hcm", choices=["hcm", "beijing"])
    a = ap.parse_args()
    r = run(a.city)
    show = r[r["station"] == "ALL"][["eval_set", "model", "horizon", "n", "MAE", "RMSE", "R2", "MAE_high", "AQI_acc"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
