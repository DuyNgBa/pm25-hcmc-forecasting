"""G0-04, G0-05 — Ghi kết quả (results/<tên-phương-pháp>.csv, cùng mẫu cột) và lưu file dự đoán theo format chung.

Mỗi file kết quả — một dòng = một (mô hình, cấu hình, tầm, trạm/ALL, tập đánh giá, seed):
    member, model, strategy, ablation, train_frac, horizon, station, L, seed,
    MAE, RMSE, R2, MAE_high, AQI_acc, AQI_macroF1, notes,
    eval_set, n, run_id          <- 3 cột BỔ SUNG (đề xuất của TV3, cần cả nhóm OK):
       eval_set: "test" | "val" | "fold1".."fold4"  (không có cột này thì không phân biệt được
                 kết quả rolling-origin với test chính)
       n       : số giờ đo thật dùng để tính độ đo (kiểm tra nhanh số bất thường)
       run_id  : khoá nối sang file dự đoán results/preds/<run_id>.parquet
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.eval.metrics import evaluate_grouped

RESULT_COLUMNS = ["member", "model", "strategy", "ablation", "train_frac", "horizon", "station", "L", "seed",
                  "MAE", "RMSE", "R2", "MAE_high", "AQI_acc", "AQI_macroF1", "notes",
                  "eval_set", "n", "run_id"]
PRED_COLUMNS = ["city", "station_id", "time_origin", "horizon", "y_true", "y_pred", "model", "config"]

ALLOWED = {
    "member": {"Duy Ngô", "Minh Trí", "Bo bo"},
    "strategy": {"S0", "S1", "S2", "S2_full", "S2_head", "S3", "-"},
    "ablation": {"a", "b", "c", "d", "c_oracle", "-"},   # c_oracle: khí tượng thực tại t+h (thiết lập phụ)
}


def make_run_id(**cfg) -> str:
    """run_id ngắn, tất định từ cấu hình (cùng cấu hình -> cùng id)."""
    s = json.dumps(cfg, sort_keys=True, default=str)
    return f"{cfg.get('model', 'm')}_{hashlib.md5(s.encode()).hexdigest()[:8]}"


def results_rows(preds: pd.DataFrame, *, member: str, model: str, strategy: str, ablation: str,
                 train_frac: float, L: int, seed: int, eval_set: str, run_id: str, notes: str = "",
                 per_station: bool = True) -> pd.DataFrame:
    """Tính độ đo từ bảng dự đoán -> các dòng results.csv (ALL + từng trạm, theo từng tầm)."""
    for k, allowed in ALLOWED.items():
        v = {"member": member, "strategy": strategy, "ablation": ablation}[k]
        if v not in allowed:
            raise ValueError(f"{k}={v!r} không hợp lệ; cho phép: {sorted(allowed)}")
    by_h = evaluate_grouped(preds, ["horizon"]).assign(station="ALL")
    parts = [by_h]
    if per_station:
        parts.append(evaluate_grouped(preds, ["horizon", "station_id"]).rename(columns={"station_id": "station"}))
    res = pd.concat(parts, ignore_index=True)
    res = res.assign(member=member, model=model, strategy=strategy, ablation=ablation, train_frac=train_frac,
                     L=L, seed=seed, notes=notes, eval_set=eval_set, run_id=run_id)
    return res[RESULT_COLUMNS]


def append_results(rows: pd.DataFrame, path) -> None:
    """Ghi thêm vào results.csv. Nếu đã có dòng cùng (run_id, eval_set, horizon, station) -> thay thế."""
    path = Path(path)
    rows = rows[RESULT_COLUMNS]
    if path.exists() and path.stat().st_size > 0:
        old = pd.read_csv(path)
        key = ["run_id", "eval_set", "horizon", "station"]
        old = old[~old.set_index(key).index.isin(rows.set_index(key).index)]
        rows = pd.concat([old, rows], ignore_index=True)
    rows.to_csv(path, index=False, float_format="%.4f")


def save_predictions(preds: pd.DataFrame, preds_dir, run_id: str, eval_set: str) -> Path:
    """Lưu dự đoán theo format chung: results/preds/<run_id>__<eval_set>.parquet."""
    missing = [c for c in PRED_COLUMNS if c not in preds.columns]
    if missing:
        raise KeyError(f"File dự đoán thiếu cột {missing}")
    p = Path(preds_dir) / f"{run_id}__{eval_set}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    preds[PRED_COLUMNS].to_parquet(p, index=False)
    return p


def write_template(path) -> None:
    """Tạo file mẫu results_template.csv (chỉ header + 1 dòng ví dụ để mọi người nhìn format)."""
    ex = {c: "" for c in RESULT_COLUMNS}
    ex.update(member="Bo bo", model="persistence", strategy="S0", ablation="-", train_frac=1.0, horizon=1,
              station="ALL", L=0, seed=0, notes="ví dụ – xoá dòng này", eval_set="test", n=0,
              run_id="persistence_xxxxxxxx")
    pd.DataFrame([ex], columns=RESULT_COLUMNS).to_csv(path, index=False)


if __name__ == "__main__":
    from src.utils import ROOT

    write_template(ROOT / "results" / "results_template.csv")
    print("Đã tạo results/results_template.csv;", dt.date.today())
