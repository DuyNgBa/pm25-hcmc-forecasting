"""G2-49 (+ điều kiện trước G2-46/47) — Kiểm tra kết quả LSTM/GRU/Seq2Seq TRƯỚC khi tổng hợp, không train lại.

    python -m src.models.rnn_check                      # kiểm tra bộ dữ liệu khớp dữ liệu local (hoặc bộ duy nhất)
    python -m src.models.rnn_check --data-fp 638755cc9b7d
    python -m src.models.rnn_check --require-gpu        # mọi lần chạy phải chạy trên cuda (kết quả Kaggle)
    python -m src.models.rnn_check --groups rq1 seq2seq # chỉ yêu cầu đủ một số nhóm thí nghiệm
    python -m src.models.rnn_check --truth-anyway       # vẫn đối chiếu y_true với hcm_schema local khi data_fp khác

Đầu vào: results/rnn_runs.csv, results/rnn.csv, results/preds/, results/baselines.csv (+ dự đoán persistence),
configs/rnn.yaml. Đầu ra: reports/rnn/g2_49_check.md và g2_49_check.json; mã thoát 1 nếu có LỖI.

Kiểm tra:
 1. Đủ cấu hình: mọi (mô hình, ablation, chiến lược, % train, tập đánh giá, seed) mà rnn_experiments sinh ra cho
    các nhóm rq1, rolling, rq2, seq2seq, rq3, curve, oracle đều có ĐÚNG MỘT lần chạy, cùng một data_fp.
 2. Siêu tham số mỗi lần chạy = best của cell trong configs/rnn.yaml, và best được tối ưu trên cùng data_fp.
 3. File dự đoán (format chung): đủ cột, city = hcm, đúng tầm {1, 6, 24}, không trùng khoá, y_pred hữu hạn,
    model/config khớp run_id, thời điểm mục tiêu t+h nằm trong tập đánh giá.
 4. Đối chiếu với persistence của nhóm trên CÙNG tập đánh giá: cùng tập khoá (trạm, gốc t, tầm) và CÙNG y_true
    từng dòng (không chỉ cùng số dòng). Số giờ đo thật n mỗi tầm = n của persistence trong results/baselines.csv.
 5. Nếu dữ liệu local có cùng data_fp: y_true = PM2.5 ĐO THẬT (is_imputed = 0) tại t+h trong hcm_schema.
 6. Tính lại MAE/RMSE từ file dự đoán, phải khớp results/rnn.csv và rnn_runs.csv.
 7. Mọi lần chạy cùng raw_fp/hcm_fp/beijing_fp; so với bản khoá data/DATA_LOCK.json của nhóm (chỉ ghi nhận: khớp /
    không khớp / chưa có bản khoá). rnn_report dùng kết quả này để quyết định nhãn "CHÍNH THỨC".
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np
import pandas as pd

from src.data.splits import get_split_ranges
from src.eval.results import PRED_COLUMNS
from src.models.provenance import LEGACY_FP, compare_lock, load_lock, runs_with_fp, select_fp, try_data_provenance
from src.models.rnn_experiments import GROUPS, HORIZONS, RUNS_LOG, load_rnn_config, specs_for
from src.utils import ROOT, load_config, path_of, results_file

OUT = ROOT / "reports" / "rnn"
ATTRS = ["model", "arch", "cell", "ablation", "strategy", "train_frac", "eval_set", "seed"]
KEYS = ["station_id", "time_origin", "horizon"]


class Log:
    def __init__(self):
        self.errors, self.warnings, self.info = [], [], []

    def err(self, m):
        self.errors.append(m)

    def warn(self, m):
        self.warnings.append(m)


def expected_table(rcfg: dict, groups, cells=("lstm", "gru")) -> pd.DataFrame:
    rows = []
    for g in groups:
        for sp in specs_for(g, rcfg, cells):
            rows.append({"group": g, "model": sp.model_name, "arch": sp.arch, "cell": sp.cell, "ablation": sp.ablation,
                         "strategy": sp.strategy, "train_frac": float(sp.train_frac), "eval_set": sp.eval_set,
                         "seed": int(sp.seed), "L": sp.L, "hp": json.dumps(sp.hp, sort_keys=True)})
    return pd.DataFrame(rows)


def check_coverage(runs: pd.DataFrame, exp: pd.DataFrame, log: Log) -> pd.DataFrame:
    r = runs.copy()
    r["train_frac"] = r["train_frac"].astype(float)
    r["seed"] = r["seed"].astype(int)
    cnt = r.groupby(ATTRS).size().rename("n_runs").reset_index()
    m = exp.merge(cnt, on=ATTRS, how="left").fillna({"n_runs": 0})
    for _, x in m[m["n_runs"] == 0].iterrows():
        log.err(f"Thiếu lần chạy [{x.group}] {x.model} ab={x.ablation} {x.strategy} frac={x.train_frac} "
                f"{x.eval_set} seed={x.seed}")
    for _, x in m[m["n_runs"] > 1].iterrows():
        log.err(f"Trùng {int(x.n_runs)} lần chạy cho {x.model} ab={x.ablation} {x.strategy} {x.eval_set} seed={x.seed}")
    extra = r.merge(exp[ATTRS].drop_duplicates(), on=ATTRS, how="left", indicator=True)
    extra = extra[extra["_merge"] == "left_only"]
    if len(extra):
        log.warn(f"{len(extra)} lần chạy ngoài danh sách nhóm được kiểm tra (vd smoke hoặc nhóm khác): bỏ qua")
    # siêu tham số
    hp_of = {c: json.dumps(hp, sort_keys=True) for c, hp in exp.groupby("cell")["hp"].first().map(json.loads).items()}
    for _, x in r.iterrows():
        if x["cell"] in hp_of and "hp" in r and json.dumps(json.loads(x["hp"]), sort_keys=True) != hp_of[x["cell"]]:
            log.err(f"{x['run_id']}: siêu tham số khác best.{x['cell']} trong configs/rnn.yaml")
    return m


def eval_range(cfg: dict, eval_set: str):
    fold = None if eval_set == "test" else int(eval_set.replace("fold", ""))
    return get_split_ranges(cfg, "hcm", fold=fold)["test"]


def compare_with_reference(pred: pd.DataFrame, ref: pd.DataFrame, label: str, log: Log, atol: float = 1e-6) -> None:
    """pred và ref (persistence) phải có CÙNG tập khoá (trạm, gốc t, tầm) và CÙNG y_true từng dòng."""
    a = pred.set_index(KEYS)["y_true"]
    b = ref.set_index(KEYS)["y_true"]
    only_a, only_b = a.index.difference(b.index), b.index.difference(a.index)
    if len(only_a) or len(only_b):
        log.err(f"{label}: tập (trạm, gốc t, tầm) khác persistence: thừa {len(only_a)}, thiếu {len(only_b)}"
                + (f" (vd {only_a[0]})" if len(only_a) else f" (vd {only_b[0]})"))
    common = a.index.intersection(b.index)
    x, y = a.loc[common].to_numpy(float), b.loc[common].to_numpy(float)
    bad = ~((np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= atol))
    if bad.any():
        k = common[np.argmax(bad)]
        log.err(f"{label}: {int(bad.sum())} dòng y_true khác persistence (vd {k}: {a.loc[k]} vs {b.loc[k]})")


def check_pred_file(pred: pd.DataFrame, run: pd.Series, cfg: dict, log: Log) -> None:
    rid, es = run["run_id"], run["eval_set"]
    lab = f"{rid}__{es}"
    miss = [c for c in PRED_COLUMNS if c not in pred.columns]
    if miss:
        log.err(f"{lab}: thiếu cột {miss}")
        return
    if set(pred["city"].unique()) != {"hcm"}:
        log.err(f"{lab}: city = {sorted(pred['city'].unique())}, phải chỉ có hcm")
    if set(pred["horizon"].unique()) != set(HORIZONS):
        log.err(f"{lab}: tầm = {sorted(pred['horizon'].unique())}, phải là {list(HORIZONS)}")
    if pred.duplicated(KEYS).any():
        log.err(f"{lab}: {int(pred.duplicated(KEYS).sum())} dòng trùng khoá (trạm, gốc t, tầm)")
    if not np.isfinite(pred["y_pred"].to_numpy(float)).all():
        log.err(f"{lab}: y_pred có NaN/inf")
    if (pred["config"] != rid).any() or (pred["model"] != run["model"]).any():
        log.err(f"{lab}: cột model/config không khớp run_id")
    stations = set(cfg["stations"]["hcm"])
    if not set(pred["station_id"]).issubset(stations):
        log.err(f"{lab}: trạm lạ {sorted(set(pred['station_id']) - stations)}")
    lo, hi = eval_range(cfg, es)
    tt = pd.to_datetime(pred["time_origin"]) + pd.to_timedelta(pred["horizon"], unit="h")
    if (tt < lo).any() or (tt > hi).any():
        log.err(f"{lab}: có thời điểm mục tiêu t+h ngoài {es} [{lo} .. {hi}]")


def check_truth(pred: pd.DataFrame, truth: pd.Series, label: str, log: Log) -> None:
    tt = pd.to_datetime(pred["time_origin"]) + pd.to_timedelta(pred["horizon"], unit="h")
    idx = pd.MultiIndex.from_arrays([pred["station_id"], tt])
    t = truth.reindex(idx).to_numpy(float)
    y = pred["y_true"].to_numpy(float)
    bad = ~((np.isnan(t) & np.isnan(y)) | (np.abs(t - y) <= 1e-6))
    if bad.any():
        log.err(f"{label}: {int(bad.sum())} dòng y_true khác PM2.5 đo thật tại t+h trong hcm_schema")


def scores(pred: pd.DataFrame) -> pd.DataFrame:
    ok = pred["y_true"].notna()
    e = (pred.loc[ok, "y_pred"] - pred.loc[ok, "y_true"])
    g = e.groupby(pred.loc[ok, "horizon"])
    return pd.DataFrame({"n": g.size(), "MAE": g.apply(lambda v: v.abs().mean()),
                         "RMSE": g.apply(lambda v: float(np.sqrt((v ** 2).mean())))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-fp", default=None)
    ap.add_argument("--groups", nargs="+", default=GROUPS, choices=GROUPS)
    ap.add_argument("--cells", nargs="+", default=["lstm", "gru"])
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--truth-anyway", action="store_true",
                    help="đối chiếu y_true với hcm_schema local dù data_fp khác (PM2.5 không đổi, chỉ khí tượng đổi)")
    a = ap.parse_args()
    cfg, rcfg, log = load_config(), load_rnn_config(), Log()

    runs_all = runs_with_fp(pd.read_csv(RUNS_LOG))
    local = try_data_provenance(cfg)
    fp = select_fp(runs_all, a.data_fp, local and local["data_fp"])
    runs = runs_all[runs_all["data_fp"] == fp].copy()
    dvs = sorted(runs["data_version"].dropna().astype(str).unique())
    dv = dvs[0] if len(dvs) == 1 else None
    if fp == LEGACY_FP:
        log.warn("Đây là các lần chạy CŨ (trước khi có dấu vân tay dữ liệu: v0.9-dev, khí tượng best_match). "
                 "Không đối chiếu được với dữ liệu hiện tại; chỉ là kết quả thử nghiệm.")
    if len(dvs) > 1:
        log.err(f"Cùng data_fp nhưng nhiều data_version {dvs}")

    # 7. dấu vân tay thành phần + bản khoá của nhóm
    comp = {}
    for k in ("raw_fp", "hcm_fp", "beijing_fp"):
        v = runs[k].dropna().astype(str).unique() if k in runs else []
        if len(v) > 1:
            log.err(f"Các lần chạy có nhiều giá trị {k}: {list(v)} (dữ liệu bị đổi giữa chừng)")
        comp[k] = v[0] if len(v) == 1 else None
    lock_diff = compare_lock({"data_version": dv, **comp}, load_lock()) if fp != LEGACY_FP else ["dữ liệu cũ"]
    if lock_diff and fp != LEGACY_FP:
        log.warn("Chưa khớp bản dữ liệu nhóm đã khoá -> chưa thể là kết quả chính thức: " + "; ".join(lock_diff))

    # 1-2. đủ cấu hình + siêu tham số
    for c in a.cells:
        b = (rcfg.get("best") or {}).get(c) or {}
        if b.get("data_fp") != fp and fp != LEGACY_FP:
            log.err(f"configs/rnn.yaml best.{c} có data_fp={b.get('data_fp')}, khác {fp}: dùng configs/rnn.yaml "
                    f"tải về cùng kết quả GPU")
    exp = expected_table(rcfg, a.groups, a.cells)
    cov = check_coverage(runs, exp, log)
    if a.require_gpu and (runs["device"] != "cuda").any():
        log.err(f"{int((runs['device'] != 'cuda').sum())} lần chạy không chạy trên GPU (cột device)")

    # 3-6. từng file dự đoán
    res = pd.read_csv(results_file(cfg, "rnn"))
    res_all = res[res["station"] == "ALL"]
    base = pd.read_csv(results_file(cfg, "baselines"))
    base_p = base[(base["model"] == "persistence") & (base["station"] == "ALL")]
    if dv:
        sel = base_p[base_p["notes"].astype(str).str.contains(f"data={dv};", regex=False)]
        base_p = sel if len(sel) else base_p
    preds_dir = path_of(cfg, "preds_dir")
    truth = None
    if local and (local["data_fp"] == fp or a.truth_anyway):
        h = pd.read_parquet(path_of(cfg, "hcm_schema"))
        h = h[h["city"] == "hcm"]
        truth = h.assign(v=h["pm25"].where(h["is_imputed"] == 0)).set_index(["station_id", "time"])["v"]
    else:
        log.warn("Dữ liệu local khác data_fp của kết quả (hoặc chưa dựng): bỏ qua bước 5 (đối chiếu hcm_schema).")
    ref_cache, n_files = {}, 0
    for _, run in runs.iterrows():
        rid, es = run["run_id"], run["eval_set"]
        lab = f"{rid}__{es}"
        f = preds_dir / f"{rid}__{es}.parquet"
        if not f.exists():
            log.err(f"{lab}: thiếu file dự đoán {f.relative_to(ROOT)}")
            continue
        pred = pd.read_parquet(f)
        n_files += 1
        check_pred_file(pred, run, cfg, log)
        if es not in ref_cache:
            bp = base_p[base_p["eval_set"] == es]
            rids = bp["run_id"].unique()
            rf = preds_dir / f"{rids[0]}__{es}.parquet" if len(rids) == 1 else None
            ref_cache[es] = (pd.read_parquet(rf) if rf is not None and rf.exists() else None, bp.set_index("horizon"))
            if ref_cache[es][0] is None:
                log.err(f"Không có dự đoán persistence cho {es} (results/baselines.csv có {len(rids)} run_id): "
                        f"chạy python -m src.baselines trên cùng dữ liệu")
        ref, bp = ref_cache[es]
        if ref is not None:
            compare_with_reference(pred, ref, lab, log)
        sc = scores(pred)
        for hz in HORIZONS:
            if hz not in sc.index:
                continue
            if hz in bp.index and int(bp.loc[hz, "n"]) != int(sc.loc[hz, "n"]):
                log.err(f"{lab}: n tầm {hz} = {int(sc.loc[hz, 'n'])}, persistence = {int(bp.loc[hz, 'n'])}")
            rr = res_all[(res_all["run_id"] == rid) & (res_all["eval_set"] == es) & (res_all["horizon"] == hz)]
            if len(rr) != 1:
                log.err(f"{lab}: results/rnn.csv có {len(rr)} dòng ALL cho tầm {hz} (phải là 1)")
            elif abs(rr["MAE"].iloc[0] - sc.loc[hz, "MAE"]) > 6e-4 or int(rr["n"].iloc[0]) != int(sc.loc[hz, "n"]):
                log.err(f"{lab}: MAE/n tầm {hz} trong rnn.csv khác giá trị tính lại từ file dự đoán")
            if f"MAE_h{hz}" in run and abs(run[f"MAE_h{hz}"] - sc.loc[hz, "MAE"]) > 6e-4:
                log.err(f"{lab}: MAE_h{hz} trong rnn_runs.csv khác giá trị tính lại")
        st = res[(res["run_id"] == rid) & (res["eval_set"] == es) & (res["station"] != "ALL")]
        miss_st = set(cfg["stations"]["hcm"]) - set(st["station"])
        if miss_st:
            log.warn(f"{lab}: results/rnn.csv thiếu dòng theo trạm {sorted(miss_st)}")
        if truth is not None:
            check_truth(pred, truth, lab, log)

    ok = not log.errors
    devices = runs["device"].value_counts().to_dict()
    summary = {"checked_at": dt.datetime.now().isoformat(timespec="seconds"), "data_fp": fp, "data_version": dv,
               "meteo_model": next(iter(runs.get("meteo_model", pd.Series(dtype=str)).dropna().unique()), None),
               "groups": a.groups, "n_runs": int(len(runs)), "n_expected": int(len(exp)),
               "n_missing": int((cov["n_runs"] == 0).sum()), "n_pred_files": n_files, "devices": devices,
               "truth_checked": truth is not None, "ok": ok, **comp,
               "matches_data_lock": not lock_diff, "data_lock_diff": lock_diff,
               "n_errors": len(log.errors), "n_warnings": len(log.warnings)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "g2_49_check.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    cov_tbl = cov.groupby(["group", "model"]).agg(can=("n_runs", "size"), co=("n_runs", lambda s: int((s > 0).sum())))
    md = ["# G2-49. Kiểm tra kết quả LSTM / GRU / Seq2Seq", "",
          f"Sinh bởi `python -m src.models.rnn_check` lúc {summary['checked_at']}.", "",
          f"- Bộ dữ liệu: `data_version={dv}`, `data_fp={fp}`, khí tượng `{summary['meteo_model']}`",
          f"- Lần chạy: {len(runs)} (cần {len(exp)} cho các nhóm {', '.join(a.groups)}), file dự đoán đọc được: {n_files}",
          f"- Thiết bị: {devices}",
          f"- Đối chiếu y_true với hcm_schema local: {'có' if truth is not None else 'không (dữ liệu local khác)'}",
          f"- Khớp bản dữ liệu nhóm đã khoá (`data/DATA_LOCK.json`): {'CÓ' if not lock_diff else 'KHÔNG – ' + '; '.join(lock_diff)}",
          f"- **Kết luận: {'ĐẠT' if ok else 'KHÔNG ĐẠT'}** ({len(log.errors)} lỗi, {len(log.warnings)} cảnh báo)", "",
          "## Độ phủ cấu hình", "", cov_tbl.rename(columns={"can": "cần", "co": "có"}).reset_index().to_markdown(index=False), ""]
    if log.errors:
        md += ["## Lỗi", ""] + [f"- {e}" for e in log.errors[:200]] + ([f"- … và {len(log.errors) - 200} lỗi khác"]
                                                                       if len(log.errors) > 200 else []) + [""]
    if log.warnings:
        md += ["## Cảnh báo", ""] + [f"- {w}" for w in log.warnings] + [""]
    (OUT / "g2_49_check.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:9]))
    if log.errors:
        print("\n".join(f"  LỖI: {e}" for e in log.errors[:15]))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
