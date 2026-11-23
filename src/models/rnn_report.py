"""G2-46, G2-47 (+ bảng RQ1/RQ2/RQ3, đường cong dữ liệu) — Tổng hợp kết quả LSTM / GRU / Seq2Seq.

    python -m src.models.rnn_report                      # đọc results/rnn.csv, rnn_runs.csv, baselines.csv -> reports/rnn/
    python -m src.models.rnn_report --data-fp <data_fp>  # chọn bộ dữ liệu khi nhật ký có nhiều bộ

CHỈ tổng hợp các lần chạy của MỘT bộ dữ liệu (data_fp, xem src/models/provenance.py), không bao giờ trộn số của dữ
liệu cũ và mới. Trạng thái ghi ở đầu summary.md: "CHÍNH THỨC" chỉ khi ĐỦ cả ba điều kiện:
  (1) dữ liệu của MỌI lần chạy khớp từng dấu vân tay (raw_fp, hcm_fp, beijing_fp, data_version) với bản khoá của nhóm
      data/DATA_LOCK.json (commit cùng tag data-v1.0, G1-66). Chỉ ghi "v1.0" trong configs/data.yaml là KHÔNG đủ;
  (2) bản khoá ghi data_version = v1.0;
  (3) `python -m src.models.rnn_check` đã ĐẠT cho đúng data_fp đó.
Thiếu một điều kiện thì là "THỬ NGHIỆM / CHỜ KIỂM CHỨNG", kèm lý do.
Không train lại: chỉ đọc file kết quả (vd giải nén rnn_results.zip từ Kaggle rồi chạy lệnh này).

Đầu ra (reports/rnn/):
  rq1_test.csv / .md          RQ1: MAE, RMSE, R², MAE_high trên test ở t+1/6/24, TB ± độ lệch chuẩn 3 seed, so với persistence
  rq1_rolling.csv             RQ1 rolling-origin: theo fold và trung bình 4 fold
  g2_46_lstm_vs_gru.csv       G2-46: sai số + số tham số + thời gian train
  g2_47_seq2seq_vs_direct.csv G2-47: Seq2Seq vs dự báo trực tiếp
  rq2_ablation.csv            RQ2: a / b / c / d, % cải thiện so với (a)
  rq3_transfer.csv            RQ3: S0 / S1 / S2_full / S2_head / S3
  rq3_curve.csv + figures/rq3_curve.png   G2-45: MAE theo % dữ liệu train TP.HCM
  oracle.csv                  G2-48: khí tượng thật tại t+h so với thiết lập chính
  vs_paper2.csv               chuẩn bị G3-10: "TB tầm 1..24" của RNN, persistence và bài [2] (Rakholia et al. 2022)
  summary.md                  tất cả bảng trên dạng markdown để dán vào báo cáo
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402

from src.models.provenance import (LEGACY_FP, compare_lock, load_lock, runs_with_fp, select_fp,  # noqa: E402
                                   try_data_provenance)
from src.utils import ROOT  # noqa: E402

OUT = ROOT / "reports" / "rnn"
METRICS = ["MAE", "RMSE", "R2", "MAE_high"]
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e1"


def load(data_fp: str | None = None):
    r = pd.read_csv(ROOT / "results" / "rnn.csv")
    runs = runs_with_fp(pd.read_csv(ROOT / "results" / "rnn_runs.csv"))
    base = pd.read_csv(ROOT / "results" / "baselines.csv")
    local = try_data_provenance()
    fp = select_fp(runs, data_fp, local and local["data_fp"])
    runs = runs[runs["data_fp"] == fp].copy()
    r = r[r["station"] == "ALL"].merge(runs[["run_id", "eval_set"]], on=["run_id", "eval_set"])
    r["cell"] = r["notes"].str.extract(r"cell=(\w+)")
    r["arch"] = r["notes"].str.extract(r"arch=(\w+)")
    dvs = runs["data_version"].dropna().astype(str).unique()
    dv = dvs[0] if len(dvs) == 1 else None
    if dv is None and fp == LEGACY_FP:                 # nhật ký cũ không có cột data_version: lấy từ notes của rnn.csv
        v = r["notes"].astype(str).str.extract(r"data=([^;\s]+)")[0].dropna().unique()
        dv = v[0] if len(v) == 1 else None
    base = base[base["station"] == "ALL"]
    if dv:
        sel = base[base["notes"].astype(str).str.contains(f"data={dv};", regex=False)]
        base = sel if len(sel) else base
    comp = {}
    for k in ("raw_fp", "hcm_fp", "beijing_fp"):
        v = runs[k].dropna().astype(str).unique() if k in runs else []
        comp[k] = v[0] if len(v) == 1 else (None if len(v) == 0 else "NHIỀU GIÁ TRỊ")
    meta = {"data_fp": fp, "data_version": dv, "local_fp": local and local["data_fp"], **comp,
            "meteo_model": next(iter(runs.get("meteo_model", pd.Series(dtype=str)).dropna().unique()), None),
            "devices": runs["device"].value_counts().to_dict() if "device" in runs else {}}
    return r, runs, base, meta


def status_of(meta: dict, chk: dict | None = None, lock="auto") -> tuple[bool, str]:
    """(chính thức?, câu mô tả). Chính thức = khớp bản khoá dữ liệu của nhóm (v1.0) + rnn_check ĐẠT cho đúng data_fp.

    chk, lock: truyền vào để test; mặc định đọc reports/rnn/g2_49_check.json và data/DATA_LOCK.json.
    """
    if chk is None:
        chk_path = OUT / "g2_49_check.json"
        chk = json.loads(chk_path.read_text(encoding="utf-8")) if chk_path.exists() else {}
    checked = bool(chk.get("ok")) and chk.get("data_fp") == meta["data_fp"]
    lock = load_lock() if lock == "auto" else lock
    lock_diff = compare_lock(meta, lock)
    if meta["data_fp"] == LEGACY_FP:
        return False, ("**KẾT QUẢ THỬ NGHIỆM (dữ liệu cũ).** Chạy trên CPU với `data v0.9-dev` và khí tượng Open-Meteo "
                       "`best_match` (trước khi nhóm cố định `era5_seamless`). Chỉ dùng để thử pipeline, KHÔNG dùng để "
                       "kết luận; mọi số sẽ được thay bằng lần chạy trên `data-v1.0` (GPU).")
    if not lock_diff and lock.get("data_version") == "v1.0" and checked:
        return True, (f"**KẾT QUẢ CHÍNH THỨC** (`data-v1.0`; khớp bản khoá `data/DATA_LOCK.json`: raw_fp "
                      f"`{meta['raw_fp']}`, hcm_fp `{meta['hcm_fp']}`, beijing_fp `{meta['beijing_fp']}`; đã qua "
                      f"`rnn_check` lúc {chk.get('checked_at')}).")
    why = []
    if meta["data_version"] != "v1.0":
        why.append(f"data_version = {meta['data_version']} (chưa phải v1.0)")
    if lock_diff:
        why.append("chưa khớp bản dữ liệu nhóm đã khoá: " + "; ".join(lock_diff))
    elif lock.get("data_version") != "v1.0":
        why.append(f"bản khoá ghi data_version = {lock.get('data_version')}")
    if not checked:
        why.append("chưa qua `python -m src.models.rnn_check` cho đúng data_fp này")
    return False, "**KẾT QUẢ THỬ NGHIỆM / CHỜ KIỂM CHỨNG:** " + "; ".join(why) + "."


def mean_std(df, by, metrics=METRICS):
    g = df.groupby(by)[metrics]
    m, s, n = g.mean(), g.std(), g.size().rename("n_seeds")
    out = m.round(3).astype(str) + " ± " + s.fillna(0).round(3).astype(str)
    return m, s, out.join(n)


def persistence(base, eval_set="test"):
    p = base[(base["model"] == "persistence") & (base["eval_set"] == eval_set)].set_index("horizon")
    return p


def rq1(r, base):
    d = r[(r["strategy"] == "S0") & (r["ablation"] == "c") & (r["eval_set"] == "test") & (r["train_frac"] == 1.0)]
    m, s, txt = mean_std(d, ["model", "cell", "horizon"])
    p = persistence(base)
    m = m.reset_index()
    m["MAE_persistence"] = m["horizon"].map(p["MAE"])
    m["skill_vs_persistence_pct"] = ((1 - m["MAE"] / m["MAE_persistence"]) * 100).round(1)
    hm = base[(base["model"] == "hourly_mean") & (base["eval_set"] == "test")].set_index("horizon")
    m["MAE_hourly_mean"] = m["horizon"].map(hm["MAE"])
    return m, txt.reset_index()


def rolling(r, base):
    d = r[(r["strategy"] == "S0") & (r["ablation"] == "c") & r["eval_set"].str.startswith("fold")]
    if d.empty:
        return pd.DataFrame()
    per = d.pivot_table(index=["model", "horizon"], columns="eval_set", values="MAE")
    per["mean_4_fold"] = per.mean(axis=1)
    pb = base[(base["model"] == "persistence") & base["eval_set"].str.startswith("fold")]
    per["persistence_mean_4_fold"] = pb.groupby("horizon")["MAE"].mean().reindex(per.index.get_level_values(1)).values
    return per.round(3).reset_index()


def seq2seq_vs_direct(r):
    d = r[(r["strategy"] == "S0") & (r["ablation"] == "c") & (r["eval_set"] == "test") & (r["train_frac"] == 1.0)]
    m, s, txt = mean_std(d, ["arch", "cell", "horizon"], ["MAE", "RMSE", "R2", "MAE_high"])
    return m.round(3).reset_index(), txt.reset_index()


def rq2(r):
    d = r[(r["strategy"] == "S0") & (r["eval_set"] == "test") & (r["train_frac"] == 1.0) & (r["arch"] == "direct")
          & r["ablation"].isin(["a", "b", "c", "d"])]
    m = d.groupby(["model", "ablation", "horizon"])["MAE"].agg(["mean", "std", "size"]).reset_index()
    a = m[m["ablation"] == "a"].set_index(["model", "horizon"])["mean"]
    m["improve_vs_a_pct"] = ((1 - m["mean"] / m.set_index(["model", "horizon"]).index.map(a)) * 100).round(1)
    return m.round(3)


def rq3(r):
    d = r[(r["arch"] == "direct") & (r["ablation"] == "c") & (r["eval_set"] == "test")]
    m = d.groupby(["model", "strategy", "train_frac", "horizon"])["MAE"].mean().reset_index()
    return m.round(3)


def curve_fig(c):
    c = c[c["strategy"].isin(["S0", "S2_full", "S3"])]
    if c.empty or c["train_frac"].nunique() < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
    colors = {"S0": BLUE, "S2_full": ORANGE, "S3": AQUA}
    names = {"S0": "S0 chỉ TP.HCM", "S2_full": "S2 pretrain Bắc Kinh + fine-tune", "S3": "S3 train chung"}
    for ax, h in zip(axes, (1, 6, 24)):
        for model, ls in (("LSTM", "-"), ("GRU", "--")):
            for st, col in colors.items():
                g = c[(c["model"] == model) & (c["strategy"] == st) & (c["horizon"] == h)].sort_values("train_frac")
                if len(g):
                    ax.plot(g["train_frac"] * 100, g["MAE"], ls, color=col, marker="o", ms=6, lw=2,
                            label=f"{names[st]} ({model})")
        ax.set_title(f"t+{h}h", loc="left", fontsize=11, color=INK)
        ax.set_xlabel("% dữ liệu train TP.HCM", color=MUTED)
        ax.set_xticks([25, 50, 100])
        ax.grid(color=GRID)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axes[0].set_ylabel("MAE trên test (µg/m³)", color=MUTED)
    axes[-1].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle("G2-45. Dữ liệu Bắc Kinh giúp nhiều hơn khi dữ liệu TP.HCM ít? (nét liền LSTM, nét đứt GRU)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / "rq3_curve.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def lstm_vs_gru(r, runs):
    """G2-46: LSTM vs GRU cùng thiết lập chính (trực tiếp, ablation c, S0, test, 100%), TB ± std qua seed."""
    d = r[(r["arch"] == "direct") & (r["strategy"] == "S0") & (r["ablation"] == "c") & (r["eval_set"] == "test")
          & (r["train_frac"] == 1.0)]
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    g = d.groupby(["horizon", "model"])
    m = g[["MAE", "RMSE", "R2", "MAE_high"]].mean().unstack("model")
    sd = g["MAE"].std().unstack("model")
    n = g.size().unstack("model")
    out = pd.DataFrame(index=m.index)
    for mod in [c for c in ("LSTM", "GRU") if c in m["MAE"]]:
        out[f"{mod} MAE"] = m["MAE"][mod].round(3).astype(str) + " ± " + sd[mod].fillna(0).round(3).astype(str)
        out[f"{mod} RMSE"] = m["RMSE"][mod].round(3)
        out[f"{mod} MAE_high"] = m["MAE_high"][mod].round(3)
        out[f"{mod} số seed"] = n[mod]
    if {"LSTM", "GRU"} <= set(m["MAE"].columns):
        out["GRU so với LSTM (MAE, %)"] = ((m["MAE"]["GRU"] / m["MAE"]["LSTM"] - 1) * 100).round(1)
    rr = runs[(runs["arch"] == "direct") & (runs["strategy"] == "S0") & (runs["ablation"] == "c")
              & (runs["eval_set"] == "test") & (runs["train_frac"] == 1.0)]
    cost = rr.groupby("model").agg(L=("L", "first"), params=("params", "first"), n_seeds=("seed", "nunique"),
                                   train_sec_TB=("train_sec", "mean"), epoch_tot_nhat_TB=("best_epoch", "mean"),
                                   thiet_bi=("device", lambda s: ",".join(sorted(set(map(str, s))))))
    cost["giay_moi_epoch"] = (rr.groupby("model")["train_sec"].sum() / rr.groupby("model")["epochs_run"].sum())
    hp = rr.groupby("model")["hp"].first().map(json.loads)
    cost["hidden/layers/dropout/lr/batch"] = hp.map(lambda h: f"{h['hidden']}/{h['layers']}/{h['dropout']}/"
                                                            f"{h['lr']}/{h['batch_size']}")
    return out.reset_index(), cost.round(2).reset_index()


def seq2seq_table(r, runs):
    """G2-47: Seq2Seq so với trực tiếp, CÙNG cell và cùng siêu tham số, TB ± std qua seed."""
    d = r[(r["strategy"] == "S0") & (r["ablation"] == "c") & (r["eval_set"] == "test") & (r["train_frac"] == 1.0)]
    if d.empty or d["arch"].nunique() < 2:
        return pd.DataFrame()
    g = d.groupby(["cell", "horizon", "arch"])["MAE"]
    m, sd, n = g.mean().unstack("arch"), g.std().unstack("arch"), g.size().unstack("arch")
    out = pd.DataFrame(index=m.index)
    for a_ in ("direct", "seq2seq"):
        out[f"{a_} MAE"] = m[a_].round(3).astype(str) + " ± " + sd[a_].fillna(0).round(3).astype(str)
        out[f"{a_} số seed"] = n[a_]
    out["Seq2Seq so với trực tiếp (MAE, %)"] = ((m["seq2seq"] / m["direct"] - 1) * 100).round(1)
    rr = runs[(runs["strategy"] == "S0") & (runs["ablation"] == "c") & (runs["eval_set"] == "test")
              & (runs["train_frac"] == 1.0)]
    t = rr.groupby(["cell", "arch"])["train_sec"].mean().unstack("arch")
    out = out.reset_index()
    out["thời gian train Seq2Seq / trực tiếp"] = out["cell"].map((t["seq2seq"] / t["direct"]).round(2))
    return out


def oracle(r):
    d = r[(r["eval_set"] == "test") & (r["strategy"] == "S0") & (r["arch"] == "direct") & (r["train_frac"] == 1.0)
          & r["ablation"].isin(["c", "c_oracle"])]
    return d.groupby(["model", "ablation", "horizon"])["MAE"].mean().unstack("ablation").round(3).reset_index()


def persistence_avg24() -> dict:
    """Persistence trên CÙNG tập gốc t test (L_ref = 72), gộp tầm 1..24 — mốc so sánh cho bảng bài [2]."""
    from src.baselines import FEATS, L_REF, persistence
    from src.data.pipeline import prepare
    ws = prepare("hcm").windows("test", FEATS, L=L_REF)
    yt, yp = ws.y_true_ug(), persistence(ws)
    ok = ~np.isnan(yt)
    e = (yp - yt)[ok]
    return {"MAE_avg24": float(np.abs(e).mean()), "RMSE_avg24": float(np.sqrt((e ** 2).mean()))}


def vs_paper2(runs):
    rr = runs[(runs["strategy"] == "S0") & (runs["ablation"].isin(["a", "c"])) & (runs["eval_set"] == "test")
              & (runs["train_frac"] == 1.0)]
    if "MAE_avg24" not in rr or rr["MAE_avg24"].isna().all():
        return pd.DataFrame()
    ours = (rr.groupby(["model", "ablation"])[["MAE_avg24", "RMSE_avg24"]].mean().round(3).reset_index()
              .assign(source="Nhóm (test 03–06/2022)"))
    p = persistence_avg24()
    ours = pd.concat([ours, pd.DataFrame([{"model": "persistence", "ablation": "-", **p,
                                           "source": "Nhóm (test 03–06/2022)"}])], ignore_index=True)
    ref_path = ROOT / "overview" / "G0-07_bai2_reference.csv"          # bảng số của G0-07 (đã commit)
    ref = pd.read_csv(ref_path).dropna(how="all") if ref_path.exists() else None
    if ref is not None:
        ref = ref[ref["task"] == "hourly"].pivot_table(index="model", columns="metric", values="value").reset_index()
        ref = ref.rename(columns={"MAE": "MAE_avg24", "RMSE": "RMSE_avg24"}).assign(
            ablation="chỉ PM2.5", source="Bài [2] Table 5 (bộ con 3 tháng 2021, TB theo trạm)")
        ours = pd.concat([ours, ref], ignore_index=True)
    return ours[["source", "model", "ablation", "MAE_avg24", "RMSE_avg24"]].round(3)


def design_note(runs):
    """So sánh v1 (không residual, đã lưu trữ) với v2 (residual) trên cùng cấu hình RQ1 của LSTM."""
    arch = ROOT / "results" / "archive" / "rnn_v1_no_residual" / "rnn_runs.csv"
    if not arch.exists():
        return pd.DataFrame()
    sel = lambda d: d[(d["strategy"] == "S0") & (d["ablation"] == "c") & (d["eval_set"] == "test")
                      & (d["train_frac"] == 1.0)]
    cols = ["MAE_h1", "MAE_h6", "MAE_h24", "MAE_avg24", "best_epoch"]
    v1 = sel(pd.read_csv(arch)).groupby("model")[cols].mean().assign(version="v1 không residual")
    v2 = sel(runs).groupby("model")[cols].mean().assign(version="v2 residual (dùng cho mọi kết quả)")
    return pd.concat([v1, v2]).reset_index()[["version", "model"] + cols].round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-fp", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    r, runs, base, meta = load(a.data_fp)
    official, status = status_of(meta)
    md = ["# Kết quả LSTM / GRU / Seq2Seq (TV2 – Minh Trí)", "", "> " + status, "",
          f"Bộ dữ liệu: `data_version={meta['data_version']}`, `data_fp={meta['data_fp']}`, khí tượng "
          f"`{meta['meteo_model']}`; thiết bị {meta['devices']}; {len(runs)} lần chạy.", "",
          "Sinh tự động bởi `python -m src.models.rnn_report` từ `results/rnn.csv`. Đơn vị µg/m³, chỉ tính trên giờ đo thật,",
          "tập gốc t chung với mọi mô hình của nhóm (cột n bằng nhau). ± = độ lệch chuẩn qua các seed.", ""]
    if not official:
        md += ["_Các nhận xét về khí tượng (RQ2) và dữ liệu Bắc Kinh (RQ3) trong báo cáo CHƯA được rút ra từ bảng này._", ""]

    def add(title, df, fname):
        if df is None or len(df) == 0:
            md.extend([f"## {title}", "", "_Chưa có kết quả._", ""])
            return
        df.to_csv(OUT / fname, index=False)
        md.extend([f"## {title}", "", df.to_markdown(index=False), ""])

    md.extend(["## Ghi chú thiết kế: vì sao dùng kết nối phần dư (residual)", "",
               "Lượt chạy đầu (v1) dự báo thẳng mức PM2.5. Mô hình overfit sau 2–3 epoch (train loss giảm mạnh, val loss tăng),",
               "early stopping dừng sớm và LSTM trực tiếp **thua persistence ở t+1**. Bản v2 dự báo độ thay đổi so với PM2.5 tại t",
               "(ŷ(t+k) = y(t) + Δk). Đầu vào giữ nguyên; HPO 30 lần thử được chạy LẠI cho v2. Kết quả v1 lưu ở",
               "`results/archive/rnn_v1_no_residual/` (MAE, TB 3 seed):", ""])
    dn = design_note(runs)
    if len(dn):
        dn.to_csv(OUT / "design_note_residual.csv", index=False)
        p = persistence(base)
        ptxt = "; ".join(f"t+{h} = {p.loc[h, 'MAE']:.2f}" for h in (1, 6, 24) if h in p.index)
        md.extend([dn.to_markdown(index=False), "",
                   "v1 chạy trên `v0.9-dev` + khí tượng `best_match`; chỉ để minh hoạ lý do chọn residual.", "",
                   f"Persistence trên cùng test: MAE {ptxt}.", ""])
    m1, t1 = rq1(r, base)
    add("RQ1 – Test chính (G2-34, G2-32: TB ± std 3 seed)", t1, "rq1_test.csv")
    add("RQ1 – So với baseline", m1[["model", "cell", "horizon", "MAE", "MAE_persistence", "MAE_hourly_mean",
                                     "skill_vs_persistence_pct"]].round(3), "rq1_vs_baseline.csv")
    add("RQ1 – Rolling-origin 4 fold (G2-35), MAE", rolling(r, base), "rq1_rolling.csv")
    g46, g46c = lstm_vs_gru(r, runs)
    add("G2-46 – LSTM vs GRU: sai số (trực tiếp, ablation c, S0, test)", g46, "g2_46_lstm_vs_gru.csv")
    add("G2-46 – LSTM vs GRU: kích thước và chi phí", g46c, "g2_46_cost.csv")
    add("G2-47 – Seq2Seq vs dự báo trực tiếp (cùng cell, cùng siêu tham số)", seq2seq_table(r, runs),
        "g2_47_seq2seq_vs_direct.csv")
    add("RQ2 – Ablation (G2-36..39), MAE TB 3 seed", rq2(r), "rq2_ablation.csv")
    c = rq3(r)
    add("RQ3 – Chiến lược dùng dữ liệu Bắc Kinh (G2-40..45), MAE", c, "rq3_transfer.csv")
    curve_fig(c)
    add("G2-48 – Khí tượng thật tại t+h (c_oracle) vs thiết lập chính (c)", oracle(r), "oracle.csv")
    add("Chuẩn bị G3-10 – So với bài [2] (TB tầm 1..24). Lưu ý: KHÁC tập test và cách chia nên chỉ tham khảo",
        vs_paper2(runs), "vs_paper2.csv")
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
