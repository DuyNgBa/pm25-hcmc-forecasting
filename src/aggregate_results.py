"""G3-01, G3-04 — Gộp và kiểm tra mọi results/<phương-pháp>.csv; bảng + đường cong RQ3.

    python -m src.aggregate_results

Đầu ra (reports/):
  results_check.md          — ô còn thiếu, số bất thường, mô hình thua persistence
  results_summary.csv       — trung bình ± độ lệch chuẩn theo seed (station = ALL)
  rq3_table.csv/.md         — bảng S0–S3 × train_frac × tầm (MAE, RMSE) cho từng mô hình
  figures/rq3_data_curve.png — MAE theo % dữ liệu train TP.HCM (mỗi tầm một panel)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils import ROOT, load_config  # noqa: E402

REP = ROOT / "reports"
MODELS = {"Duy Ngô": ["LightGBM", "XGBoost", "RandomForest", "Ridge"],
          "Minh Trí": ["LSTM", "GRU", "Seq2Seq"], "Bo bo": ["PatchTST"]}
STRATEGY_COLOR = {"S0": "#2a78d6", "S2_full": "#eb6834", "S2_head": "#1baf7a", "S3": "#6250d6", "S2": "#eb6834"}


def load(cfg) -> pd.DataFrame:
    """Gộp mọi results/<phương-pháp>.csv (bỏ file mẫu, file tham chiếu Bắc Kinh, file HPO)."""
    from src.eval.results import RESULT_COLUMNS

    skip = {"results_template.csv", "beijing_reference.csv"}
    files = [p for p in sorted((ROOT / cfg["paths"]["results_dir"]).glob("*.csv"))
             if p.name not in skip and not p.name.startswith("hpo_")]
    if not files:
        raise FileNotFoundError("Chưa có file kết quả nào trong results/")
    parts = []
    for p in files:
        d = pd.read_csv(p)
        miss = [c for c in RESULT_COLUMNS if c not in d.columns]
        if miss:
            print(f"[CẢNH BÁO] {p.name} thiếu cột {miss} -> bỏ qua; sửa theo results_template.csv")
            continue
        parts.append(d[RESULT_COLUMNS])
    df = pd.concat(parts, ignore_index=True)
    return df[~df["notes"].fillna("").str.contains("SMOKE")]


def check(df: pd.DataFrame) -> str:
    """Kiểm tra số bất thường và độ đầy đủ; trả về markdown."""
    lines = ["# Kiểm tra results.csv\n", f"Tổng số dòng: {len(df)}\n"]
    bad = df[(df["MAE"] < 0) | (df["RMSE"] < df["MAE"] - 1e-9) | (df["R2"] > 1) | df["MAE"].isna()]
    lines.append(f"## Dòng bất thường (MAE<0, RMSE<MAE, R2>1, NaN): {len(bad)}\n")
    if len(bad):
        lines.append(bad.head(30).to_markdown(index=False) + "\n")
    # Tập đánh giá chung -> n phải giống nhau giữa MỌI mô hình/L/ablation cùng (eval_set, horizon)
    allr = df[df["station"] == "ALL"]
    nvar = allr.groupby(["eval_set", "horizon"])["n"].nunique()
    lines.append("## Số giờ đánh giá (n) khác nhau giữa các mô hình cùng (eval_set, tầm)\n")
    lines.append((nvar[nvar > 1].to_frame("số giá trị n khác nhau").to_markdown() if (nvar > 1).any()
                  else "Không có — mọi mô hình đánh giá trên cùng tập giờ.") + "\n")
    # thua persistence
    pers = allr[allr["model"] == "persistence"].set_index(["eval_set", "horizon"])["MAE"]
    others = allr[~allr["model"].isin(["persistence", "hourly_mean"])].copy()
    others["persistence_MAE"] = [pers.get((e, h), np.nan) for e, h in zip(others["eval_set"], others["horizon"])]
    lose = others[others["MAE"] >= others["persistence_MAE"]]
    lines.append(f"## Cấu hình KHÔNG thắng persistence (MAE): {len(lose)}\n")
    if len(lose):
        lines.append(lose[["member", "model", "strategy", "ablation", "train_frac", "L", "seed", "eval_set",
                           "horizon", "MAE", "persistence_MAE"]].to_markdown(index=False) + "\n")
    # ô thiếu: mỗi mô hình phải có đủ 3 tầm cho test
    lines.append("## Độ phủ (số dòng ALL trên test theo mô hình × chiến lược × ablation)\n")
    cov = allr[allr["eval_set"] == "test"].pivot_table(index=["member", "model"], columns=["strategy", "ablation"],
                                                        values="MAE", aggfunc="count")
    lines.append(cov.fillna(0).astype(int).to_markdown() + "\n")
    return "\n".join(lines)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    allr = df[df["station"] == "ALL"]
    keys = ["member", "model", "strategy", "ablation", "train_frac", "L", "eval_set", "horizon"]
    agg = allr.groupby(keys).agg(MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"), RMSE_mean=("RMSE", "mean"),
                                 RMSE_std=("RMSE", "std"), R2_mean=("R2", "mean"), MAE_high_mean=("MAE_high", "mean"),
                                 AQI_acc_mean=("AQI_acc", "mean"), AQI_macroF1_mean=("AQI_macroF1", "mean"),
                                 n_seeds=("seed", "nunique"), n=("n", "first")).reset_index()
    return agg


def rq3(summ: pd.DataFrame):
    d = summ[(summ["eval_set"] == "test") & summ["strategy"].isin(["S0", "S1", "S2", "S2_full", "S2_head", "S3"])
             & summ["ablation"].isin(["c", "a"]) & ~summ["model"].isin(["persistence", "hourly_mean"])]
    if d.empty:
        print("Chưa có kết quả RQ3.")
        return
    d = d.assign(cell=d["MAE_mean"].map("{:.2f}".format) + " ± " + d["MAE_std"].fillna(0).map("{:.2f}".format))
    tbl = d.pivot_table(index=["model", "ablation", "strategy"], columns=["train_frac", "horizon"], values="cell",
                        aggfunc="first")
    tbl.to_csv(REP / "rq3_table.csv")
    (REP / "rq3_table.md").write_text("MAE (µg/m³, TB ± độ lệch chuẩn qua seed), test TP.HCM\n\n" + tbl.to_markdown(),
                                     encoding="utf-8")
    for model, g in d.groupby("model"):
        hs = sorted(g["horizon"].unique())
        fig, axes = plt.subplots(1, len(hs), figsize=(4 * len(hs), 3.6), sharey=False)
        axes = np.atleast_1d(axes)
        for ax, h in zip(axes, hs):
            for st, gs in g[g["horizon"] == h].groupby("strategy"):
                gs = gs.sort_values("train_frac")
                if st == "S1":   # zero-shot không phụ thuộc lượng dữ liệu TP.HCM -> đường ngang
                    ax.axhline(gs["MAE_mean"].iloc[0], color="#52514e", ls="--", lw=1.2, label="S1 (zero-shot)")
                    continue
                ax.errorbar(gs["train_frac"] * 100, gs["MAE_mean"], yerr=gs["MAE_std"].fillna(0), marker="o", ms=5,
                            lw=2, capsize=3, color=STRATEGY_COLOR.get(st, "#888"), label=st)
            ax.set(title=f"t+{h}h", xlabel="% dữ liệu train TP.HCM", xticks=[25, 50, 100])
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#e6e5e1")
        axes[0].set_ylabel("MAE trên test (µg/m³)")
        axes[-1].legend(frameon=False, fontsize=8)
        fig.suptitle(f"RQ3 — {model}: dữ liệu Bắc Kinh giúp bao nhiêu khi dữ liệu TP.HCM ít?", fontsize=11)
        fig.tight_layout()
        fig.savefig(REP / "figures" / f"rq3_data_curve_{model}.png", dpi=150)
        plt.close(fig)


def main():
    cfg = load_config()
    df = load(cfg)
    (REP / "results_check.md").write_text(check(df), encoding="utf-8")
    s = summary(df)
    s.to_csv(REP / "results_summary.csv", index=False, float_format="%.4f")
    rq3(s)
    print(f"Đã ghi reports/results_check.md, results_summary.csv ({len(s)} dòng), rq3_table.*")


if __name__ == "__main__":
    main()
