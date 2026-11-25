"""G1-64 — So sánh Bắc Kinh và TP.HCM (cập nhật bảng 4.1 của đề xuất).

    python -m src.eda.compare_cities

Đầu ra (cùng thư mục với EDA HealthyAir của TV1):
  reports/eda/g1_64_beijing_vs_hcm.csv / .md   — bảng số liệu (cập nhật bảng 4.1)
  reports/eda/figures/g1_64_*.png              — hình cho báo cáo
Tính trên dữ liệu SAU làm sạch, chỉ giá trị đo thật (is_imputed = 0), toàn bộ giai đoạn.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.pipeline import load_schema  # noqa: E402
from src.utils import ROOT, load_config  # noqa: E402

# Bảng màu tham chiếu (dataviz skill, slot 1–2): xanh = TP.HCM, cam = Bắc Kinh
COL = {"hcm": "#2a78d6", "beijing": "#eb6834"}
NAME = {"hcm": "TP.HCM", "beijing": "Bắc Kinh"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
EDA_DIR = ROOT / "reports" / "eda"
FIG = EDA_DIR / "figures"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def real_pm25(df):
    d = df[["station_id", "time", "pm25", "is_imputed"]].copy()
    d.loc[d["is_imputed"] == 1, "pm25"] = np.nan
    return d


def stats(df: pd.DataFrame) -> dict:
    d = real_pm25(df)
    p = d["pm25"].dropna()
    wide = d.pivot(index="time", columns="station_id", values="pm25")
    corr = wide.corr(min_periods=500).to_numpy()
    iu = np.triu_indices_from(corr, k=1)
    acf1 = np.nanmean([wide[c].autocorr(1) for c in wide])
    acf24 = np.nanmean([wide[c].autocorr(24) for c in wide])
    diurnal = d.groupby(d["time"].dt.hour)["pm25"].mean()
    monthly = d.groupby(d["time"].dt.month)["pm25"].mean()
    return {
        "Số trạm": wide.shape[1],
        "Số giờ mỗi trạm (lưới)": len(wide),
        "Tổng số giờ đo thật": int(p.size),
        "% giờ thiếu (sau làm sạch)": round(100 * (1 - p.size / wide.size), 1),
        "PM2.5 trung bình (µg/m³)": round(p.mean(), 1),
        "PM2.5 trung vị (µg/m³)": round(p.median(), 1),
        "PM2.5 phân vị 95 (µg/m³)": round(p.quantile(0.95), 1),
        "% giờ > 35,4 µg/m³": round(100 * (p > 35.4).mean(), 1),
        "Giờ đỉnh chu kỳ ngày": int(diurnal.idxmax()),
        "Giờ thấp nhất chu kỳ ngày": int(diurnal.idxmin()),
        "Tháng cao nhất": int(monthly.idxmax()),
        "Tháng thấp nhất": int(monthly.idxmin()),
        "Tỉ số tháng cao/thấp": round(monthly.max() / monthly.min(), 2),
        "Tương quan TB giữa các trạm": round(np.nanmean(corr[iu]), 2),
        "ACF lag 1h (TB các trạm)": round(acf1, 2),
        "ACF lag 24h (TB các trạm)": round(acf24, 2),
    }, d


def plot_profiles(ds: dict):
    # Chu kỳ ngày & năm: chuẩn hoá theo trung bình của từng thành phố (tỉ số so với TB) để 2 thành phố
    # cùng một trục mà không cần trục kép (mức nồng độ chênh ~4 lần).
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for city, d in ds.items():
        m = d["pm25"].mean()
        di = d.groupby(d["time"].dt.hour)["pm25"].mean() / m
        mo = d.groupby(d["time"].dt.month)["pm25"].mean() / m
        axes[0].plot(di.index, di.values, color=COL[city], lw=2, label=NAME[city])
        axes[1].plot(mo.index, mo.values, color=COL[city], lw=2, marker="o", ms=4, label=NAME[city])
        axes[0].annotate(NAME[city], (23, di.values[-1]), xytext=(4, 0), textcoords="offset points",
                         color=INK, fontsize=9, va="center")
    axes[0].set(xlabel="Giờ trong ngày (giờ địa phương)", ylabel="PM2.5 / trung bình của thành phố",
                title="Chu kỳ ngày", xticks=range(0, 24, 3))
    axes[1].set(xlabel="Tháng", title="Chu kỳ năm", xticks=range(1, 13))
    for ax in axes:
        _style(ax)
        ax.axhline(1, color=MUTED, lw=0.8, ls=":")
    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "g1_64_cycles.png", dpi=150)
    plt.close(fig)


def plot_distribution(ds: dict):
    """Mật độ của log10(PM2.5) (KDE). Dùng KDE thay histogram vì PM2.5 Bắc Kinh ghi số NGUYÊN ->
    histogram trên thang log có các "gai" giả ở giá trị nhỏ."""
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    grid = np.linspace(0, np.log10(600), 400)
    for city, d in ds.items():
        v = np.log10(d["pm25"].dropna().to_numpy())
        v = np.random.default_rng(0).choice(v, min(len(v), 60000), replace=False)
        ax.plot(10 ** grid, gaussian_kde(v, bw_method=0.08)(grid), color=COL[city], lw=2,
                label=f"{NAME[city]} (trung vị {d['pm25'].median():.1f})")
    ax.axvline(35.4, color=MUTED, lw=1, ls="--")
    ax.text(37, ax.get_ylim()[1] * 0.92, "35,4 µg/m³", color=MUTED, fontsize=8)
    ax.set_xscale("log")
    ax.set(xlabel="PM2.5 (µg/m³, thang log)", ylabel="Mật độ của log10(PM2.5)", title="Phân phối PM2.5 theo giờ")
    _style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "g1_64_distribution.png", dpi=150)
    plt.close(fig)


def plot_acf(ds: dict, max_lag: int = 72):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for city, d in ds.items():
        wide = d.pivot(index="time", columns="station_id", values="pm25")
        acf = [np.nanmean([wide[c].autocorr(k) for c in wide]) for k in range(1, max_lag + 1)]
        ax.plot(range(1, max_lag + 1), acf, color=COL[city], lw=2, label=NAME[city])
    for k in (1, 24, 48):
        ax.axvline(k, color=GRID, lw=1)
    ax.set(xlabel="Độ trễ (giờ)", ylabel="Tự tương quan (TB các trạm)", title="ACF của PM2.5", xticks=[1, 12, 24, 36, 48, 60, 72])
    _style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "g1_64_acf.png", dpi=150)
    plt.close(fig)


def main():
    cfg = load_config()
    FIG.mkdir(parents=True, exist_ok=True)
    res, ds = {}, {}
    for city in ["beijing", "hcm"]:
        res[NAME[city]], ds[city] = stats(load_schema(cfg, city))
    tbl = pd.DataFrame(res)
    tbl.to_csv(EDA_DIR / "g1_64_beijing_vs_hcm.csv")
    (EDA_DIR / "g1_64_beijing_vs_hcm.md").write_text(tbl.to_markdown(), encoding="utf-8")
    plot_profiles(ds)
    plot_distribution(ds)
    plot_acf(ds)
    print(tbl.to_string())


if __name__ == "__main__":
    main()
