"""G1-42, G1-43, G1-44 — EDA: quan hệ giữa khí tượng (Open-Meteo) và PM2.5 tại TP.HCM.

    python -m src.eda.eda_meteo      # cần data/interim/hcm_merged.parquet (python -m src.data.merge_meteo)

Chỉ dùng giờ có PM2.5 ĐO THẬT (is_imputed = 0, pm25 không NaN). Tương quan Spearman vì PM2.5 lệch phải.
Đầu ra (reports/meteo/ và reports/meteo/figures/):
  G1-42  g1_42_spearman_by_station.csv + g1_42_spearman_heatmap.png   tương quan từng biến × từng trạm
         g1_42_lead_corr.csv            khí tượng tại t vs PM2.5 tại t+h (h = 0, 1, 6, 24)
         g1_42_monthly.csv + g1_42_monthly.png   PM2.5 và lượng mưa theo tháng
  G1-43  g1_43_rain_vs_dry.csv + g1_43_rain_vs_dry.png   PM2.5 khi có mưa / không mưa (6 giờ qua), theo trạm
         g1_43_pm25_by_wind_speed.csv, g1_43_pm25_by_rain.csv + g1_43_wind_rain.png   theo khoảng gió / mưa 24h
  G1-44  g1_44_wind_sector_by_station.csv + g1_44_wind_rose.png   hoa gió từng trạm (16 hướng)
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap  # noqa: E402

from src.data.openmeteo_common import load_om_config, rpath  # noqa: E402

FEATS = ["temp", "rh", "dew", "pressure", "cloud_cover", "precip", "precip_6h", "precip_24h",
         "hours_since_rain", "wind_speed", "wind_speed_6h", "wind_gusts", "wind_u", "wind_v"]
# Bảng màu: xanh = series 1, cam = series 2; thang tuần tự 1 màu (xanh nhạt -> đậm); phân kỳ xanh - xám - cam
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#256abf", "#86b6ef", "#f0efec", "#f2a27f", "#c94f1f"])
SECT8 = ["Bắc", "Đông Bắc", "Đông", "Đông Nam", "Nam", "Tây Nam", "Tây", "Tây Bắc"]
STATION_LABEL = {"hcm_01": "S1 nền đô thị", "hcm_02": "S2 giao thông", "hcm_03": "S3 công nghiệp",
                 "hcm_04": "S4 dân cư", "hcm_05": "S5 giao thông", "hcm_06": "S6 giao thông + dân cư"}


def style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def summary(s: pd.Series) -> pd.Series:
    return pd.Series({"n": s.size, "median": s.median(), "mean": s.mean(), "q25": s.quantile(.25),
                      "q75": s.quantile(.75), "pct_high": (s > 35.4).mean() * 100})


def binned(df, col, bins, labels):
    b = pd.cut(df[col], bins=bins, labels=labels, right=False)
    return df.groupby(b, observed=False)["pm25"].apply(summary).unstack().round(2)


def line_with_band(ax, tab, xlabel, title):
    x = np.arange(len(tab))
    ax.fill_between(x, tab["q25"], tab["q75"], color=BLUE, alpha=.15, lw=0, label="Khoảng tứ phân vị")
    ax.plot(x, tab["median"], color=BLUE, lw=2, marker="o", ms=8, label="Trung vị")
    ax.set_xticks(x, tab.index.astype(str))
    for xi, (m, n) in enumerate(zip(tab["median"], tab["n"])):
        ax.annotate(f"n={int(n)}", (xi, m), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=7, color=MUTED)
    style(ax, title, xlabel, "PM2.5 (µg/m³)")
    ax.legend(frameon=False, fontsize=8)


# ------------------------------------------------------------------ G1-42
def g1_42(full, obs, out, fig_dir):
    sp = {sid: g[FEATS].corrwith(g["pm25"], method="spearman") for sid, g in obs.groupby("station_id")}
    sp["ALL"] = obs[FEATS].corrwith(obs["pm25"], method="spearman")
    sp = pd.DataFrame(sp)
    sp = sp.reindex(sp["ALL"].abs().sort_values(ascending=False).index).round(3)
    sp.to_csv(out / "g1_42_spearman_by_station.csv")

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    lim = np.ceil(sp.abs().max().max() * 10) / 10
    im = ax.imshow(sp.values, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(sp.shape[1]), [STATION_LABEL.get(c, "6 trạm gộp") for c in sp.columns],
                  rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(sp.shape[0]), sp.index, fontsize=9)
    for i in range(sp.shape[0]):
        for j in range(sp.shape[1]):
            v = sp.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > lim * 0.6 else INK)
    ax.axvline(sp.shape[1] - 1.5, color="white", lw=3)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Spearman ρ với PM2.5", color=MUTED)
    ax.set_title("G1-42. Tương quan Spearman giữa khí tượng và PM2.5, theo trạm", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "g1_42_spearman_heatmap.png", dpi=130)
    plt.close(fig)

    rows = []
    for h in [0, 1, 6, 24]:
        fut = full.groupby("station_id")["pm25"].shift(-h)
        ok = fut.notna() & (full.groupby("station_id")["is_imputed"].shift(-h) == 0)
        for f in FEATS:
            rows.append({"feature": f, "h": h, "spearman": full.loc[ok, f].corr(fut[ok], method="spearman")})
    lead = pd.DataFrame(rows).pivot(index="feature", columns="h", values="spearman").round(3)
    lead.columns = [f"h={c}" for c in lead.columns]
    lead.reindex(lead["h=0"].abs().sort_values(ascending=False).index).to_csv(out / "g1_42_lead_corr.csv")

    full = full.assign(month=full["time"].dt.to_period("M"))
    obs = obs.assign(month=obs["time"].dt.to_period("M"))
    mon = pd.DataFrame({
        "pm25_mean": obs.groupby("month")["pm25"].mean(),
        "precip_total_mm": full.groupby("month")["precip"].sum() / full["station_id"].nunique(),
        "wind_speed_mean": full.groupby("month")["wind_speed"].mean(),
        "rh_mean": full.groupby("month")["rh"].mean()}).round(2)
    mon.to_csv(out / "g1_42_monthly.csv")
    # Hai đại lượng khác thang đo -> hai biểu đồ chung trục thời gian (không dùng trục kép)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    x = np.arange(len(mon))
    axes[0].plot(x, mon["pm25_mean"], color=BLUE, lw=2, marker="o", ms=6)
    style(axes[0], "PM2.5 trung bình theo tháng (6 trạm)", "", "µg/m³")
    axes[1].bar(x, mon["precip_total_mm"], color=BLUE, width=0.6)
    style(axes[1], "Tổng lượng mưa theo tháng (Open-Meteo, TB 6 trạm)", "", "mm")
    axes[1].set_xticks(x, [str(p) for p in mon.index], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(fig_dir / "g1_42_monthly.png", dpi=130)
    plt.close(fig)
    return sp, lead, mon


# ------------------------------------------------------------------ G1-43
def g1_43(obs, out, fig_dir):
    # "Có mưa" = có ít nhất 1 giờ mưa >= 0,1 mm trong 6 giờ qua; "không mưa" = 6 giờ qua khô hoàn toàn
    o = obs.dropna(subset=["precip_6h"]).copy()
    o["rain_state"] = np.where(o["precip_6h"] >= 0.1, "Có mưa (6h qua)", "Không mưa (6h qua)")
    tab = (o.groupby(["station_id", "rain_state"])["pm25"].apply(summary).unstack().round(2))
    allst = o.groupby("rain_state")["pm25"].apply(summary).unstack().round(2)
    allst.index = pd.MultiIndex.from_product([["ALL"], allst.index])
    tab = pd.concat([tab, allst])
    med = tab["median"].unstack()
    tab["giam_trung_vi_pct"] = np.nan
    for sid in med.index:
        d, w = med.loc[sid, "Không mưa (6h qua)"], med.loc[sid, "Có mưa (6h qua)"]
        tab.loc[(sid, "Có mưa (6h qua)"), "giam_trung_vi_pct"] = round((d - w) / d * 100, 1)
    tab.to_csv(out / "g1_43_rain_vs_dry.csv")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sids = sorted(o["station_id"].unique()) + ["ALL"]
    for k, (state, color) in enumerate([("Không mưa (6h qua)", ORANGE), ("Có mưa (6h qua)", BLUE)]):
        data = [o.loc[(o["station_id"] == s) | (s == "ALL"), :].query("rain_state == @state")["pm25"] for s in sids]
        pos = np.arange(len(sids)) * 3 + k * 1.1
        bp = ax.boxplot(data, positions=pos, widths=0.9, patch_artist=True, showfliers=False,
                        medianprops={"color": "white", "lw": 2}, whiskerprops={"color": color},
                        capprops={"color": color})
        for b in bp["boxes"]:
            b.set(facecolor=color, edgecolor=color)
        bp["boxes"][0].set_label(state)
    ax.set_xticks(np.arange(len(sids)) * 3 + 0.55, [STATION_LABEL.get(s, "6 trạm gộp") for s in sids],
                  rotation=20, ha="right", fontsize=8)
    style(ax, "G1-43. PM2.5 khi có mưa và không mưa trong 6 giờ qua (hộp = tứ phân vị, vạch trắng = trung vị)",
          "", "PM2.5 (µg/m³)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / "g1_43_rain_vs_dry.png", dpi=130)
    plt.close(fig)

    ws = binned(obs, "wind_speed_6h", [0, 1, 1.5, 2, 2.5, 3, 4, 99], ["<1", "1-1.5", "1.5-2", "2-2.5", "2.5-3", "3-4", "≥4"])
    rain = binned(obs, "precip_24h", [0, 0.1, 1, 5, 10, 20, 999], ["0", "0.1-1", "1-5", "5-10", "10-20", "≥20"])
    ws.to_csv(out / "g1_43_pm25_by_wind_speed.csv")
    rain.to_csv(out / "g1_43_pm25_by_rain.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    line_with_band(axes[0], ws, "Tốc độ gió trung bình 6h (m/s)", "PM2.5 theo tốc độ gió")
    line_with_band(axes[1], rain, "Tổng mưa 24h qua (mm)", "PM2.5 theo lượng mưa 24h qua")
    fig.tight_layout()
    fig.savefig(fig_dir / "g1_43_wind_rain.png", dpi=130)
    plt.close(fig)
    return tab, ws, rain


# ------------------------------------------------------------------ G1-44
def g1_44(obs, out, fig_dir):
    """Hoa gió theo trạm: độ dài cánh = % số giờ gió thổi từ hướng đó; màu = trung vị PM2.5 khi gió từ hướng đó."""
    o = obs.copy()
    calm = o["wind_speed"] < 0.5
    o["sector16"] = (((o["wind_dir"] + 11.25) % 360) // 22.5).astype(int)
    o["sector8"] = pd.Categorical(np.array(SECT8)[(((o["wind_dir"] + 22.5) % 360) // 45).astype(int)], SECT8)
    tab = (o[~calm].groupby(["station_id", "sector8"], observed=False)["pm25"]
             .agg(n="size", median="median", mean="mean", pct_high=lambda s: (s > 35.4).mean() * 100))
    tab["share_hours_pct"] = tab["n"] / tab.groupby(level=0)["n"].transform("sum") * 100
    tab.round(2).to_csv(out / "g1_44_wind_sector_by_station.csv")

    bounds = [0, 14, 16, 18, 20, 22, 25, 35]
    cmap, norm = ListedColormap(SEQ_BLUE), BoundaryNorm(bounds, len(SEQ_BLUE))
    theta = np.deg2rad(np.arange(16) * 22.5)
    sids = sorted(o["station_id"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(14, 10), subplot_kw={"projection": "polar"})
    fig.subplots_adjust(wspace=0.5, hspace=0.45, right=0.84)
    for ax, sid in zip(axes.ravel(), sids):
        g = o[(o["station_id"] == sid) & ~calm]
        st = g.groupby("sector16")["pm25"].agg(["size", "median"]).reindex(range(16))
        share = st["size"].fillna(0) / st["size"].sum() * 100
        ax.bar(theta, share, width=np.deg2rad(22.5) * 0.92, color=cmap(norm(st["median"].fillna(0))),
               edgecolor="white", linewidth=1)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]), ["B", "ĐB", "Đ", "ĐN", "N", "TN", "T", "TB"],
                      fontsize=8, color=MUTED)
        ax.tick_params(axis="y", labelsize=7, colors=MUTED)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_rlabel_position(22.5)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.grid(color=GRID)
        ax.set_title(f"{STATION_LABEL[sid]}\n(n = {len(g)} giờ, lặng gió {calm[o['station_id'] == sid].mean() * 100:.1f}%)",
                     fontsize=9, color=INK, pad=12)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.88, 0.25, 0.015, 0.5])
    cb = fig.colorbar(sm, cax=cax, ticks=bounds[:-1])
    cb.ax.set_yticklabels([f"{b}" for b in bounds[:-2]] + [f"≥{bounds[-2]}"])
    cb.set_label("Trung vị PM2.5 (µg/m³) khi gió thổi từ hướng đó", color=MUTED)
    fig.suptitle("G1-44. Hoa gió theo trạm: độ dài cánh = % số giờ gió thổi TỪ hướng đó; màu = trung vị PM2.5",
                 fontsize=11, x=0.02, ha="left")
    fig.savefig(fig_dir / "g1_44_wind_rose.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return tab


def main() -> None:
    cfg = load_om_config()
    out = rpath(cfg, "report_dir")
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    full = pd.read_parquet(rpath(cfg, "merged_hcm")).sort_values(["station_id", "time"]).reset_index(drop=True)
    obs = full[(full["is_imputed"] == 0) & full["pm25"].notna()].copy()
    print(f"Số giờ có PM2.5 đo thật: {len(obs)}")

    sp, lead, mon = g1_42(full, obs, out, fig_dir)
    rd, ws, rain = g1_43(obs, out, fig_dir)
    sec = g1_44(obs, out, fig_dir)
    pd.set_option("display.width", 200)
    print("\n## G1-42 Spearman\n", sp.to_string())
    print("\n## G1-42 Dẫn trước\n", lead.to_string())
    print("\n## G1-43 Mưa / không mưa\n", rd.to_string())
    print("\n## G1-43 Theo gió\n", ws.to_string(), "\n\n## G1-43 Theo mưa 24h\n", rain.to_string())
    s8 = sec["median"].unstack().round(1)
    print("\n## G1-44 Trung vị PM2.5 theo hướng gió (8 hướng) × trạm\n", s8.to_string())
    print("\n## G1-44 % số giờ theo hướng\n", sec["share_hours_pct"].unstack().round(1).to_string())


if __name__ == "__main__":
    main()
