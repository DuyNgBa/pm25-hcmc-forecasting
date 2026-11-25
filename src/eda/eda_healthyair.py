from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLEAN_PATH = Path("data/interim/healthyair_clean.parquet")
OUT_DIR = Path("reports/eda")
THRESHOLD = 35.4

STATION_TYPE = {
    "hcm_01": "Nền đô thị",
    "hcm_02": "Giao thông",
    "hcm_03": "Công nghiệp",
    "hcm_04": "Dân cư",
    "hcm_05": "Giao thông",
    "hcm_06": "Giao thông + dân cư",
}
DRY_MONTHS = [12, 1, 2, 3, 4]
WEEKDAYS = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
LOCKDOWN_START = pd.Timestamp("2021-07-09")
LOCKDOWN_END = pd.Timestamp("2021-10-01")


class Logger:
    def __init__(self):
        self.lines = ["# EDA PM2.5 HealthyAir", "",
                      "Chỉ dùng giá trị đo thật (`is_imputed == 0`, `pm25` không NaN).", ""]

    def __call__(self, text=""):
        print(text)
        self.lines.append(str(text))

    def table(self, df, index=True):
        self(df.to_markdown(index=index))
        self()


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figures" / name, dpi=130)
    plt.close(fig)


def load():
    df = pd.read_parquet(CLEAN_PATH)
    df = df[(df["is_imputed"] == 0) & df["pm25"].notna()].copy()
    df["hour"] = df["time"].dt.hour
    df["dow"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    df["season"] = np.where(df["month"].isin(DRY_MONTHS), "Mùa khô", "Mùa mưa")
    df["type"] = df["station_id"].map(STATION_TYPE)
    return df


def g1_20_distribution(df, log):
    log("## G1-20. Phân phối PM2.5 theo trạm")
    stats = df.groupby("station_id")["pm25"].agg(
        so_gio="count", tb="mean", trung_vi="median", do_lech="std",
        p95=lambda s: s.quantile(0.95), lon_nhat="max",
        pct_vuot=lambda s: (s > THRESHOLD).mean() * 100).round(1)
    stats = stats.rename(columns={"pct_vuot": f"pct_gio_vuot_{THRESHOLD}"})
    stats.insert(0, "loai_tram", stats.index.map(STATION_TYPE))
    stats.to_csv(OUT_DIR / "g1_20_distribution.csv")
    log.table(stats)

    stations = sorted(df["station_id"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    for ax, sid in zip(axes.flat, stations):
        s = df.loc[df["station_id"] == sid, "pm25"]
        ax.hist(s.clip(upper=120), bins=60, color="#2e75b6")
        ax.axvline(THRESHOLD, color="red", ls="--", lw=1)
        ax.set_title(f"{sid} ({STATION_TYPE[sid]})", fontsize=9)
    fig.suptitle(f"Phân phối PM2.5 theo trạm (cắt hiển thị ở 120; vạch đỏ = {THRESHOLD} µg/m³)")
    save(fig, "g1_20_histogram.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot([df.loc[df["station_id"] == s, "pm25"] for s in stations], tick_labels=stations, showfliers=False)
    ax.axhline(THRESHOLD, color="red", ls="--", lw=1, label=f"{THRESHOLD} µg/m³")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("Boxplot PM2.5 theo trạm (không vẽ outlier)")
    ax.legend()
    save(fig, "g1_20_boxplot.png")


def g1_21_cycles(df, log):
    log("## G1-21. Chu kỳ ngày và chu kỳ tuần")
    hourly = df.pivot_table(index="hour", columns="station_id", values="pm25", aggfunc="mean").round(1)
    weekly = df.pivot_table(index="dow", columns="station_id", values="pm25", aggfunc="mean").round(1)
    weekly.index = WEEKDAYS
    hourly.to_csv(OUT_DIR / "g1_21_hourly.csv")
    weekly.to_csv(OUT_DIR / "g1_21_weekly.csv")
    peak = pd.DataFrame({"gio_cao_nhat": hourly.idxmax(), "gia_tri_cao_nhat": hourly.max(),
                         "gio_thap_nhat": hourly.idxmin(), "gia_tri_thap_nhat": hourly.min()})
    log("Giờ cao nhất / thấp nhất theo trạm:")
    log.table(peak)
    wk = df.assign(cuoi_tuan=df["dow"] >= 5).pivot_table(index="station_id", columns="cuoi_tuan", values="pm25", aggfunc="mean").round(1)
    wk.columns = ["ngay_thuong", "cuoi_tuan"]
    wk["chenh_lech_pct"] = ((wk["cuoi_tuan"] / wk["ngay_thuong"] - 1) * 100).round(1)
    log("Ngày thường so với cuối tuần:")
    log.table(wk)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for sid in hourly.columns:
        ax[0].plot(hourly.index, hourly[sid], marker="o", ms=3, label=sid)
        ax[1].plot(range(7), weekly[sid], marker="o", ms=3, label=sid)
    ax[0].set_xticks(range(0, 24, 2))
    ax[0].set_xlabel("Giờ trong ngày")
    ax[0].set_ylabel("PM2.5 trung bình (µg/m³)")
    ax[0].set_title("Chu kỳ ngày")
    ax[1].set_xticks(range(7))
    ax[1].set_xticklabels(WEEKDAYS)
    ax[1].set_title("Chu kỳ tuần")
    ax[0].legend(fontsize=7)
    save(fig, "g1_21_cycles.png")


def g1_22_annual(df, log):
    log("## G1-22. Chu kỳ năm: theo tháng, mùa khô và mùa mưa")
    monthly = df.pivot_table(index="month", columns="station_id", values="pm25", aggfunc="mean").round(1)
    monthly.to_csv(OUT_DIR / "g1_22_monthly.csv")
    log("PM2.5 trung bình theo tháng (gộp các năm):")
    log.table(monthly)
    season = df.pivot_table(index="station_id", columns="season", values="pm25", aggfunc="mean").round(1)
    season["kho_chia_mua"] = (season["Mùa khô"] / season["Mùa mưa"]).round(2)
    season.to_csv(OUT_DIR / "g1_22_season.csv")
    log("Mùa khô (tháng 12–4) so với mùa mưa (tháng 5–11):")
    log.table(season)
    coverage = df.groupby(df["time"].dt.to_period("M"))["pm25"].count()
    log(f"Lưu ý: dữ liệu từ {df['time'].min():%m/%Y} đến {df['time'].max():%m/%Y}, nên các tháng 3–6 có 2 năm, các tháng 7–11 chỉ có năm 2021 (trùng giai đoạn giãn cách).")
    log()

    fig, ax = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [2, 1]})
    for sid in monthly.columns:
        ax[0].plot(monthly.index, monthly[sid], marker="o", ms=3, label=sid)
    ax[0].axvspan(4.5, 11.5, color="#d9e8f5", alpha=0.6, label="Mùa mưa")
    ax[0].set_xticks(range(1, 13))
    ax[0].set_xlabel("Tháng")
    ax[0].set_ylabel("PM2.5 trung bình (µg/m³)")
    ax[0].set_title("PM2.5 theo tháng")
    ax[0].legend(fontsize=7)
    x = np.arange(len(season.index))
    ax[1].bar(x - 0.2, season["Mùa khô"], 0.4, label="Mùa khô", color="#d98c3f")
    ax[1].bar(x + 0.2, season["Mùa mưa"], 0.4, label="Mùa mưa", color="#2e75b6")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(season.index, rotation=45, fontsize=8)
    ax[1].set_title("Mùa khô và mùa mưa")
    ax[1].legend(fontsize=7)
    save(fig, "g1_22_annual.png")

    fig, ax = plt.subplots(figsize=(13, 3.5))
    daily = df.groupby([df["time"].dt.floor("D"), "station_id"])["pm25"].mean().unstack()
    for sid in daily.columns:
        ax.plot(daily.index, daily[sid].rolling(7, min_periods=3).mean(), lw=1, label=sid)
    ax.axvspan(LOCKDOWN_START, LOCKDOWN_END, color="#eeeeee", label="Giãn cách")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("PM2.5 trung bình ngày, trượt 7 ngày")
    ax.legend(fontsize=7, ncol=4)
    save(fig, "g1_22_timeseries.png")
    return coverage


def acf(s, max_lag):
    return [s.autocorr(k) for k in range(1, max_lag + 1)]


def g1_23_correlation(df, log):
    log("## G1-23. Tương quan giữa các trạm và tự tương quan")
    wide = df.pivot(index="time", columns="station_id", values="pm25")
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="h")
    wide = wide.reindex(full)
    corr = wide.corr(min_periods=500).round(2)
    corr.to_csv(OUT_DIR / "g1_23_station_corr.csv")
    log("Tương quan PM2.5 giữa các trạm (Pearson, theo từng cặp giờ cùng có dữ liệu):")
    log.table(corr)
    upper = corr.values[np.triu_indices_from(corr.values, 1)]
    log(f"- Trung bình các cặp: {np.nanmean(upper):.2f}; thấp nhất: {np.nanmin(upper):.2f}; cao nhất: {np.nanmax(upper):.2f}")
    log()

    ac = pd.DataFrame({sid: acf(wide[sid], 72) for sid in wide.columns}, index=range(1, 73))
    ac.to_csv(OUT_DIR / "g1_23_acf.csv")
    log("Tự tương quan (ACF) theo trạm:")
    log.table(ac.loc[[1, 3, 6, 12, 24, 48]].round(2))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    im = ax[0].imshow(corr.values, vmin=0, vmax=1, cmap="Blues")
    ax[0].set_xticks(range(len(corr)))
    ax[0].set_yticks(range(len(corr)))
    ax[0].set_xticklabels(corr.columns, rotation=45, fontsize=8)
    ax[0].set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax[0].text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                       color="white" if corr.values[i, j] > 0.6 else "black")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[0].set_title("Tương quan PM2.5 giữa các trạm")
    for sid in ac.columns:
        ax[1].plot(ac.index, ac[sid], lw=1, label=sid)
    for lag in [1, 24, 48]:
        ax[1].axvline(lag, color="grey", ls=":", lw=0.8)
    ax[1].set_xlabel("Độ trễ (giờ)")
    ax[1].set_ylabel("ACF")
    ax[1].set_title("Tự tương quan PM2.5 (lag 1–72 giờ)")
    ax[1].legend(fontsize=7)
    save(fig, "g1_23_correlation_acf.png")


def g1_24_station_type(df, log):
    log("## G1-24. So sánh theo loại khu vực")
    by_type = df.groupby("type")["pm25"].agg(
        so_tram=lambda s: df.loc[s.index, "station_id"].nunique(), so_gio="count", tb="mean", trung_vi="median",
        p95=lambda s: s.quantile(0.95), pct_vuot=lambda s: (s > THRESHOLD).mean() * 100).round(1)
    by_type.to_csv(OUT_DIR / "g1_24_by_type.csv")
    log.table(by_type)
    hourly = df.pivot_table(index="hour", columns="type", values="pm25", aggfunc="mean")
    amp = (hourly.max() - hourly.min()).round(1)
    log("Biên độ chu kỳ ngày (cao nhất − thấp nhất trong ngày) theo loại khu vực:")
    log.table(amp.rename("bien_do").to_frame())
    log("Lưu ý: mỗi loại chỉ có 1–2 trạm, nên khác biệt giữa các loại cũng chính là khác biệt giữa từng trạm cụ thể.")
    log()

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    types = list(by_type.index)
    ax[0].boxplot([df.loc[df["type"] == t, "pm25"] for t in types], tick_labels=types, showfliers=False)
    ax[0].tick_params(axis="x", labelsize=8, rotation=20)
    ax[0].set_ylabel("PM2.5 (µg/m³)")
    ax[0].set_title("PM2.5 theo loại khu vực")
    for t in hourly.columns:
        ax[1].plot(hourly.index, hourly[t], marker="o", ms=3, label=t)
    ax[1].set_xticks(range(0, 24, 2))
    ax[1].set_xlabel("Giờ trong ngày")
    ax[1].set_title("Chu kỳ ngày theo loại khu vực")
    ax[1].legend(fontsize=7)
    save(fig, "g1_24_station_type.png")


def g1_25_lockdown(df, log):
    log("## G1-25. Ảnh hưởng của giai đoạn giãn cách")
    periods = {
        "truoc (09/06–08/07/2021)": (pd.Timestamp("2021-06-09"), LOCKDOWN_START),
        "gian_cach (09/07–30/09/2021)": (LOCKDOWN_START, LOCKDOWN_END),
        "sau (01/10–31/10/2021)": (LOCKDOWN_END, pd.Timestamp("2021-11-01")),
    }
    rows = []
    for name, (a, b) in periods.items():
        sub = df[(df["time"] >= a) & (df["time"] < b)]
        g = sub.groupby("station_id")["pm25"]
        rows.append(pd.DataFrame({"giai_doan": name, "so_gio": g.count(), "tb": g.mean().round(1)}))
    comp = pd.concat(rows).reset_index().pivot(index="station_id", columns="giai_doan", values=["tb", "so_gio"])
    comp.columns = [f"{a}_{b}" for a, b in comp.columns]
    comp.to_csv(OUT_DIR / "g1_25_lockdown.csv")
    log("PM2.5 trung bình trước / trong / sau giãn cách (cùng mùa mưa để hạn chế ảnh hưởng của mùa):")
    log.table(comp)
    log("Trạm có dưới 200 giờ dữ liệu trong một giai đoạn thì kết quả giai đoạn đó không đáng tin.")
    log()

    in_ld = df[(df["time"] >= LOCKDOWN_START) & (df["time"] < LOCKDOWN_END)]
    rain_non_ld = df[(df["season"] == "Mùa mưa") & ~((df["time"] >= LOCKDOWN_START) & (df["time"] < LOCKDOWN_END))]
    h_ld = in_ld.groupby("hour")["pm25"].mean()
    h_non = rain_non_ld.groupby("hour")["pm25"].mean()
    log("Chu kỳ ngày (gộp các trạm): giãn cách so với các tháng mùa mưa khác:")
    log(f"- Giãn cách: cao nhất {h_ld.max():.1f} lúc {h_ld.idxmax()}h, thấp nhất {h_ld.min():.1f}, biên độ {h_ld.max() - h_ld.min():.1f}")
    log(f"- Mùa mưa không giãn cách: cao nhất {h_non.max():.1f} lúc {h_non.idxmax()}h, thấp nhất {h_non.min():.1f}, biên độ {h_non.max() - h_non.min():.1f}")
    log()

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(comp.index))
    tb_cols = [c for c in comp.columns if c.startswith("tb_")]
    for k, c in enumerate(tb_cols):
        ax[0].bar(x + (k - 1) * 0.27, comp[c], 0.27, label=c.replace("tb_", ""))
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(comp.index, rotation=45, fontsize=8)
    ax[0].set_ylabel("PM2.5 trung bình (µg/m³)")
    ax[0].set_title("Trước / trong / sau giãn cách")
    ax[0].legend(fontsize=7)
    ax[1].plot(h_ld.index, h_ld, marker="o", ms=3, label="Giãn cách")
    ax[1].plot(h_non.index, h_non, marker="o", ms=3, label="Mùa mưa, không giãn cách")
    ax[1].set_xticks(range(0, 24, 2))
    ax[1].set_xlabel("Giờ trong ngày")
    ax[1].set_title("Chu kỳ ngày: giãn cách và bình thường")
    ax[1].legend(fontsize=7)
    save(fig, "g1_25_lockdown.png")


def main():
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    log = Logger()
    df = load()
    log(f"Số giờ dùng cho EDA: {len(df)} ({df['station_id'].nunique()} trạm)")
    log()
    g1_20_distribution(df, log)
    g1_21_cycles(df, log)
    g1_22_annual(df, log)
    g1_23_correlation(df, log)
    g1_24_station_type(df, log)
    g1_25_lockdown(df, log)
    (OUT_DIR / "eda_log.md").write_text("\n".join(log.lines), encoding="utf-8")
    log(f"Đã lưu kết quả vào {OUT_DIR}")


if __name__ == "__main__":
    main()
