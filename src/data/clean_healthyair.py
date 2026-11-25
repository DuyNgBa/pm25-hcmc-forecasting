import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RENAME = {
    "TSP": "tsp",
    "PM2.5": "pm25",
    "O3": "o3",
    "CO": "co",
    "NO2": "no2",
    "SO2": "so2",
    "Temperature": "temp_sensor",
    "Humidity": "rh_sensor",
}
VARS = list(RENAME.values())
POLLUTANTS = ["tsp", "pm25", "o3", "co", "no2", "so2"]
KEY = ["city", "station_id", "time"]
SO2_QCVN_1H = 350
SO2_QCVN_24H = 125


class Logger:
    def __init__(self):
        self.lines = ["# Log làm sạch dữ liệu HealthyAir", ""]

    def __call__(self, text=""):
        print(text)
        self.lines.append(str(text))

    def table(self, df, index=True):
        self(df.to_markdown(index=index) if len(df) else "Không có")
        self()


def load_config(path="configs/cleaning.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def runs(mask):
    out = []
    i, n = 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def missing_table(df, cols=VARS):
    return df.groupby("station_id")[cols].apply(lambda g: g.isna().mean() * 100).round(1)


def load_raw(path, log):
    raw = pd.read_csv(path)
    log("## G1-01. Đọc dữ liệu thô")
    log(f"- Số dòng trong file: {len(raw)}")
    log(f"- Dòng không có thời gian (bị bỏ): {raw['date'].isna().sum()}")
    df = raw.dropna(subset=["date"]).copy()
    df["time"] = pd.to_datetime(df["date"], format="%d-%m-%Y %H:%M")
    df["station_id"] = "hcm_" + df["Station_No"].astype(int).astype(str).str.zfill(2)
    df["city"] = "hcm"
    df = df.rename(columns=RENAME)[KEY + VARS]
    log("- Kiểu dữ liệu: " + ", ".join(f"{c}={df[c].dtype}" for c in df.columns))
    log(f"- Khoảng thời gian: {df['time'].min()} → {df['time'].max()}")
    log(f"- Số trạm: {df['station_id'].nunique()}")
    dup = df.duplicated(KEY).sum()
    log(f"- Dòng trùng khoá (station_id, time): {dup}" + (" → giữ dòng đầu tiên" if dup else " → không cần xử lý"))
    df = df.drop_duplicates(KEY)
    log(f"- Thời điểm không tròn giờ: {(df['time'].dt.minute != 0).sum()}")
    log(f"- Giá trị âm: {(df[VARS] < 0).sum().sum()}")
    log("- Đơn vị: µg/m³ cho các chất ô nhiễm, °C cho nhiệt độ, % cho độ ẩm")
    log()
    return df


def make_grid(df, log):
    start, end = df["time"].min(), df["time"].max()
    hours = pd.date_range(start, end, freq="h")
    stations = sorted(df["station_id"].unique())
    grid = pd.MultiIndex.from_product([stations, hours], names=["station_id", "time"]).to_frame(index=False)
    grid["city"] = "hcm"
    out = grid.merge(df, on=KEY, how="left")[KEY + VARS].sort_values(KEY).reset_index(drop=True)
    log("## G1-02. Tạo lưới giờ đầy đủ")
    log(f"- Lưới: {start} → {end}, {len(hours)} giờ × {len(stations)} trạm = {len(out)} dòng")
    log(f"- Số dòng có trong file gốc: {len(df)}; số giờ được thêm (toàn NaN): {len(out) - len(df)}")
    log()
    return out


def find_frozen(df, col, min_hours):
    rows = []
    for sid, g in df.groupby("station_id"):
        v = g[col].values
        t = g["time"].values
        i, n = 0, len(v)
        while i < n:
            if np.isnan(v[i]):
                i += 1
                continue
            j = i
            while j + 1 < n and v[j + 1] == v[i]:
                j += 1
            if j - i + 1 >= min_hours:
                rows.append({"station_id": sid, "variable": col, "value": v[i],
                             "start": pd.Timestamp(t[i]), "end": pd.Timestamp(t[j]), "hours": j - i + 1})
            i = j + 1
    return pd.DataFrame(rows, columns=["station_id", "variable", "value", "start", "end", "hours"])


def find_gaps(df, col, min_hours):
    rows = []
    for sid, g in df.groupby("station_id"):
        t = g["time"].values
        for i, j in runs(g[col].isna().values):
            if j - i + 1 >= min_hours:
                rows.append({"station_id": sid, "start": pd.Timestamp(t[i]), "end": pd.Timestamp(t[j]),
                             "hours": j - i + 1, "days": round((j - i + 1) / 24, 1)})
    return pd.DataFrame(rows, columns=["station_id", "start", "end", "hours", "days"])


def find_spikes(df, threshold):
    wide = df.pivot(index="time", columns="station_id", values="pm25")
    rows = []
    for sid in wide.columns:
        s = wide[sid]
        diff = s.diff()
        for t in diff.index[diff.abs() > threshold]:
            i = s.index.get_loc(t)
            prev, cur = s.iloc[i - 1], s.iloc[i]
            nxt = s.iloc[i + 1] if i + 1 < len(s) else np.nan
            others = wide.drop(columns=sid).loc[t] - wide.drop(columns=sid).shift(1).loc[t]
            n_others_up = int((others > 30).sum()) if diff[t] > 0 else int((others < -30).sum())
            if diff[t] > 0:
                single = not np.isnan(nxt) and abs(nxt - prev) < 0.5 * abs(cur - prev)
            else:
                single = not np.isnan(nxt) and abs(nxt - cur) > 0.5 * abs(prev - cur) and abs(nxt - prev) < 0.5 * abs(prev - cur)
            if n_others_up >= 2:
                goi_y, ly_do = "giữ", f"{n_others_up} trạm khác cùng biến động mạnh → sự kiện vùng"
            elif single:
                goi_y, ly_do = "loại", "nhảy một điểm rồi quay lại ngay, các trạm khác không đổi → nghi lỗi đo"
            else:
                goi_y, ly_do = "giữ", "mức cao duy trì sau khi nhảy → có thể là sự kiện cục bộ"
            if np.isnan(nxt):
                goi_y, ly_do = "cần xem", "giờ sau bị thiếu dữ liệu, không đủ căn cứ tự động"
            rows.append({"id": len(rows) + 1, "station_id": sid, "time": t, "truoc": round(prev, 1),
                         "tai_diem": round(cur, 1), "sau": round(nxt, 1) if not np.isnan(nxt) else np.nan,
                         "chenh_lech": round(diff[t], 1), "so_tram_khac_cung_bien_dong": n_others_up,
                         "goi_y": goi_y, "ly_do": ly_do, "quyet_dinh": goi_y})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for i, r in out.iterrows():
        parent = out[(out["station_id"] == r["station_id"]) & (out["time"] == r["time"] - pd.Timedelta(hours=1)) & (out["chenh_lech"] > 0)]
        if r["chenh_lech"] < 0 and len(parent):
            pid = int(parent["id"].iloc[0])
            out.loc[i, ["goi_y", "ly_do", "quyet_dinh"]] = [f"theo #{pid}", f"nhịp quay về của điểm #{pid}, không xử lý riêng", f"theo #{pid}"]
    return out


def plot_spikes(df, spikes, path):
    if spikes.empty:
        return
    wide = df.pivot(index="time", columns="station_id", values="pm25")
    n = len(spikes)
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.4), squeeze=False)
    for ax, (_, r) in zip(axes.flat, spikes.iterrows()):
        win = wide.loc[r["time"] - pd.Timedelta(hours=24): r["time"] + pd.Timedelta(hours=24)]
        for sid in win.columns:
            if sid != r["station_id"]:
                ax.plot(win.index, win[sid], color="#bbbbbb", lw=0.7)
        ax.plot(win.index, win[r["station_id"]], color="#c0392b", lw=1.3)
        ax.axvline(r["time"], color="black", ls=":", lw=0.8)
        ax.set_title(f"#{r['id']} {r['station_id']} {r['time']:%d/%m/%y %H}h\ngợi ý: {r['goi_y']}", fontsize=7)
        ax.tick_params(labelsize=5)
        ax.set_xticks([])
    for ax in axes.flat[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def interpolate_short(df, cols, max_hours):
    df = df.copy()
    filled = {}
    for c in cols:
        flag = np.zeros(len(df), dtype=bool)
        for sid, idx in df.groupby("station_id").groups.items():
            s = df.loc[idx, c]
            interp = s.interpolate(method="linear", limit_area="inside")
            keep = np.zeros(len(s), dtype=bool)
            for i, j in runs(s.isna().values):
                if j - i + 1 <= max_hours and i > 0 and j < len(s) - 1:
                    keep[i:j + 1] = True
            vals = s.values.copy()
            vals[keep] = interp.values[keep]
            df.loc[idx, c] = vals
            flag[df.index.get_indexer(idx)] = keep
        filled[c] = flag
    return df, filled


def plot_report_figures(raw_grid, df, report_dir):
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    s4 = raw_grid[raw_grid["station_id"] == "hcm_04"].set_index("time")
    fig, ax = plt.subplots(2, 1, figsize=(11, 4.5), sharex=True)
    ax[0].plot(s4.index, s4["temp_sensor"], lw=0.6)
    ax[0].set_ylabel("Nhiệt độ (°C)")
    ax[1].plot(s4.index, s4["rh_sensor"], lw=0.6, color="tab:orange")
    ax[1].set_ylabel("Độ ẩm (%)")
    ax[0].set_title("Trạm 4: nhiệt độ và độ ẩm đứng yên ở 20,4°C / 12,5% (dữ liệu gốc)")
    plt.tight_layout()
    plt.savefig(fig_dir / "issue1_station4_frozen.png", dpi=130)
    plt.close()

    s3 = raw_grid[raw_grid["station_id"] == "hcm_03"].set_index("time")
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(s3.index, s3["tsp"], lw=0.6, label="TSP")
    ax.plot(s3.index, s3["pm25"], lw=0.6, label="PM2.5", alpha=0.8)
    ax.set_title("Trạm 3: TSP bằng 0 từ 04/2021, PM2.5 > TSP (dữ liệu gốc)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "issue2_station3_tsp.png", dpi=130)
    plt.close()

    miss = raw_grid.assign(day=raw_grid["time"].dt.floor("D")).groupby(["station_id", "day"])["pm25"].apply(lambda s: s.notna().mean()).unstack()
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.imshow(miss.values, aspect="auto", cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(miss.index)))
    ax.set_yticklabels(miss.index)
    ticks = np.linspace(0, miss.shape[1] - 1, 8).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([miss.columns[i].strftime("%m/%Y") for i in ticks])
    ax.set_title("Tỉ lệ giờ có PM2.5 theo ngày (trắng = mất dữ liệu)")
    plt.tight_layout()
    plt.savefig(fig_dir / "issue3_pm25_availability.png", dpi=130)
    plt.close()

    s5 = raw_grid[raw_grid["station_id"] == "hcm_05"]["no2"].dropna()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(s5, bins=120)
    ax.set_title("Trạm 5: phân bố NO2, dồn cục ở 270–283 µg/m³ (bão hoà)")
    ax.set_xlabel("µg/m³")
    plt.tight_layout()
    plt.savefig(fig_dir / "issue5_station5_no2.png", dpi=130)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 3.2))
    data = [raw_grid.loc[raw_grid["station_id"] == s, "so2"].dropna() for s in sorted(raw_grid["station_id"].unique())]
    ax.boxplot(data, tick_labels=sorted(raw_grid["station_id"].unique()), showfliers=False)
    ax.axhline(SO2_QCVN_24H, color="red", ls="--", lw=1, label="QCVN 05:2023 TB 24 giờ (125)")
    ax.axhline(SO2_QCVN_1H, color="darkred", ls=":", lw=1, label="QCVN 05:2023 TB 1 giờ (350)")
    ax.set_ylabel("SO2 (µg/m³)")
    ax.set_title("SO2 theo trạm")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_dir / "issue7_so2.png", dpi=130)
    plt.close()


def write_dictionary(path, cfg):
    rows = [
        ("city", "string", "", "Thành phố, luôn là `hcm`", "Thêm mới"),
        ("station_id", "string", "", "Mã trạm `hcm_01`…`hcm_06` (từ `Station_No`)", "Đổi tên"),
        ("time", "datetime64", "giờ địa phương (UTC+7)", "Mốc đầu giờ; giá trị là trung bình 01:00–01:59 cho dòng 01:00", "Lưới giờ đầy đủ; giờ thiếu = NaN"),
        ("pm25", "float", "µg/m³", "PM2.5, biến mục tiêu", "0 → NaN; điểm nhảy bị đánh dấu loại → NaN; KHÔNG cắt outlier; nội suy khoảng trống ≤ " + str(cfg["interp_max_hours"]) + " giờ"),
        ("tsp", "float", "µg/m³", "Tổng bụi lơ lửng", "≤ 0 → NaN; cắt outlier; nội suy khoảng ngắn. Không dùng làm đặc trưng chính"),
        ("o3", "float", "µg/m³", "Ozone (gốc ppb × 1,963)", "0 → NaN; cắt outlier; nội suy khoảng ngắn"),
        ("co", "float", "µg/m³", "CO (gốc ppm × 1146)", "0 → NaN; cắt outlier; nội suy khoảng ngắn"),
        ("no2", "float", "µg/m³", "NO2 (gốc ppm × 1882)", "0 → NaN; trạm 5 trong khoảng bão hoà → NaN; cắt outlier; nội suy khoảng ngắn"),
        ("so2", "float", "µg/m³", "SO2 (gốc ppm × 2620)", "Cắt outlier; nội suy khoảng ngắn. Không dùng làm đặc trưng chính (nghi sai lệch cảm biến)"),
        ("temp_sensor", "float", "°C", "Nhiệt độ cảm biến tại trạm", "Trạm 4 đóng băng → NaN; cắt outlier; nội suy khoảng ngắn. Chỉ dùng để kiểm tra khớp giờ với Open-Meteo"),
        ("rh_sensor", "float", "%", "Độ ẩm cảm biến tại trạm", "Như temp_sensor"),
        ("is_lockdown", "int8", "0/1", "1 trong thời gian giãn cách theo Chỉ thị 16 toàn TP.HCM", f"{cfg['lockdown_start']} đến trước {cfg['lockdown_end']}"),
        ("is_imputed", "int8", "0/1", "1 nếu `pm25` được nội suy (không phải giá trị đo thật)", "Không dùng các dòng is_imputed = 1 làm nhãn khi đánh giá"),
    ]
    lines = ["# Data dictionary: healthyair_clean.parquet", "",
             "Khoá: (`city`, `station_id`, `time`). Giá trị thiếu: `NaN`. Sinh bởi `src/data/clean_healthyair.py` với cấu hình `configs/cleaning.json`.", "",
             "| Cột | Kiểu | Đơn vị | Ý nghĩa | Cách xử lý |", "|---|---|---|---|---|"]
    lines += [f"| `{a}` | {b} | {c} | {d} | {e} |" for a, b, c, d, e in rows]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(config_path="configs/cleaning.json"):
    cfg = load_config(config_path)
    report_dir = Path(cfg["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg["out_path"]).parent.mkdir(parents=True, exist_ok=True)
    log = Logger()

    df = make_grid(load_raw(cfg["raw_path"], log), log)
    raw_grid = df.copy()

    before = missing_table(df)
    before.to_csv(report_dir / "g1_03_missing_before.csv")
    log("## G1-03. % thiếu theo biến × trạm (TRƯỚC làm sạch)")
    log.table(before)

    frozen = pd.concat([find_frozen(df, c, cfg["frozen_min_hours"]) for c in VARS], ignore_index=True)
    frozen.to_csv(report_dir / "g1_04_frozen_segments.csv", index=False)
    log(f"## G1-04. Đoạn giá trị không đổi ≥ {cfg['frozen_min_hours']} giờ")
    log.table(frozen.groupby(["station_id", "variable", "value"]).agg(so_doan=("hours", "size"), tong_gio=("hours", "sum")).reset_index(), index=False)

    log("## G1-05. Lỗi #1: nhiệt độ/độ ẩm trạm 4 đóng băng")
    st4 = frozen[(frozen["station_id"] == "hcm_04") & frozen["variable"].isin(["temp_sensor", "rh_sensor"])]
    seg_rows = []
    for var, g in st4.groupby("variable"):
        value = g["value"].mode()[0]
        start, end = g["start"].min(), g["end"].max()
        m = (df["station_id"] == "hcm_04") & df["time"].between(start, end) & (df[var] == value)
        seg_rows.append({"variable": var, "gia_tri": value, "bat_dau": start, "ket_thuc": end, "so_gio_gan_nan": int(m.sum())})
        df.loc[m, var] = np.nan
    seg = pd.DataFrame(seg_rows)
    seg.to_csv(report_dir / "g1_05_station4_frozen.csv", index=False)
    log.table(seg, index=False)

    log("## G1-06. Lỗi #2: TSP ≤ 0")
    gt_before = (df["pm25"] > df["tsp"]).groupby(df["station_id"]).sum()
    bad = (df["tsp"] <= 0).groupby(df["station_id"]).sum()
    df.loc[df["tsp"] <= 0, "tsp"] = np.nan
    gt_after = (df["pm25"] > df["tsp"]).groupby(df["station_id"]).sum()
    t6 = pd.DataFrame({"tsp_le_0_thanh_nan": bad, "pm25_gt_tsp_truoc": gt_before, "pm25_gt_tsp_sau": gt_after})
    t6.to_csv(report_dir / "g1_06_tsp.csv")
    log.table(t6)

    log("## G1-10. Lỗi #5: NO2 bão hoà")
    sat = cfg["no2_saturation"]
    m = (df["station_id"] == sat["station_id"]) & df["no2"].between(sat["low"], sat["high"])
    n_valid = df.loc[df["station_id"] == sat["station_id"], "no2"].notna().sum()
    log(f"- {sat['station_id']}: NO2 trong [{sat['low']}, {sat['high']}] µg/m³ → NaN: {int(m.sum())} giờ / {n_valid} giờ có NO2 ({m.sum() / n_valid * 100:.1f}%)")
    df.loc[m, "no2"] = np.nan
    log()

    log("## G1-11. Lỗi #6a: giá trị 0 → NaN")
    zero_cols = ["pm25", "o3", "co", "no2"]
    zeros = (df[zero_cols] == 0).groupby(df["station_id"]).sum()
    zeros.to_csv(report_dir / "g1_11_zeros.csv")
    for c in zero_cols:
        df.loc[df[c] == 0, c] = np.nan
    log.table(zeros)

    log(f"## G1-13. Lỗi #6c: PM2.5 nhảy > {cfg['spike_threshold']} µg/m³ trong 1 giờ")
    dec_path = report_dir / "g1_13_spike_decisions.csv"
    spikes = find_spikes(raw_grid, cfg["spike_threshold"])
    if dec_path.exists():
        old = pd.read_csv(dec_path, parse_dates=["time"])[["station_id", "time", "quyet_dinh"]]
        spikes = spikes.drop(columns="quyet_dinh").merge(old, on=["station_id", "time"], how="left")
        spikes["quyet_dinh"] = spikes["quyet_dinh"].fillna(spikes["goi_y"])
        log("- Dùng quyết định đã chỉnh trong g1_13_spike_decisions.csv")
    spikes.to_csv(dec_path, index=False)
    plot_spikes(raw_grid, spikes, report_dir / "g1_13_spikes.png")
    drop = spikes[spikes["quyet_dinh"] == "loại"]
    for _, r in drop.iterrows():
        df.loc[(df["station_id"] == r["station_id"]) & (df["time"] == r["time"]), "pm25"] = np.nan
    n_events = int((~spikes["goi_y"].str.startswith("theo")).sum())
    log(f"- Tìm thấy {len(spikes)} lần nhảy, gộp thành {n_events} sự kiện; gợi ý loại {int((spikes['goi_y'] == 'loại').sum())}, giữ {int((spikes['goi_y'] == 'giữ').sum())}, cần xem {int((spikes['goi_y'] == 'cần xem').sum())}; đã loại theo quyết định: {len(drop)}")
    log("- Hình từng điểm: g1_13_spikes.png. Sửa cột `quyet_dinh` (giữ/loại) trong g1_13_spike_decisions.csv rồi chạy lại.")
    log.table(spikes[["id", "station_id", "time", "truoc", "tai_diem", "sau", "so_tram_khac_cung_bien_dong", "goi_y", "quyet_dinh"]], index=False)

    gaps = find_gaps(df, "pm25", cfg["long_gap_hours"])
    gaps.to_csv(report_dir / "g1_07_pm25_long_gaps.csv", index=False)
    log(f"## G1-07. Lỗi #3: khối mất PM2.5 ≥ {cfg['long_gap_hours']} giờ (KHÔNG nội suy)")
    log.table(gaps.groupby("station_id").agg(so_khoi=("hours", "size"), tong_gio=("hours", "sum"), dai_nhat_gio=("hours", "max")))
    log("Các khối từ 7 ngày trở lên:")
    log.table(gaps[gaps["days"] >= 7], index=False)

    log("## G1-12. Lỗi #6b: cắt outlier ở phân vị " + str(cfg["outlier_quantile"] * 100))
    train = df[df["time"] <= pd.Timestamp(cfg["train_end"])]
    clip_cols = [c for c in VARS if c != "pm25"]
    thr = {c: float(train[c].quantile(cfg["outlier_quantile"])) for c in clip_cols + ["pm25"]}
    clipped = {c: int((df[c] > thr[c]).sum()) for c in clip_cols}
    for c in clip_cols:
        df[c] = df[c].clip(upper=thr[c])
    with open(cfg["thresholds_path"], "w", encoding="utf-8") as f:
        json.dump({"train_end": cfg["train_end"], "quantile": cfg["outlier_quantile"],
                   "upper": {k: round(v, 3) for k, v in thr.items()}, "pm25_clipped": False}, f, ensure_ascii=False, indent=2)
    log(f"- Ngưỡng tính trên tập train (đến {cfg['train_end']}), ghi vào {cfg['thresholds_path']}")
    log.table(pd.DataFrame({"nguong_tren": {k: round(v, 1) for k, v in thr.items()},
                            "so_gio_bi_cat": {**clipped, "pm25": f"không cắt ({int((df['pm25'] > thr['pm25']).sum())} giờ vượt ngưỡng)"}}))

    df["is_lockdown"] = ((df["time"] >= pd.Timestamp(cfg["lockdown_start"])) & (df["time"] < pd.Timestamp(cfg["lockdown_end"]))).astype("int8")
    log("## G1-08. Biến is_lockdown")
    log(f"- is_lockdown = 1 từ {cfg['lockdown_start']} đến trước {cfg['lockdown_end']}")
    log("- Nguồn: Công văn 2279/UBND-VX (08/7/2021) và Công văn 3072/UBND-VX (15/9/2021) của UBND TP.HCM")
    log()

    log("## G1-14. Lỗi #7: SO2 cao bất thường")
    so2 = raw_grid.groupby("station_id")["so2"].agg(
        tb="mean", trung_vi="median",
        pct_vuot_350_1h=lambda s: (s > SO2_QCVN_1H).mean() * 100).round(1)
    so2["trung_vi_ppm"] = (so2["trung_vi"] / 2620).round(3)
    so2.to_csv(report_dir / "g1_14_so2.csv")
    log.table(so2)
    log("- QCVN 05:2023/BTNMT: SO2 trung bình 1 giờ ≤ 350, 24 giờ ≤ 125, năm ≤ 50 µg/Nm³")
    log()

    pre_interp = df.copy()
    log("## G1-09. Biến dùng được theo trạm")
    has_pm = df[df["pm25"].notna()]
    others = [c for c in VARS if c != "pm25"]
    cond = has_pm.groupby("station_id")[others].apply(lambda g: g.isna().mean() * 100).round(1)
    usable = pd.DataFrame(np.where(cond <= cfg["usable_max_missing"] * 100, "dùng được", "chỉ thí nghiệm bổ sung"), index=cond.index, columns=cond.columns)
    usable.loc[:, "no2"] = np.where(usable.index == sat["station_id"], "chỉ thí nghiệm bổ sung", usable["no2"])
    usable.loc[:, "so2"] = "không dùng làm đặc trưng chính"
    usable.insert(0, "pm25", "biến mục tiêu")
    cond.to_csv(report_dir / "g1_09_missing_when_pm25_present.csv")
    usable.to_csv(report_dir / "g1_09_usable_variables.csv")
    log(f"% thiếu trong những giờ có PM2.5 (ngưỡng dùng được ≤ {int(cfg['usable_max_missing'] * 100)}%):")
    log.table(cond)
    log.table(usable)

    log(f"## G1-15. Nội suy tuyến tính khoảng trống ≤ {cfg['interp_max_hours']} giờ (trong từng trạm)")
    df, filled = interpolate_short(df, VARS, cfg["interp_max_hours"])
    df["is_imputed"] = filled["pm25"].astype("int8")
    log.table(pd.DataFrame({c: pd.Series(filled[c]).groupby(df["station_id"].values).sum() for c in VARS}))

    log("## G1-16. Nhãn đánh giá chỉ dùng giá trị đo thật")
    log(f"- Cột is_imputed = 1 ở {int(df['is_imputed'].sum())} giờ PM2.5 được nội suy")
    log("- Gửi TV3: trong hàm đánh giá, chỉ tính lỗi trên các dòng có `is_imputed == 0` và `pm25` không NaN")
    log()

    after_clean = missing_table(pre_interp)
    after_interp = missing_table(df)
    comp = pd.concat({"truoc": before, "sau_lam_sach": after_clean, "sau_noi_suy": after_interp}, axis=1)
    comp = comp.swaplevel(axis=1).sort_index(axis=1, level=0)
    comp.to_csv(report_dir / "g1_17_missing_before_after.csv")
    log("## G1-17. % thiếu trước / sau làm sạch / sau nội suy")
    log.table(pd.DataFrame({c: before[c].astype(str) + " → " + after_clean[c].astype(str) + " → " + after_interp[c].astype(str) for c in VARS}))

    df = df.sort_values(KEY).reset_index(drop=True)
    df.to_parquet(cfg["out_path"], index=False)
    write_dictionary(cfg["dictionary_path"], cfg)
    plot_report_figures(raw_grid, df, report_dir)
    log("## G1-18. Xuất dữ liệu")
    log(f"- {cfg['out_path']}: {len(df)} dòng, {df.shape[1]} cột")
    log(f"- Data dictionary: {cfg['dictionary_path']}")
    log(f"- Hình cho báo cáo: {report_dir / 'figures'}")
    (report_dir / "g1_cleaning_log.md").write_text("\n".join(log.lines), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/cleaning.json")