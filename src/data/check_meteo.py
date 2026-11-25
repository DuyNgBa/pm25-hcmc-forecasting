"""G1-34, G1-35 — Kiểm tra chất lượng khí tượng Open-Meteo.

    python -m src.data.check_meteo                          # dùng data/raw/beijing (12 file PRSA của TV3)
    python -m src.data.check_meteo --beijing-dir <thư mục>  # nếu file Bắc Kinh để chỗ khác

1) G1-34, khớp giờ TP.HCM: tính tương quan giữa nhiệt độ/độ ẩm cảm biến HealthyAir và Open-Meteo khi
   dịch Open-Meteo đi từ -8 đến +8 giờ. Lệch tốt nhất phải bằng 0 (hoặc +1, vì HealthyAir là trung bình
   T:00-T:59, còn Open-Meteo là giá trị tức thời tại T). Bỏ trạm 4 (cảm biến đóng băng) và các giá trị 0.
2) G1-35, kiểm chứng ở Bắc Kinh: so khí tượng gốc PRSA (TEMP, DEWP, PRES, WSPM, RAIN) với Open-Meteo
   tại cùng trạm, cùng giờ: tương quan, độ lệch trung bình (bias), MAE, và lệch giờ tốt nhất.

Đầu ra: reports/meteo/g1_34_lag_hcm.csv, g1_35_beijing_validation.csv, g1_35_lag_beijing.csv,
        reports/meteo/figures/g1_34_lag_hcm.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.openmeteo_common import ROOT, SHORT_NAMES, load_om_config, rpath  # noqa: E402


# Thứ tự màu phân loại cố định (một màu cho mỗi trạm, không xoay vòng)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def lag_corr(a: pd.Series, b: pd.Series, lags) -> dict:
    """corr(a(t), b(t+lag)); a, b đánh chỉ số theo thời gian trên lưới giờ đầy đủ."""
    return {lag: a.corr(b.shift(-lag)) for lag in lags}


def check_hcm(cfg: dict, out_dir: Path) -> pd.DataFrame:
    raw = pd.read_csv(rpath(cfg, "healthyair_raw"))
    raw = raw.dropna(subset=["date"])
    raw["time"] = pd.to_datetime(raw["date"], format="%d-%m-%Y %H:%M")
    raw["station_id"] = "hcm_" + raw["Station_No"].astype(int).astype(str).str.zfill(2)
    om = pd.read_parquet(rpath(cfg, "out_hcm")).rename(columns=SHORT_NAMES)
    K = cfg["timezone_check"]["max_lag_hours"]
    lags = range(-K, K + 1)
    rows = []
    for sid, g in raw.groupby("station_id"):
        g = g.drop_duplicates("time").set_index("time").sort_index()
        o = om[om["station_id"] == sid].set_index("time").sort_index()
        idx = o.index
        for sensor, omv in [("Temperature", "temp"), ("Humidity", "rh")]:
            s = g[sensor].where(g[sensor] > 0)
            # Bỏ đoạn đóng băng: cùng một giá trị lặp >= 24 giờ liên tiếp
            run = s.groupby((s != s.shift()).cumsum()).transform("size")
            s = s.where(run < 24).reindex(idx)
            cors = lag_corr(s, o[omv], lags)
            best = max(cors, key=lambda k: cors[k] if pd.notna(cors[k]) else -9)
            rows.append({"station_id": sid, "variable": omv, "n": int(s.notna().sum()),
                         "best_lag": best, "corr_best": cors[best], "corr_lag0": cors[0],
                         **{f"lag_{k:+d}": v for k, v in cors.items()}})
    res = pd.DataFrame(rows)
    res.round(4).to_csv(out_dir / "g1_34_lag_hcm.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    names = {"temp": "Nhiệt độ", "rh": "Độ ẩm"}
    for ax, var in zip(axes, ["temp", "rh"]):
        for i, (_, r) in enumerate(res[res["variable"] == var].iterrows()):
            ax.plot(list(lags), [r[f"lag_{k:+d}"] for k in lags], marker="o", ms=4, lw=2,
                    color=PALETTE[i], label=f"{r['station_id']} (n={r['n']})")
        ax.axvline(0, color="grey", ls="--", lw=1)
        ax.set_title(f"{names[var]}: cảm biến vs Open-Meteo dịch k giờ", loc="left", fontsize=11)
        ax.set_xlabel("k (giờ)")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("Pearson r")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("G1-34: tương quan cao nhất ở k = 0/+1, nên HealthyAir dùng giờ Việt Nam (UTC+7)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "g1_34_lag_hcm.png", dpi=130)
    plt.close(fig)
    return res


def check_beijing(cfg: dict, bj_dir: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    om = pd.read_parquet(rpath(cfg, "out_beijing")).rename(columns=SHORT_NAMES)
    pairs = [("TEMP", "temp"), ("DEWP", "dew"), ("PRES", "pressure"), ("WSPM", "wind_speed"), ("RAIN", "precip")]
    rows, lag_rows = [], []
    for sid in cfg["cities"]["beijing"]["stations"]:
        f = next(bj_dir.glob(f"PRSA_Data_{sid}_*.csv"))
        d = pd.read_csv(f)
        d["time"] = pd.to_datetime(d[["year", "month", "day", "hour"]])
        d = d.set_index("time").sort_index()
        o = om[om["station_id"] == sid].set_index("time").sort_index().reindex(d.index)
        o_full = om[om["station_id"] == sid].set_index("time").sort_index()
        for nat, omv in pairs:
            a, b = d[nat], o[omv]
            m = a.notna() & b.notna()
            rows.append({"station_id": sid, "native": nat, "openmeteo": omv, "n": int(m.sum()),
                         "corr": a[m].corr(b[m]), "bias_om_minus_native": (b[m] - a[m]).mean(),
                         "mae": (b[m] - a[m]).abs().mean(),
                         "mean_native": a[m].mean(), "mean_om": b[m].mean()})
            if nat in ("TEMP", "DEWP"):
                full_idx = pd.date_range(o_full.index.min(), o_full.index.max(), freq="h")
                cors = lag_corr(a.reindex(full_idx), o_full[omv].reindex(full_idx), range(-3, 4))
                best = max(cors, key=cors.get)
                lag_rows.append({"station_id": sid, "variable": omv, "best_lag": best,
                                 **{f"lag_{k:+d}": v for k, v in cors.items()}})
    res, lag = pd.DataFrame(rows), pd.DataFrame(lag_rows)
    res.round(4).to_csv(out_dir / "g1_35_beijing_validation.csv", index=False)
    lag.round(4).to_csv(out_dir / "g1_35_lag_beijing.csv", index=False)
    return res, lag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beijing-dir", default=None)
    args = ap.parse_args()
    cfg = load_om_config()
    out_dir = rpath(cfg, "report_dir")
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    hcm = check_hcm(cfg, out_dir)
    print("## G1-34 lệch giờ tốt nhất (TP.HCM)")
    print(hcm[["station_id", "variable", "n", "best_lag", "corr_best", "corr_lag0"]].round(3).to_string(index=False))

    bj_dir = Path(args.beijing_dir) if args.beijing_dir else rpath(cfg, "beijing_raw_dir")
    if not bj_dir.is_absolute():
        bj_dir = ROOT / bj_dir
    if not any(bj_dir.glob("PRSA_Data_*.csv")):
        print(f"[bỏ qua G1-35] không thấy file PRSA trong {bj_dir}")
        return
    res, lag = check_beijing(cfg, bj_dir, out_dir)
    print("\n## G1-35 Open-Meteo vs khí tượng gốc Bắc Kinh (trung bình 12 trạm)")
    print(res.groupby(["native", "openmeteo"])[["n", "corr", "bias_om_minus_native", "mae"]]
             .mean().round(3).to_string())
    print("\nLệch giờ tốt nhất (Bắc Kinh):", lag.groupby("variable")["best_lag"].value_counts().to_dict())


if __name__ == "__main__":
    main()
