"""G1-33, G1-41 — Ghép khí tượng Open-Meteo vào dữ liệu HealthyAir đã làm sạch (TV1) -> hcm_merged.parquet.

    python -m src.data.merge_meteo   # cần data/interim/healthyair_clean.parquet (TV1) và meteo_hcm.parquet

Left join theo khoá (city, station_id, time) — station_id "hcm_01".."hcm_06" tương ứng Station_No 1..6.
Giữ nguyên mọi dòng của HealthyAir (lưới giờ đầy đủ của TV1). Open-Meteo không thiếu giờ nào nên mọi dòng
đều có khí tượng. Trong bảng ghép, tên biến Open-Meteo được đổi sang tên ngắn của lược đồ chung
(temp, rh, dew, precip, pressure, cloud_cover, wind_speed, wind_dir, wind_gusts) để dễ dùng cho EDA và mô hình.
Đầu ra: data/interim/hcm_merged.parquet + reports/meteo/g1_33_merge_check.csv
"""
from __future__ import annotations

import pandas as pd

from src.data.openmeteo_common import SHORT_NAMES, load_om_config, rpath

KEY = ["city", "station_id", "time"]


def merge(ha: pd.DataFrame, om: pd.DataFrame, stations: dict) -> pd.DataFrame:
    assert not ha.duplicated(KEY).any() and not om.duplicated(KEY).any()
    df = ha.merge(om.rename(columns=SHORT_NAMES), on=KEY, how="left", validate="one_to_one")
    df.insert(2, "area_type", df["station_id"].map({k: v["area_type"] for k, v in stations.items()}))
    assert len(df) == len(ha)
    return df


def main() -> None:
    cfg = load_om_config()
    ha = pd.read_parquet(rpath(cfg, "healthyair_clean"))
    om = pd.read_parquet(rpath(cfg, "out_hcm"))
    df = merge(ha, om, cfg["cities"]["hcm"]["stations"])

    met_cols = [SHORT_NAMES.get(c, c) for c in om.columns if c not in KEY]
    assert df[met_cols].isna().sum().sum() == 0, "Có dòng HealthyAir không ghép được khí tượng"
    chk = df.groupby("station_id").agg(
        rows=("time", "size"), t_min=("time", "min"), t_max=("time", "max"),
        pm25_available_pct=("pm25", lambda s: s.notna().mean() * 100),
        meteo_missing_pct=("temp", lambda s: s.isna().mean() * 100))
    # Đối chiếu nhiệt độ hai nguồn trên dữ liệu đã làm sạch (bước 4 trong đề xuất, mục 3.2)
    g = df.groupby("station_id")
    chk["n_temp_sensor"] = g["temp_sensor"].count()
    chk["corr_temp_sensor_vs_om"] = g.apply(lambda x: x["temp_sensor"].corr(x["temp"]), include_groups=False)
    chk["bias_temp_om_minus_sensor"] = g.apply(lambda x: (x["temp"] - x["temp_sensor"]).mean(), include_groups=False)
    out_rep = rpath(cfg, "report_dir")
    out_rep.mkdir(parents=True, exist_ok=True)
    chk.round(3).to_csv(out_rep / "g1_33_merge_check.csv")

    out = rpath(cfg, "merged_hcm")
    df.to_parquet(out, index=False)
    print(f"Ghép xong: {len(df)} dòng × {df.shape[1]} cột -> {out.name}")
    print(chk.round(3).to_string())


if __name__ == "__main__":
    main()
