"""G1-26, G1-27 — Xuất bảng toạ độ trạm ra data/stations/stations_{hcm,beijing}.csv.

    python -m src.data.stations

- TP.HCM (G1-26): toạ độ từ Bảng 1 bài Data in Brief [1] (đã đổi lại 2 cột bị ghi ngược tên), đối chiếu
  với bảng toạ độ trong đề xuất (làm tròn 4 chữ số) -> cột diff_m = khoảng cách lệch (mét).
- Bắc Kinh (G1-27): bộ UCI chỉ có tên trạm. Toạ độ lấy từ Bảng 1 của
  Wu, L., Xie, J., Kang, K. (2022). Changing weekend effects of air pollutants in Beijing under 2020 COVID-19
  lockdown controls. npj Urban Sustainability, 2, 23. https://doi.org/10.1038/s42949-022-00070-0
  (bảng mô tả 32 trạm quan trắc quốc gia/thành phố Bắc Kinh: kinh độ, vĩ độ, độ cao, loại vị trí).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.openmeteo_common import ROOT, load_om_config

# Bảng toạ độ trong đề xuất (De_xuat_do_an_PM25_TPHCM_v1.pdf, mục 3.2)
PROPOSAL_HCM = {"hcm_01": (10.8699, 106.7960), "hcm_02": (10.7410, 106.6171), "hcm_03": (10.8162, 106.6204),
                "hcm_04": (10.8158, 106.7174), "hcm_05": (10.7764, 106.6878), "hcm_06": (10.7805, 106.6595)}
HCM_SOURCE = ("Rakholia et al. (2023), Data in Brief 46, 108774, Table 1 "
              "(cột latitude/longitude bị ghi ngược tên trong bài gốc, đã đổi lại)")
# Bảng 1 của Wu et al. (2022): tên viết tắt, mô tả vị trí, độ cao (m)
BEIJING_META = {
    "Aotizhongxin": ("ATZX", "Inside Olympic Center (Sports Center in urban)", 42),
    "Changping": ("CP", "Urban Area of Changping (Block in town)", 74),
    "Dingling": ("DL", "Dingling, Changping (Park in suburban)", 231),
    "Dongsi": ("DS", "Dongcheng District Environmental Protection Bureau (Block in urban)", 31),
    "Guanyuan": ("GY", "Beijing Municipal Party School (School in urban)", 47),
    "Gucheng": ("GC", "Shijingshan Environmental Protection Bureau (Institution in urban)", 71),
    "Huairou": ("HR", "Urban Area of Huairou (Block in town)", 52),
    "Nongzhanguan": ("NZG", "Residential Area of Agricultural Exhibition Hall (Residential zone in urban)", 43),
    "Shunyi": ("SY", "Urban Area of Shunyi (Block in town)", 15),
    "Tiantan": ("TT", "Inside Temple of Heaven Park (Park in urban)", 33),
    "Wanliu": ("WL", "Inside Changchun Fitness Park (Park in urban)", 44),
    "Wanshouxigong": ("WSXG", "Inside Wanshou Park (Park in urban)", 51),
}
BEIJING_SOURCE = ("Wu, Xie & Kang (2022), npj Urban Sustainability 2:23, Table 1, "
                  "https://doi.org/10.1038/s42949-022-00070-0")


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_om_config()
    hcm = []
    for sid, st in cfg["cities"]["hcm"]["stations"].items():
        plat, plon = PROPOSAL_HCM[sid]
        hcm.append({"station_id": sid, "station_no": int(sid[-2:]), "lat": st["lat"], "lon": st["lon"],
                    "area_type": st["area_type"], "lat_proposal": plat, "lon_proposal": plon,
                    "diff_m": round(float(haversine_m(st["lat"], st["lon"], plat, plon)), 1),
                    "source": HCM_SOURCE})
    bj = []
    for sid, st in cfg["cities"]["beijing"]["stations"].items():
        abbr, loc, elev = BEIJING_META[sid]
        bj.append({"station_id": sid, "abbr": abbr, "lat": st["lat"], "lon": st["lon"], "elevation_m": elev,
                   "location": loc, "source": BEIJING_SOURCE})
    return pd.DataFrame(hcm), pd.DataFrame(bj)


def main() -> None:
    hcm, bj = build()
    out = ROOT / "data" / "stations"
    out.mkdir(parents=True, exist_ok=True)
    hcm.to_csv(out / "stations_hcm.csv", index=False, lineterminator="\n")
    bj.to_csv(out / "stations_beijing.csv", index=False, lineterminator="\n")
    print(hcm.drop(columns="source").to_string(index=False))
    print(f"\nLệch lớn nhất so với đề xuất: {hcm['diff_m'].max()} m (do đề xuất làm tròn 4 chữ số)")
    print(bj.drop(columns=["source", "location"]).to_string(index=False))


if __name__ == "__main__":
    main()
