"""Tiện ích dùng chung cho phần khí tượng Open-Meteo (TV2)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

# Tên biến API -> tên ngắn của lược đồ chung (khớp OPENMETEO_MAP trong src/data/schema.py của TV3)
SHORT_NAMES = {
    "temperature_2m": "temp", "relative_humidity_2m": "rh", "dew_point_2m": "dew",
    "precipitation": "precip", "surface_pressure": "pressure", "cloud_cover": "cloud_cover",
    "wind_speed_10m": "wind_speed", "wind_direction_10m": "wind_dir", "wind_gusts_10m": "wind_gusts",
}


def load_om_config(path: str | Path = "configs/openmeteo.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def rpath(cfg: dict, key: str) -> Path:
    """Đường dẫn tuyệt đối của một mục trong config (tương đối so với gốc repo)."""
    return ROOT / cfg[key]


def raw_json(cfg: dict, city: str, station_id: str) -> Path:
    """File JSON thô (phản hồi API) của một trạm."""
    return rpath(cfg, "raw_dir") / f"{city}_{station_id}.json"


def read_raw(cfg: dict, city: str, station_id: str) -> pd.DataFrame:
    """Đọc JSON thô -> DataFrame theo giờ, tên cột GIỮ NGUYÊN như API (time + 9 biến)."""
    js = json.loads(raw_json(cfg, city, station_id).read_text(encoding="utf-8"))
    df = pd.DataFrame(js["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df
