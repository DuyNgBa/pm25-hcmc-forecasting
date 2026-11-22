"""G1-36, G1-39, G1-40 — Kiểm tra đặc trưng khí tượng bằng ví dụ tay và kiểm tra KHÔNG nhìn trước tương lai.

    python -m pytest tests/test_meteo_features.py -v

Không cần mạng hay dữ liệu thật: mọi test dùng chuỗi tự tạo. Test cuối (đánh dấu 'data') chạy trên
data/interim/meteo_hcm.parquet nếu file đã có.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.meteo_features import add_features, hours_since_event, wind_components
from src.data.meteo_future import add_future_meteo, future_block
from src.data.openmeteo_common import ROOT

FCFG = {"rain_threshold_mm": 0.1, "precip_windows": [6, 24], "wind_mean_window": 6, "hours_since_rain_cap": 168}


def make_station(n=72, seed=0, sid="hcm_01"):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "city": "hcm", "station_id": sid,
        "time": pd.date_range("2021-03-01", periods=n, freq="h"),
        "wind_speed_10m": rng.uniform(0, 6, n).round(2),
        "wind_direction_10m": rng.integers(1, 361, n).astype(float),
        "precipitation": np.where(rng.random(n) < 0.2, rng.uniform(0.1, 5, n), 0.0).round(1),
        "temperature_2m": rng.normal(28, 2, n).round(1),
    })


# ---------------- G1-36: wind_u, wind_v bằng ví dụ tay ----------------
@pytest.mark.parametrize("speed,direction,u,v,meaning", [
    (5, 360, 0.0, -5.0, "gió Bắc: thổi TỪ Bắc xuống Nam -> v âm"),
    (5, 90, -5.0, 0.0, "gió Đông: thổi TỪ Đông sang Tây -> u âm"),
    (5, 180, 0.0, 5.0, "gió Nam (gió mùa mùa mưa TP.HCM): thổi lên Bắc -> v dương"),
    (5, 270, 5.0, 0.0, "gió Tây: thổi sang Đông -> u dương"),
    (2 * np.sqrt(2), 225, 2.0, 2.0, "gió Tây Nam: thổi về Đông Bắc -> u, v dương"),
    (0, 123, 0.0, 0.0, "lặng gió"),
])
def test_wind_components_hand_examples(speed, direction, u, v, meaning):
    uu, vv = wind_components(speed, direction)
    assert np.isclose(uu, u, atol=1e-9) and np.isclose(vv, v, atol=1e-9), meaning


def test_wind_components_keep_speed():
    d = make_station()
    u, v = wind_components(d["wind_speed_10m"], d["wind_direction_10m"])
    assert np.allclose(np.hypot(u, v), d["wind_speed_10m"])


def test_wind_components_match_team_schema():
    """Phải cùng công thức với src/data/schema.py của TV3 (nếu đã có trong repo)."""
    schema = pytest.importorskip("src.data.schema")
    d = make_station()
    a = wind_components(d["wind_speed_10m"], d["wind_direction_10m"])
    b = schema.wind_uv(d["wind_speed_10m"], d["wind_direction_10m"])
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


# ---------------- G1-37, G1-38: giá trị tính tay ----------------
def test_rolling_values_by_hand():
    d = make_station()
    d["precipitation"] = 0.0
    d.loc[10, "precipitation"] = 2.0
    d.loc[12, "precipitation"] = 1.0
    f = add_features(d, FCFG)
    assert np.isnan(f.loc[4, "precip_6h"]) and f.loc[5, "precip_6h"] == 0      # cần đủ 6 giờ
    assert f.loc[10, "precip_6h"] == 2.0                                       # giờ 5..10
    assert f.loc[12, "precip_6h"] == 3.0                                       # giờ 7..12
    assert f.loc[15, "precip_6h"] == 3.0 and f.loc[16, "precip_6h"] == 1.0     # giờ 10 rời cửa sổ ở t=16
    assert f.loc[30, "precip_24h"] == 3.0
    assert np.isclose(f.loc[20, "wind_speed_6h"], d.loc[15:20, "wind_speed_10m"].mean())


def test_hours_since_rain_by_hand():
    ev = np.array([0, 0, 1, 0, 0, 0, 1, 0], dtype=bool)
    assert hours_since_event(ev, cap=168).tolist() == [168, 168, 0, 1, 2, 3, 0, 1]
    assert hours_since_event(np.zeros(5, bool), cap=3).tolist() == [3, 3, 3, 3, 3]
    assert hours_since_event(np.array([1, 0, 0, 0, 0, 0], bool), cap=3).tolist() == [0, 1, 2, 3, 3, 3]


def test_rain_threshold():
    d = make_station(n=10)
    d["precipitation"] = [0, 0.05, 0, 0.1, 0, 0, 0, 0, 0, 0]
    f = add_features(d, FCFG)
    # 0.05 mm < ngưỡng 0.1 -> không tính là có mưa
    assert f["hours_since_rain"].tolist()[:6] == [168, 168, 168, 0, 1, 2]


# ---------------- G1-39: KHÔNG nhìn trước tương lai ----------------
@pytest.mark.parametrize("cut", [6, 24, 40, 60])
def test_features_do_not_look_ahead(cut):
    """Đổi toàn bộ dữ liệu SAU mốc `cut` -> mọi đặc trưng tại các giờ <= cut phải giữ nguyên."""
    a = make_station(seed=1)
    b = a.copy()
    rng = np.random.default_rng(99)
    after = b.index > cut
    b.loc[after, "wind_speed_10m"] = rng.uniform(0, 20, after.sum())
    b.loc[after, "wind_direction_10m"] = rng.integers(1, 361, after.sum())
    b.loc[after, "precipitation"] = rng.uniform(0, 50, after.sum())
    fa, fb = add_features(a, FCFG), add_features(b, FCFG)
    cols = ["wind_u", "wind_v", "wind_dir_sin", "wind_dir_cos", "precip_6h", "precip_24h",
            "hours_since_rain", "wind_speed_6h"]
    pd.testing.assert_frame_equal(fa.loc[:cut, cols], fb.loc[:cut, cols])


def test_requires_full_hourly_grid():
    d = make_station().drop(index=[5])      # thiếu 1 giờ -> rolling theo dòng sẽ sai, phải báo lỗi
    with pytest.raises(AssertionError):
        add_features(d, FCFG)


# ---------------- G1-40: khí tượng tương lai ----------------
def test_future_meteo_is_value_at_t_plus_h():
    d = pd.concat([make_station(sid="hcm_01", seed=1), make_station(sid="hcm_02", seed=2)], ignore_index=True)
    f = add_future_meteo(d, cols=["temperature_2m"], horizons=[1, 6, 24])
    for sid, g in f.groupby("station_id"):
        g = g.reset_index(drop=True)
        assert g.loc[0, "temperature_2m_fut6"] == g.loc[6, "temperature_2m"]
        assert g.loc[10, "temperature_2m_fut24"] == g.loc[34, "temperature_2m"]
        assert g["temperature_2m_fut24"].iloc[-24:].isna().all()      # không mượn dữ liệu trạm khác


def test_future_block_shape_and_values():
    d = make_station(seed=3)
    origins = d.loc[[0, 10, 60], ["city", "station_id", "time"]].reset_index(drop=True)
    X, ok = future_block(d, origins, cols=["temperature_2m", "precipitation"], H=24)
    assert X.shape == (3, 24, 2)
    assert np.isclose(X[1, 0, 0], d.loc[11, "temperature_2m"])          # t+1
    assert np.isclose(X[1, 23, 1], d.loc[34, "precipitation"])          # t+24
    assert ok.tolist() == [True, True, False]                           # gốc 60: t+24 vượt cuối chuỗi


# ---------------- Trên dữ liệu thật (nếu có) ----------------
@pytest.mark.skipif(not (ROOT / "data/interim/meteo_hcm.parquet").exists(), reason="chưa chạy meteo_features")
def test_real_data_no_lookahead_and_complete():
    m = pd.read_parquet(ROOT / "data/interim/meteo_hcm.parquet")
    g = m[m["station_id"] == "hcm_01"].sort_values("time").reset_index(drop=True)
    for t in [100, 5000, 11000]:   # tính lại tay từ dữ liệu quá khứ tại vài mốc
        assert np.isclose(g.loc[t, "precip_24h"], g.loc[t - 23:t, "precipitation"].sum())
        assert np.isclose(g.loc[t, "wind_speed_6h"], g.loc[t - 5:t, "wind_speed_10m"].mean())
    assert m.drop(columns=["precip_6h", "precip_24h", "wind_speed_6h"]).isna().sum().sum() == 0
