"""Unit test cho pipeline chung (G1-52, G1-59 và các hàm dùng chung).

    pytest -q                      # chạy tất cả
    pytest -q -k leakage           # chỉ test chống rò rỉ

Phần lớn test chạy trên dữ liệu tổng hợp nhỏ (không cần tải dữ liệu); các test đánh dấu
`needs_data` chạy trên dữ liệu thật nếu data/processed/*_schema.parquet đã có, nếu không thì bỏ qua.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.cleaning_common import interpolate_short_gaps
from src.data.pipeline import prepare
from src.data.scaling import fit_scalers, inverse_pm25, transform
from src.data.schema import SCHEMA_COLS, wind_uv
from src.data.splits import get_split_ranges, in_range, subsample_train_range
from src.data.windows import add_neighbor_pm25, feature_columns, make_windows
from src.eval.aqi import pm25_to_level
from src.eval.metrics import regression_scores
from src.utils import ROOT, load_config

CFG = load_config()
HAS_DATA = (ROOT / CFG["paths"]["hcm_schema"]).exists() and (ROOT / CFG["paths"]["beijing_schema"]).exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="chưa build data/processed/*_schema.parquet")


# ------------------------------------------------------------------ dữ liệu tổng hợp
def synthetic(n_hours=24 * 60, stations=("hcm_01", "hcm_02", "hcm_03", "hcm_04", "hcm_05", "hcm_06"), city="hcm", seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2021-11-01", periods=n_hours, freq="h")
    rows = []
    for i, s in enumerate(stations):
        base = 20 + 3 * i + 8 * np.sin(2 * np.pi * (t.hour - 7) / 24)
        pm = np.clip(np.asarray(base) + rng.normal(0, 3, n_hours), 1, None)
        pm[100 + 10 * i: 110 + 10 * i] = np.nan                 # một khối thiếu
        d = pd.DataFrame({"city": city, "station_id": s, "time": t, "pm25": pm, "is_imputed": 0})
        for c in ["temp", "rh", "dew", "wind_speed", "wind_u", "wind_v", "precip", "precip_6h",
                  "precip_24h", "pressure", "temp_ha", "rh_ha"]:
            d[c] = rng.normal(size=n_hours)
        d["precip"] = d["precip_6h"] = d["precip_24h"] = np.abs(d["precip"])
        h, w = t.hour, t.dayofweek
        d["hour_sin"], d["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
        d["dow_sin"], d["dow_cos"] = np.sin(2 * np.pi * w / 7), np.cos(2 * np.pi * w / 7)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


SYN_RNG = {"train": (pd.Timestamp("2021-11-01"), pd.Timestamp("2021-12-10 23:00")),
           "val": (pd.Timestamp("2021-12-11"), pd.Timestamp("2021-12-20 23:00")),
           "test": (pd.Timestamp("2021-12-21"), pd.Timestamp("2021-12-30 23:00"))}


def syn_prepared():
    df = synthetic()
    p = fit_scalers(df, in_range(df["time"], SYN_RNG["train"]), CFG)
    return transform(df, p), p


# ------------------------------------------------------------------ G1-52: đảo ngược chuẩn hoá
def test_inverse_roundtrip_synthetic():
    df_s, p = syn_prepared()
    ok = df_s["pm25"].notna()
    back = inverse_pm25(df_s.loc[ok, "pm25_z"], df_s.loc[ok, "station_id"], p, "hcm")
    np.testing.assert_allclose(back, df_s.loc[ok, "pm25"], rtol=1e-9, atol=1e-9)


def test_inverse_2d_shape():
    _, p = syn_prepared()
    z = np.zeros((3, 24))
    out = inverse_pm25(z, ["hcm_01", "hcm_02", "hcm_03"], p, "hcm")
    assert out.shape == (3, 24)
    # z = 0 -> exp(mu) - 1 (trung vị hình học của trạm)
    assert np.isclose(out[0, 0], np.expm1(p["pm25"]["hcm"]["hcm_01"]["mu"]))


@needs_data
@pytest.mark.parametrize("city", ["hcm", "beijing"])
def test_inverse_roundtrip_real(city):
    P = prepare(city)
    d = P.df[P.df["pm25"].notna()]
    back = inverse_pm25(d["pm25_z"], d["station_id"], P.params, city)
    np.testing.assert_allclose(back, d["pm25"], rtol=1e-8, atol=1e-6)


# ------------------------------------------------------------------ G1-59: chống rò rỉ
def test_leakage_windows_synthetic():
    df_s, _ = syn_prepared()
    feats = feature_columns("c")
    for L in (24, 48, 72):
        tr = make_windows(df_s, feats, L, 24, target_range=SYN_RNG["train"], input_start=SYN_RNG["train"][0])
        va = make_windows(df_s, feats, L, 24, target_range=SYN_RNG["val"])
        te = make_windows(df_s, feats, L, 24, target_range=SYN_RNG["test"])
        assert len(tr) and len(va) and len(te)
        # Không timestamp nào (input hay mục tiêu) của cửa sổ train >= đầu val
        assert tr.input_times().max() < SYN_RNG["val"][0]
        assert tr.target_times().max() <= SYN_RNG["train"][1]
        # Mục tiêu val/test nằm trọn trong split của mình
        assert va.target_times().min() >= SYN_RNG["val"][0] and va.target_times().max() <= SYN_RNG["val"][1]
        assert te.target_times().min() >= SYN_RNG["test"][0] and te.target_times().max() <= SYN_RNG["test"][1]


def test_scaler_fit_only_on_train():
    df = synthetic()
    tr = in_range(df["time"], SYN_RNG["train"])
    p1 = fit_scalers(df, tr, CFG)
    df2 = df.copy()
    df2.loc[~tr, "pm25"] *= 100          # phá dữ liệu ngoài train
    df2.loc[~tr, "temp"] += 50
    p2 = fit_scalers(df2, tr, CFG)
    assert p1["pm25"] == p2["pm25"] and p1["meteo"] == p2["meteo"], "scaler bị ảnh hưởng bởi dữ liệu ngoài train"


@needs_data
@pytest.mark.parametrize("fold", [None, 1, 2, 3, 4])
def test_leakage_real_hcm(fold):
    P = prepare("hcm", fold=fold)
    feats = feature_columns("a")
    tr = P.windows("train", feats, L=72)
    assert tr.input_times().max() < P.rng["val"][0]
    assert tr.target_times().max() <= P.rng["train"][1]
    assert P.params["fit_info"]["train_range"][1] == str(P.rng["train"][1])
    # scaler phải giống hệt khi fit trên bảng đã bỏ toàn bộ dữ liệu sau train
    df_cut = P.df[P.df["time"] <= P.rng["train"][1]]
    p_cut = fit_scalers(df_cut, np.ones(len(df_cut), bool) & in_range(df_cut["time"], P.rng["train"]), CFG)
    assert p_cut["pm25"] == P.params["pm25"]


# ------------------------------------------------------------------ cửa sổ, NaN, trạm lân cận
def test_windows_drop_nan_and_within_station():
    df_s, _ = syn_prepared()
    ws = make_windows(df_s, ["pm25_z"], 24, 24, target_range=SYN_RNG["train"], input_start=SYN_RNG["train"][0])
    X, Y, M = ws.materialize()
    assert not np.isnan(X).any()
    meta = ws.meta()
    # mỗi cửa sổ chỉ của 1 trạm: thời gian input liên tục đúng L giờ
    it = ws.input_times()
    assert ((it[:, -1] - it[:, 0]) == np.timedelta64(23, "h")).all()
    assert set(meta["station_id"]) == {"hcm_01", "hcm_02", "hcm_03", "hcm_04", "hcm_05", "hcm_06"}


def test_windows_fill_policy_adds_mask():
    df_s, _ = syn_prepared()
    ws = make_windows(df_s, ["pm25_z"], 24, 24, target_range=SYN_RNG["train"],
                      input_start=SYN_RNG["train"][0], nan_policy="fill")
    assert ws.feature_cols == ["pm25_z", "pm25_z_isna"]
    X, _, _ = ws.materialize()
    assert not np.isnan(X).any() and X[..., 1].max() == 1


def test_neighbors_nearest_first():
    df_s, _ = syn_prepared()
    out = add_neighbor_pm25(df_s, CFG["stations"]["hcm"])
    r = out[(out.station_id == "hcm_05")].iloc[500]
    # Trạm gần hcm_05 (Q.3) nhất theo toạ độ là hcm_06 (Q.10)
    s6 = out[(out.station_id == "hcm_06") & (out.time == r.time)]["pm25_z"].iloc[0]
    assert np.isclose(r["pm25_nb1"], s6)
    assert add_neighbor_pm25(df_s, CFG["stations"]["hcm"], enabled=False).shape == df_s.shape


# ------------------------------------------------------------------ chia dữ liệu
def test_subsample_contiguous():
    tr = (pd.Timestamp("2021-02-23"), pd.Timestamp("2021-12-31 23:00"))
    n = len(pd.date_range(*tr, freq="h"))
    for frac in (0.5, 0.25):
        s, e = subsample_train_range(tr, frac, "last")
        assert e == tr[1] and abs(len(pd.date_range(s, e, freq="h")) - n * frac) <= 1
        a = subsample_train_range(tr, frac, "random", seed=1)
        b = subsample_train_range(tr, frac, "random", seed=1)
        assert a == b and a[0] >= tr[0] and a[1] <= tr[1]


def test_folds_ordered_and_disjoint():
    for f in [None, 1, 2, 3, 4]:
        r = get_split_ranges(CFG, "hcm", fold=f)
        assert r["train"][1] < r["val"][0] <= r["val"][1] < r["test"][0] <= r["test"][1]


# ------------------------------------------------------------------ lược đồ, gió, đánh giá, AQI
def test_wind_uv_convention():
    u, v = wind_uv([5, 5, 5], [0, 90, 270])   # gió Bắc, gió Đông, gió Tây
    np.testing.assert_allclose(u, [0, -5, 5], atol=1e-9)
    np.testing.assert_allclose(v, [-5, 0, 0], atol=1e-9)


@needs_data
def test_schema_same_columns():
    a = pd.read_parquet(ROOT / CFG["paths"]["hcm_schema"])
    b = pd.read_parquet(ROOT / CFG["paths"]["beijing_schema"])
    assert list(a.columns) == list(b.columns) == SCHEMA_COLS
    assert "month" not in a.columns


def test_interpolate_only_short_bounded_gaps():
    s = pd.Series([1, np.nan, np.nan, 4, np.nan, np.nan, np.nan, np.nan, 9, np.nan])
    out, flag = interpolate_short_gaps(s, 3)
    assert np.allclose(out[:4], [1, 2, 3, 4])
    assert out[4:8].isna().all() and np.isnan(out[9])        # khối 4 giờ & đuôi không nội suy
    assert flag.tolist() == [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]


def test_metrics_ignore_nan_and_high():
    r = regression_scores([10, 40, np.nan, 50], [12, 30, 5, 50])
    assert r["n"] == 3 and np.isclose(r["MAE"], 4.0) and np.isclose(r["MAE_high"], 5.0)


def test_aqi_levels_epa_2024():
    lv = pm25_to_level([0, 9.0, 9.04, 9.1, 35.4, 35.44, 35.5, 55.4, 125.4, 125.5, 225.5, np.nan])
    assert lv.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 5, -1]
    vn = pm25_to_level([25, 25.1, 50, 80, 150, 250, 251], "vn_aqi_2019")
    assert vn.tolist() == [0, 1, 1, 2, 3, 4, 5]


# ------------------------------------------------------------------ tập đánh giá chung
@needs_data
def test_common_eval_set_same_across_L_and_ablation():
    P = prepare("hcm")
    ref = None
    for ab in ("a", "b"):
        for L in (24, 48, 72):
            ws = P.windows("test", feature_columns(ab), L=L)
            key = ws.meta().sort_values(["station_id", "time_origin"]).reset_index(drop=True)
            if ref is None:
                ref = key
            pd.testing.assert_frame_equal(key, ref)
            if ab == "b":
                assert ws.feature_cols[-2:] == ["temp_ha_s_isna", "rh_ha_s_isna"]
                assert not np.isnan(ws.materialize()[0]).any()
