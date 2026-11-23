"""Kiểm tra chống dùng nhầm kết quả/checkpoint/HPO của dữ liệu cũ và script kiểm tra G2-49. Không train, chạy vài giây."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.provenance import (LEGACY_FP, check_best, frame_fingerprint, runs_with_fp, select_fp,
                                   train_fingerprint)
from src.models.rnn_check import Log, check_coverage, check_truth, compare_with_reference, expected_table
from src.models.rnn_experiments import Spec


def toy_schema(seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2022-03-01", periods=48, freq="h")
    return pd.concat([pd.DataFrame({"city": "hcm", "station_id": s, "time": t, "pm25": rng.gamma(4, 5, len(t)),
                                    "is_imputed": np.int8(0), "temp": rng.normal(28, 2, len(t))})
                      for s in ("hcm_01", "hcm_02")], ignore_index=True)


def test_fingerprint_ignores_row_and_column_order():
    d = toy_schema()
    shuffled = d.sample(frac=1, random_state=1)[list(reversed(d.columns))]
    assert frame_fingerprint(d) == frame_fingerprint(shuffled)


def test_fingerprint_detects_one_changed_value():
    d = toy_schema()
    e = d.copy()
    e.loc[5, "temp"] += 0.01            # vd đổi nguồn khí tượng: một giá trị đổi là dấu vân tay đổi
    assert frame_fingerprint(d) != frame_fingerprint(e)


def test_fingerprint_stable_to_float_noise():
    d = toy_schema()
    e = d.copy()
    e["temp"] = e["temp"] + 1e-12       # lệch bit cuối giữa hai máy không làm đổi dấu vân tay
    assert frame_fingerprint(d) == frame_fingerprint(e)


def test_run_id_changes_with_data_key():
    s = Spec(cell="gru", hp={"L": 24, "hidden": 64, "layers": 3, "dropout": 0.3, "lr": 1e-3, "batch_size": 256})
    assert s.run_id("v0.9-dev|aaa|t") != s.run_id("v0.9-dev|bbb|t")   # cùng data_version, khác dữ liệu
    assert s.run_id("v0.9-dev|aaa|t") == s.run_id("v0.9-dev|aaa|t")


def test_train_fingerprint_ignores_best_and_seeds():
    r = {"residual": True, "train": {"max_epochs": 60}, "best": {"gru": {"L": 24}}, "seeds": [42]}
    r2 = {**r, "best": {"gru": {"L": 48}}, "seeds": [1, 2]}
    r3 = {**r, "train": {"max_epochs": 30}}
    assert train_fingerprint(r) == train_fingerprint(r2)
    assert train_fingerprint(r) != train_fingerprint(r3)


def test_check_best_rejects_hp_from_other_data():
    r = {"best": {"gru": {"L": 24, "data_fp": "old"}}}
    assert check_best(r, "gru", "new") is not None
    assert check_best(r, "gru", "old") is None
    assert check_best({"best": {"gru": {"L": 24}}}, "gru", "new") is not None    # best cũ không ghi data_fp
    assert check_best({"best": {}}, "lstm", "new") is not None


def test_select_fp_never_mixes():
    runs = runs_with_fp(pd.DataFrame({"run_id": ["a", "b"], "data_fp": [None, "x"]}))
    assert set(runs["data_fp"]) == {LEGACY_FP, "x"}
    assert select_fp(runs, local_fp="x") == "x"
    assert select_fp(runs, requested=LEGACY_FP) == LEGACY_FP
    with pytest.raises(SystemExit):
        select_fp(runs, local_fp="khác")          # 2 bộ, không bộ nào khớp local -> bắt chọn


def preds(stations=("hcm_01", "hcm_02"), shift_h=0, y_offset=0.0):
    t0 = pd.Timestamp("2022-03-02 00:00")
    rows = []
    for s in stations:
        for k in range(5):
            for h in (1, 6, 24):
                rows.append({"city": "hcm", "station_id": s, "time_origin": t0 + pd.Timedelta(hours=k + shift_h),
                             "horizon": h, "y_true": 10.0 + k + h + y_offset, "y_pred": 11.0})
    return pd.DataFrame(rows)


def test_compare_with_reference_detects_key_and_target_mismatch():
    ok = Log()
    compare_with_reference(preds(), preds(), "x", ok)
    assert not ok.errors
    shifted = Log()
    compare_with_reference(preds(shift_h=1), preds(), "x", shifted)           # lệch 1 giờ, cùng số dòng
    assert shifted.errors
    wrong_y = Log()
    compare_with_reference(preds(y_offset=0.5), preds(), "x", wrong_y)        # cùng khoá, sai mục tiêu
    assert any("y_true" in e for e in wrong_y.errors)


def test_check_truth_uses_real_value_at_t_plus_h():
    p = preds(stations=("hcm_01",))
    tt = p["time_origin"] + pd.to_timedelta(p["horizon"], unit="h")
    truth = pd.Series(p["y_true"].to_numpy(), index=pd.MultiIndex.from_arrays([p["station_id"], tt]))
    truth = truth[~truth.index.duplicated()]
    log = Log()
    check_truth(p, truth, "x", log)
    assert not log.errors
    log2 = Log()
    check_truth(p.assign(y_true=p["y_true"] + 1), truth, "x", log2)
    assert log2.errors


def test_coverage_flags_missing_and_duplicate_seed():
    rcfg = {"seeds": [42, 43, 44], "defaults": {"L": 24, "hidden": 64, "layers": 2, "dropout": 0.1, "lr": 1e-3,
                                                "batch_size": 256}}
    exp = expected_table(rcfg, ["rq1"], cells=("gru",))
    assert len(exp) == 3
    runs = exp.iloc[[0, 1, 1]].assign(run_id=["r1", "r2", "r3"])         # thiếu seed 44, trùng seed 43
    log = Log()
    check_coverage(runs, exp, log)
    assert any("Thiếu" in e for e in log.errors) and any("Trùng" in e for e in log.errors)


# ---------------- nhãn "CHÍNH THỨC" phải khớp bản dữ liệu nhóm đã khoá ----------------
from src.models.provenance import _raw_file_digest, compare_lock          # noqa: E402
from src.models.rnn_report import status_of                              # noqa: E402

META = {"data_fp": "f1", "data_version": "v1.0", "raw_fp": "r1", "hcm_fp": "h1", "beijing_fp": "b1"}
LOCK = {"data_version": "v1.0", "raw_fp": "r1", "hcm_fp": "h1", "beijing_fp": "b1"}
CHK = {"ok": True, "data_fp": "f1", "checked_at": "x"}


def test_official_needs_lock_match_not_just_v1_string():
    assert status_of(META, CHK, LOCK)[0]
    assert not status_of(META, CHK, None)[0]                                  # chưa có bản khoá
    assert not status_of({**META, "hcm_fp": "h2"}, CHK, LOCK)[0]              # "v1.0" nhưng khác dữ liệu đã khoá
    assert not status_of({**META, "raw_fp": "r2"}, CHK, LOCK)[0]
    assert not status_of(META, {**CHK, "ok": False}, LOCK)[0]                 # checker không đạt
    assert not status_of(META, {**CHK, "data_fp": "khác"}, LOCK)[0]           # checker chạy cho bộ khác
    assert not status_of({**META, "data_version": "v1.1"}, CHK, {**LOCK, "data_version": "v1.1"})[0]


def test_compare_lock_reports_each_mismatch():
    assert compare_lock(META, LOCK) == []
    d = compare_lock({**META, "beijing_fp": "x"}, LOCK)
    assert len(d) == 1 and "beijing_fp" in d[0]


def test_raw_digest_ignores_crlf_and_api_timing(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_bytes(b"x,y\n1,2\n")
    b.write_bytes(b"x,y\r\n1,2\r\n")                       # Windows autocrlf
    assert _raw_file_digest(a) == _raw_file_digest(b)
    j1, j2 = tmp_path / "j1.json", tmp_path / "j2.json"
    j1.write_text('{"generationtime_ms": 1.0, "latitude": 1, "hourly": {"t": [1, 2]}}')
    j2.write_text('{"generationtime_ms": 9.9, "latitude": 1, "hourly": {"t": [1, 2]}}')
    assert _raw_file_digest(j1) == _raw_file_digest(j2)
    j2.write_text('{"generationtime_ms": 9.9, "latitude": 1, "hourly": {"t": [1, 3]}}')
    assert _raw_file_digest(j1) != _raw_file_digest(j2)
