"""Kiểm tra phần LSTM/GRU (G2-27..G2-31): batch nhanh khớp WindowSet, không vắt qua trạm, mask, shape,
đóng băng encoder, early stopping khôi phục trọng số tốt nhất. Chỉ dùng dữ liệu tự tạo, chạy trên CPU vài giây."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.windows import make_windows
from src.models.rnn import DirectRNN, Seq2SeqRNN, build_model, count_params
from src.models.rnn_data import FastWindows, balanced_weights, batches
from src.models.train_rnn import TrainConfig, fit, masked_mse, predict_z

L, H = 12, 6


def toy_df(stations=("s1", "s2"), n=200, seed=0, city="hcm"):
    rng = np.random.default_rng(seed)
    parts = []
    for k, s in enumerate(stations):
        t = pd.date_range("2021-01-01", periods=n, freq="h")
        pm = 20 + 5 * np.sin(np.arange(n) / 4) + rng.normal(0, 1, n) + 100 * k   # mỗi trạm một mức khác hẳn
        d = pd.DataFrame({"city": city, "station_id": s, "time": t, "pm25": pm, "is_imputed": 0,
                          "pm25_z": (pm - pm.mean()) / pm.std(), "f1": rng.normal(size=n)})
        d.loc[50:53, ["pm25", "pm25_z"]] = np.nan          # lỗ thiếu -> cửa sổ chứa nó bị bỏ
        d.loc[120, "is_imputed"] = 1
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


@pytest.fixture
def ws():
    return make_windows(toy_df(), ["pm25_z", "f1"], L=L, H=H)


def test_fast_windows_match_windowset(ws):
    fw = FastWindows([ws])
    assert len(fw) == len(ws) > 0
    idx = np.array([0, 5, len(ws) // 2, len(ws) - 1])
    x, y, m, f = fw.get(idx)
    for j, i in enumerate(idx):
        xi, yi, mi = ws[i]
        np.testing.assert_allclose(x[j], xi)
        np.testing.assert_allclose(y[j], yi)
        assert (m[j] == mi).all()
    assert f is None


def test_no_window_crosses_stations(ws):
    """Trạm s2 có PM2.5 cao hơn s1 ~100 đơn vị: nếu cửa sổ vắt qua 2 trạm sẽ thấy bước nhảy z rất lớn trong x."""
    fw = FastWindows([ws])
    x, _, _, _ = fw.get(np.arange(len(fw)))
    # z được chuẩn hoá theo trạm nên trong cùng trạm |Δz| nhỏ; ranh giới 2 trạm không bao giờ nằm trong cửa sổ
    meta = ws.meta()
    assert meta.groupby("station_id").size().shape[0] == 2
    assert np.abs(np.diff(x[:, :, 0], axis=1)).max() < 3


def test_extra_const_and_balanced_weights(ws):
    ws2 = make_windows(toy_df(stations=("b1", "b2", "b3"), city="beijing", seed=1), ["pm25_z", "f1"], L=L, H=H)
    fw = FastWindows([ws2, ws], extra_const=[0.0, 1.0])
    assert fw.n_features == 3 and fw.feature_cols[-1] == "is_hcm"
    x, *_ = fw.get(np.arange(len(fw)))
    assert set(np.unique(x[:, :, -1])) == {0.0, 1.0}
    w = balanced_weights(fw, [0.5, 0.5])
    assert np.isclose(w[fw.set_id == 0].sum(), 0.5) and np.isclose(w[fw.set_id == 1].sum(), 0.5)
    got = np.random.default_rng(0).choice(len(fw), 20000, p=w)
    assert abs((fw.set_id[got] == 1).mean() - 0.5) < 0.02


@pytest.mark.parametrize("arch,cell", [("direct", "lstm"), ("direct", "gru"), ("seq2seq", "lstm"), ("seq2seq", "gru")])
def test_model_shapes(arch, cell):
    m = build_model(arch, n_features=4, H=H, cell=cell, hidden=16, layers=2, dropout=0.1)
    x = torch.randn(8, L, 4)
    assert m(x).shape == (8, H)
    mf = build_model(arch, n_features=4, H=H, cell=cell, hidden=16, layers=1, dropout=0.0, fut_dim=3)
    assert mf(x, fut=torch.randn(8, H, 3)).shape == (8, H)


def test_gru_has_fewer_params_than_lstm():
    a = count_params(DirectRNN(10, H, "lstm", 64, 2))
    b = count_params(DirectRNN(10, H, "gru", 64, 2))
    assert b < a   # GRU 3 cổng, LSTM 4 cổng


def test_masked_mse_ignores_missing():
    p = torch.tensor([[1.0, 100.0]])
    y = torch.tensor([[0.0, 0.0]])
    m = torch.tensor([[True, False]])
    assert float(masked_mse(p, y, m)) == 1.0


def test_head_only_finetune_freezes_encoder(ws):
    fw = FastWindows([ws])
    for arch in ("direct", "seq2seq"):
        model = build_model(arch, fw.n_features, H, "lstm", 8, 1, 0.0)
        enc0 = {k: v.clone() for k, v in model.encoder.state_dict().items()}
        head0 = [p.clone() for p in model.head_parameters()]
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        fit(model, fw, fw, TrainConfig(max_epochs=2, patience=5, verbose=False), "cpu", params=model.head_parameters())
        for k, v in model.encoder.state_dict().items():
            assert torch.equal(v, enc0[k]), "encoder phải giữ nguyên khi fine-tune lớp cuối"
        assert any(not torch.equal(a, b) for a, b in zip(head0, model.head_parameters()))


def test_fit_learns_and_restores_best(ws, tmp_path):
    fw = FastWindows([ws])
    model = DirectRNN(fw.n_features, H, "gru", 16, 1, 0.0)
    res = fit(model, fw, fw, TrainConfig(lr=3e-3, max_epochs=15, patience=15, verbose=False), "cpu",
              ckpt_path=tmp_path / "m.pt", seed=1)
    hist = res["history"]
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert np.isclose(res["best_val_loss"], min(h["val_loss"] for h in hist))
    from src.models.train_rnn import evaluate_loss
    assert np.isclose(evaluate_loss(model, fw, "cpu"), res["best_val_loss"], rtol=1e-4)
    assert (tmp_path / "m.pt").exists()
    assert predict_z(model, fw, "cpu").shape == (len(fw), H)


def test_seed_reproducible(ws):
    fw = FastWindows([ws])
    outs = []
    for _ in range(2):
        model = DirectRNN(fw.n_features, H, "lstm", 8, 1, 0.0)
        torch.manual_seed(0)
        model = DirectRNN(fw.n_features, H, "lstm", 8, 1, 0.0)
        fit(model, fw, fw, TrainConfig(max_epochs=2, verbose=False), "cpu", seed=7)
        outs.append(predict_z(model, fw, "cpu"))
    np.testing.assert_allclose(outs[0], outs[1], rtol=1e-5)


def test_seq2seq_teacher_forcing_only_in_training(ws):
    fw = FastWindows([ws])
    model = Seq2SeqRNN(fw.n_features, H, "gru", 8, 1, 0.0)
    x, y, m, _ = next(batches(fw, 16))
    model.eval()
    a = model(x, y=y, m=m, tf_ratio=1.0)
    b = model(x)
    assert torch.allclose(a, b), "khi eval không được dùng giá trị thật (teacher forcing)"


@pytest.mark.parametrize("arch", ["direct", "seq2seq"])
def test_residual_zero_head_equals_persistence(arch):
    """residual=True và lớp cuối = 0 -> dự báo mọi tầm = PM2.5 (z) tại gốc t (đúng persistence)."""
    m = build_model(arch, n_features=3, H=H, cell="gru", hidden=8, layers=1, dropout=0.0, residual=True)
    lin = m.head[-1]
    torch.nn.init.zeros_(lin.weight)
    torch.nn.init.zeros_(lin.bias)
    x = torch.randn(5, L, 3)
    out = m.eval()(x)
    assert torch.allclose(out, x[:, -1, :1].expand(-1, H))
    m2 = build_model(arch, n_features=3, H=H, cell="gru", hidden=8, layers=1, dropout=0.0, residual=False)
    torch.nn.init.zeros_(m2.head[-1].weight)
    torch.nn.init.zeros_(m2.head[-1].bias)
    assert torch.allclose(m2.eval()(x), torch.zeros(5, H))
