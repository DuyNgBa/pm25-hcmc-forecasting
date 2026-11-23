"""G2-27 — Dataset / DataLoader PyTorch cho LSTM/GRU, xây trên module cửa sổ CHUNG của nhóm (src/data/windows.py).

    from src.data.pipeline import prepare
    from src.data.windows import feature_columns
    from src.models.rnn_data import FastWindows, batches

    P = prepare("hcm")
    ws = P.windows("train", feature_columns("c"), L=48)       # WindowSet chung (chỉ lưu chỉ số)
    fw = FastWindows([ws])                                    # gom các chuỗi thành mảng liền -> lấy batch 1 phép
    for x, y, m, f in batches(fw, batch_size=256, shuffle=True, rng=np.random.default_rng(42)):
        ...   # x (B, L, F), y (B, H) PM2.5 dạng z, m (B, H) mục tiêu là giá trị ĐO THẬT (hoặc có giá trị), f tương lai/None

Vì sao không dùng thẳng WindowSet[i] + torch DataLoader: Bắc Kinh có ~300k cửa sổ; lấy từng mẫu bằng Python
rất chậm. FastWindows nối mọi chuỗi (city, station) thành MỘT mảng (T_tổng, F) và nhớ vị trí gốc t của
từng cửa sổ, nên một batch = một phép fancy-indexing numpy. Kết quả GIỐNG HỆT WindowSet[i] (có test).
Không tạo cửa sổ vắt qua 2 trạm vì chỉ số gốc t lấy nguyên từ WindowSet (đã lọc trong từng trạm).
"""
from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
import torch

from src.data.windows import WindowSet


class FastWindows:
    """Gộp một hoặc nhiều WindowSet (cùng L, H, cùng danh sách đặc trưng) để lấy batch nhanh.

    extra_const: danh sách hằng số, mỗi WindowSet một giá trị, được thêm như MỘT kênh đặc trưng cuối
                 (dùng cho S3: kênh is_hcm = 1 với TP.HCM, 0 với Bắc Kinh).
    """

    def __init__(self, wsets: Sequence[WindowSet], extra_const: Sequence[float] | None = None):
        if not wsets:
            raise ValueError("cần ít nhất 1 WindowSet")
        L, H = wsets[0].L, wsets[0].H
        n_feat = len(wsets[0].feature_cols)
        for w in wsets:
            if (w.L, w.H, len(w.feature_cols)) != (L, H, n_feat):
                raise ValueError("Các WindowSet phải cùng L, H và số đặc trưng")
        self.L, self.H = L, H
        self.feature_cols = list(wsets[0].feature_cols) + (["is_hcm"] if extra_const is not None else [])
        has_fut = wsets[0].future_cols is not None
        Xs, Ys, Fs, gs, self.set_id = [], [], [], [], []
        off = 0
        for k, w in enumerate(wsets):
            starts = []
            for s in w.series:
                X = s["X"]
                if extra_const is not None:
                    X = np.concatenate([X, np.full((len(X), 1), extra_const[k], np.float32)], axis=1)
                Xs.append(X.astype(np.float32))
                Ys.append(s["y_z"].astype(np.float32))
                if has_fut:
                    Fs.append(s["F"].astype(np.float32))
                starts.append(off)
                off += len(X)
            starts = np.asarray(starts, np.int64)
            if len(w):
                gs.append(starts[w.origins[:, 0]] + w.origins[:, 1])
                self.set_id.append(np.full(len(w), k, np.int16))
        self.X = np.concatenate(Xs)
        self.Yz = np.concatenate(Ys)
        self.Fut = np.concatenate(Fs) if has_fut else None
        self.g = np.concatenate(gs) if gs else np.zeros(0, np.int64)
        self.set_id = np.concatenate(self.set_id) if self.set_id else np.zeros(0, np.int16)
        self._xi = np.arange(-L + 1, 1)
        self._yi = np.arange(1, H + 1)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def fut_dim(self) -> int:
        return 0 if self.Fut is None else self.Fut.shape[1]

    def __len__(self) -> int:
        return len(self.g)

    def get(self, idx: np.ndarray):
        """Batch theo chỉ số cửa sổ: x (B,L,F), y (B,H) [NaN -> 0], m (B,H) bool, f (B,H,Ff) hoặc None."""
        g = self.g[idx]
        x = self.X[g[:, None] + self._xi]
        yi = g[:, None] + self._yi
        y = self.Yz[yi]
        m = ~np.isnan(y)
        y = np.nan_to_num(y, nan=0.0)
        f = self.Fut[yi] if self.Fut is not None else None
        return x, y, m, f


def balanced_weights(fw: FastWindows, shares: Sequence[float]) -> np.ndarray:
    """Trọng số lấy mẫu để mỗi WindowSet k chiếm tỉ lệ shares[k] trong một epoch (S3: cân bằng 2 thành phố)."""
    w = np.zeros(len(fw))
    for k, sh in enumerate(shares):
        sel = fw.set_id == k
        if sel.any():
            w[sel] = sh / sel.sum()
    return w / w.sum()


def batches(fw: FastWindows, batch_size: int, shuffle: bool = False, rng: np.random.Generator | None = None,
            weights: np.ndarray | None = None, num_samples: int | None = None,
            device: str | torch.device = "cpu") -> Iterator[tuple]:
    """Sinh batch tensor. weights + num_samples: lấy mẫu CÓ hoàn lại theo trọng số (cân bằng S3)."""
    n = len(fw)
    if weights is not None:
        rng = rng or np.random.default_rng()
        order = rng.choice(n, size=num_samples or n, replace=True, p=weights)
    elif shuffle:
        rng = rng or np.random.default_rng()
        order = rng.permutation(n)
    else:
        order = np.arange(n)
    for i in range(0, len(order), batch_size):
        x, y, m, f = fw.get(order[i:i + batch_size])
        yield (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device), torch.from_numpy(m).to(device),
               None if f is None else torch.from_numpy(f).to(device))
