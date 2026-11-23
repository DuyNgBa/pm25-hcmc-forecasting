"""G2-29, G2-30, G2-31 — Mô hình LSTM / GRU dự báo PM2.5 nhiều bước.

Hai kiến trúc, mỗi cái chạy được với cell = "lstm" hoặc "gru":

1) DirectRNN (G2-29 LSTM, G2-30 GRU) — dự báo TRỰC TIẾP 24 bước:
       x (B, L, F) --encoder RNN--> h_L (trạng thái ẩn cuối) --head Linear--> ŷ (B, 24)
   Một lần chạy cho cả 24 tầm, không tích luỹ sai số.

2) Seq2SeqRNN (G2-31) — encoder–decoder:
       encoder RNN đọc x -> trạng thái (h, c) -> decoder RNN chạy 24 bước;
       đầu vào bước k = dự báo ở bước k-1 (bước đầu = PM2.5 tại t). Khi train có teacher forcing
       (dùng giá trị thật ở bước trước với xác suất tf_ratio, chỉ khi giá trị đó có thật).
   Dự báo tuần tự nên có thể tích luỹ sai số; đổi lại decoder "nhìn thấy" quỹ đạo vừa dự báo.

Thiết lập phụ "dự báo thời tiết hoàn hảo" (G2-48): fut (B, 24, Ff) = khí tượng thật tại t+1..t+24.
  - DirectRNN: nén fut bằng 1 lớp Linear rồi nối vào h_L trước head.
  - Seq2SeqRNN: nối fut[:, k] vào đầu vào decoder ở bước k.

Kết nối phần dư (residual=True, mặc định): mô hình dự báo ĐỘ THAY ĐỔI so với PM2.5 tại gốc t thay vì mức tuyệt đối:
  - DirectRNN : ŷ(t+k) = y(t) + Δk,   Δ = head(h_L)
  - Seq2SeqRNN: ŷ(t+k) = ŷ(t+k-1) + δk  (mỗi bước cộng dồn từ giá trị trước)
  Lý do (xem reports/rnn/summary.md, mục "Ghi chú thiết kế"): bản không residual overfit sau 2–3 epoch và thua
  persistence ở t+1. Với residual, "đoán bằng giá trị hiện tại" là điểm xuất phát tự nhiên (Δ = 0) nên mô hình chỉ
  phải học phần lệch khỏi persistence. Đầu vào không đổi -> so sánh với mô hình khác vẫn công bằng.
  PM2.5 (z) tại t lấy từ x[:, -1, target_index]; target_index = 0 vì "pm25_z" luôn là đặc trưng đầu tiên.

Fine-tune "chỉ các lớp cuối" (RQ3 S2_head, G2-43): đóng băng `encoder`, chỉ train `head_parameters()`.
"""
from __future__ import annotations

import random

import torch
from torch import nn

RNN_CELLS = {"lstm": nn.LSTM, "gru": nn.GRU}


def _rnn(cell: str, n_in: int, hidden: int, layers: int, dropout: float) -> nn.Module:
    return RNN_CELLS[cell](n_in, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)


class DirectRNN(nn.Module):
    def __init__(self, n_features: int, H: int = 24, cell: str = "lstm", hidden: int = 64, layers: int = 2,
                 dropout: float = 0.1, fut_dim: int = 0, residual: bool = True, target_index: int = 0):
        super().__init__()
        self.cell, self.H, self.residual, self.target_index = cell, H, residual, target_index
        self.encoder = _rnn(cell, n_features, hidden, layers, dropout)
        self.fut_proj = nn.Sequential(nn.Linear(H * fut_dim, hidden), nn.ReLU()) if fut_dim else None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * (2 if fut_dim else 1), H))

    def forward(self, x, fut=None, **_):
        out, _ = self.encoder(x)
        h = out[:, -1]
        if self.fut_proj is not None:
            h = torch.cat([h, self.fut_proj(fut.flatten(1))], dim=1)
        out = self.head(h)
        if self.residual:
            out = out + x[:, -1, self.target_index:self.target_index + 1]
        return out

    def head_parameters(self):
        mods = [self.head] + ([self.fut_proj] if self.fut_proj is not None else [])
        return [p for m in mods for p in m.parameters()]


class Seq2SeqRNN(nn.Module):
    def __init__(self, n_features: int, H: int = 24, cell: str = "lstm", hidden: int = 64, layers: int = 2,
                 dropout: float = 0.1, fut_dim: int = 0, target_index: int = 0, residual: bool = True):
        super().__init__()
        self.cell, self.H, self.target_index, self.residual = cell, H, target_index, residual
        self.encoder = _rnn(cell, n_features, hidden, layers, dropout)
        self.decoder = _rnn(cell, 1 + fut_dim, hidden, layers, dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, fut=None, y=None, m=None, tf_ratio: float = 0.0):
        _, state = self.encoder(x)
        prev = x[:, -1, self.target_index:self.target_index + 1]          # PM2.5 (z) tại gốc t
        outs = []
        for k in range(self.H):
            inp = prev if fut is None else torch.cat([prev, fut[:, k]], dim=1)
            o, state = self.decoder(inp.unsqueeze(1), state)
            yk = self.head(o[:, 0])                                       # (B, 1)
            if self.residual:
                yk = prev + yk
            outs.append(yk)
            prev = yk
            if self.training and y is not None and tf_ratio > 0 and random.random() < tf_ratio:
                # teacher forcing: dùng giá trị thật nếu có, nếu không giữ dự báo của mô hình
                prev = torch.where(m[:, k:k + 1], y[:, k:k + 1], yk)
        return torch.cat(outs, dim=1)

    def head_parameters(self):
        return [p for m in (self.decoder, self.head) for p in m.parameters()]


def build_model(arch: str, n_features: int, H: int, cell: str, hidden: int, layers: int, dropout: float,
                fut_dim: int = 0, residual: bool = True) -> nn.Module:
    if arch == "direct":
        return DirectRNN(n_features, H, cell, hidden, layers, dropout, fut_dim, residual=residual)
    if arch == "seq2seq":
        return Seq2SeqRNN(n_features, H, cell, hidden, layers, dropout, fut_dim, residual=residual)
    raise ValueError(f"arch không hợp lệ: {arch}")


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
