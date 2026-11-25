"""G2-51, G2-52 — PatchTST (Nie et al., ICLR 2023) cho dự báo PM2.5 đa bước, có khí tượng.

Vì sao TỰ CÀI ĐẶT (≈200 dòng PyTorch) thay vì dùng thư viện?
  - Bám sát kiến trúc của repo gốc (yuqinie98/PatchTST, PatchTST_supervised/layers/PatchTST_backbone.py):
    RevIN, patching có đệm cuối (padding_patch="end"), nhúng patch tuyến tính, vị trí học được (khởi tạo
    U(-0.02, 0.02)), encoder Transformer chuẩn hoá BatchNorm, "residual attention" (cộng điểm attention
    của lớp trước), đầu ra Flatten + Linear.
  - Nhưng cần 3 thứ thư viện không cho sẵn: (1) đưa khí tượng/trạm lân cận vào dự báo PM2.5 (bản gốc
    channel-independent: mỗi kênh tự dự báo chính nó, kênh khác KHÔNG ảnh hưởng); (2) đóng băng backbone
    để fine-tune chỉ đầu ra (chiến lược S2_head); (3) nhúng thành phố cho S3 và khí tượng tương lai (oracle).

Ý tưởng trực giác (cho người mới):
  - Patching: thay vì đưa từng giờ làm một "từ", gom P giờ liên tiếp thành một "từ" (patch). Với L=48,
    P=6, stride=3 -> 16 từ. Mỗi từ mang hình dạng cục bộ (tăng/giảm trong 6 giờ) nên attention giữa các
    từ có ý nghĩa hơn, và chuỗi ngắn hơn -> nhanh hơn, ít overfit hơn với dữ liệu nhỏ.
  - Channel-independence: MỌI kênh (PM2.5, nhiệt độ, gió...) đi qua CÙNG một encoder nhưng riêng rẽ.
    Ta giữ nguyên thiết kế này cho encoder, rồi ở ĐẦU RA mới trộn biểu diễn của các kênh lại
    (head="fusion") để dự báo PM2.5. Đây là lựa chọn thiết kế G2-52, lý do ghi ở docs/notes/patchtst_design.md.

Toán (ký hiệu): x ∈ R^{B×L×C}; kênh 0 là PM2.5 (đã z-score theo trạm).
  1) RevIN chỉ trên kênh 0:  x0' = (x0 - mean_L(x0)) / std_L(x0)   (lưu mean/std để đảo lại ở đầu ra)
  2) Patching mỗi kênh:      [B, C, N, P],  N = floor((L - P)/S) + 2  (đệm cuối S bước)
  3) Nhúng:                  u = W_p · patch + W_pos         -> [B·C, N, d]
  4) Encoder (n lớp):        MultiHeadAttention + FFN, BatchNorm, residual
  5) Head fusion:            ŷ = Linear( flatten_{C,N,d}(z) [⊕ flatten(x_future)] ) ∈ R^H
  6) Đảo RevIN:              ŷ = ŷ · std(x0) + mean(x0)
"""
from __future__ import annotations

import math

import torch
from torch import nn


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., ICLR 2022) cho MỘT kênh (PM2.5).

    Chuẩn hoá theo từng cửa sổ để mô hình không phụ thuộc mức nền; đảo ngược ở đầu ra.
    """

    def __init__(self, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps, self.affine = eps, affine
        if affine:
            self.w = nn.Parameter(torch.ones(1))
            self.b = nn.Parameter(torch.zeros(1))

    def norm(self, x):                           # x: [B, L]
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        x = (x - self.mean) / self.std
        return x * self.w + self.b if self.affine else x

    def denorm(self, y):                         # y: [B, H]
        if self.affine:
            y = (y - self.b) / (self.w + self.eps * self.eps)
        return y * self.std + self.mean


class _BatchNormSeq(nn.Module):
    """BatchNorm trên chiều đặc trưng d cho tensor [B, N, d] (giống Transpose+BatchNorm1d của repo gốc)."""

    def __init__(self, d):
        super().__init__()
        self.bn = nn.BatchNorm1d(d)

    def forward(self, x):
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class EncoderLayer(nn.Module):
    """Một lớp Transformer encoder kiểu PatchTST (post-norm, BatchNorm, residual attention)."""

    def __init__(self, d_model, n_heads, d_ff, dropout, attn_dropout=0.0, norm="batch"):
        super().__init__()
        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads"
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(dropout))
        self.attn_drop = nn.Dropout(attn_dropout)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
        Norm = _BatchNormSeq if norm == "batch" else nn.LayerNorm
        self.norm1, self.norm2 = Norm(d_model), Norm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x, prev_scores=None):     # x: [B', N, d]
        B, N, d = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)  # mỗi cái [B', h, N, dk]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)                         # [B', h, N, N]
        if prev_scores is not None:              # residual attention (RealFormer) như repo gốc
            scores = scores + prev_scores
        attn = self.attn_drop(torch.softmax(scores, dim=-1))
        o = (attn @ v).transpose(1, 2).reshape(B, N, d)
        x = self.norm1(x + self.drop1(self.out(o)))
        x = self.norm2(x + self.drop2(self.ff(x)))
        return x, scores


class PatchTSTX(nn.Module):
    """PatchTST + kênh ngoại sinh (khí tượng, thời gian, trạm lân cận).

    Args:
        n_channels : C, số kênh đầu vào (kênh 0 BẮT BUỘC là PM2.5 z).
        context_len: L. horizon: H.
        patch_len, stride: P, S.
        head: "fusion" (trộn mọi kênh ở đầu ra — mặc định) | "target_only" (đúng PatchTST gốc với 1 kênh,
              các kênh khác bị bỏ qua — dùng để so sánh).
        n_cities: >0 để thêm nhúng thành phố (S3, train chung). n_future: số biến khí tượng tương lai (oracle).
    """

    def __init__(self, n_channels: int, context_len: int, horizon: int = 24, patch_len: int = 6, stride: int = 3,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 3, d_ff: int = 128, dropout: float = 0.2,
                 attn_dropout: float = 0.0, head_dropout: float = 0.1, head: str = "fusion", revin: bool = True,
                 norm: str = "batch", res_attention: bool = True, n_cities: int = 0, n_future: int = 0):
        super().__init__()
        if patch_len > context_len:
            raise ValueError("patch_len phải <= context_len")
        self.C, self.L, self.H, self.P, self.S = n_channels, context_len, horizon, patch_len, stride
        self.n_patches = (context_len - patch_len) // stride + 1 + 1        # +1 do đệm cuối
        self.head_type, self.res_attention = head, res_attention
        self.revin = RevIN() if revin else None
        self.pad = nn.ReplicationPad1d((0, stride))
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.empty(1, self.n_patches, d_model).uniform_(-0.02, 0.02))
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout, attn_dropout, norm)
                                     for _ in range(n_layers)])
        self.city_emb = nn.Embedding(n_cities, d_model) if n_cities > 0 else None
        n_head_ch = n_channels if head == "fusion" else 1
        head_in = n_head_ch * self.n_patches * d_model + n_future * horizon
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(head_in, horizon))
        self.n_future = n_future

    # ---- các nhóm tham số, dùng cho fine-tune chỉ đầu ra (S2_head) ----
    def backbone_parameters(self):
        mods = [self.embed, self.layers] + ([self.revin] if self.revin is not None else [])
        params = [p for m in mods for p in m.parameters()] + [self.pos]
        return params

    def freeze_backbone(self, freeze: bool = True):
        for p in self.backbone_parameters():
            p.requires_grad = not freeze
        # BatchNorm của backbone cũng giữ nguyên thống kê khi đóng băng
        self._frozen = freeze

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_frozen", False):
            for m in self.layers.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()
        return self

    def forward(self, x, x_future=None, city_idx=None):
        """x: [B, L, C] (kênh 0 = PM2.5 z) -> ŷ: [B, H] (z của PM2.5)."""
        B = x.shape[0]
        x = x.permute(0, 2, 1).contiguous()                 # [B, C, L]
        if self.revin is not None:
            x0 = self.revin.norm(x[:, 0, :])
            x = torch.cat([x0.unsqueeze(1), x[:, 1:, :]], dim=1)
        if self.head_type == "target_only":
            x = x[:, :1, :]
        C = x.shape[1]
        x = self.pad(x)                                     # [B, C, L+S]
        p = x.unfold(-1, self.P, self.S)                    # [B, C, N, P]
        u = self.embed(p).reshape(B * C, self.n_patches, -1) + self.pos
        if self.city_emb is not None and city_idx is not None:
            ce = self.city_emb(city_idx)                    # [B, d]
            u = u + ce.repeat_interleave(C, dim=0).unsqueeze(1)
        z = self.drop(u)
        scores = None
        for layer in self.layers:
            z, s = layer(z, scores if self.res_attention else None)
            scores = s
        z = z.reshape(B, -1)                                # flatten C × N × d
        if self.n_future:
            z = torch.cat([z, x_future.reshape(B, -1)], dim=1)
        y = self.head(z)
        if self.revin is not None:
            y = self.revin.denorm(y)
        return y


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
