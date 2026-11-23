"""G2-28 — Vòng huấn luyện LSTM/GRU: early stopping theo val, lưu checkpoint tốt nhất, seed cố định.

    from src.models.train_rnn import TrainConfig, fit, predict_z, pick_device
    res = fit(model, fw_train, fw_val, TrainConfig(lr=1e-3), device, ckpt_path="checkpoints/x.pt", seed=42)
    z = predict_z(model, fw_test, device)      # (N, H) dự báo dạng z -> đổi ngược bằng scaling.inverse_pm25

Hàm mất mát: MSE trên PM2.5 đã chuẩn hoá (z = (log1p(PM2.5) − μ_trạm) / σ_trạm), CHỈ tính trên các mục tiêu
có giá trị (mask). Early stopping theo cùng hàm mất mát trên tập val; giữ trọng số của epoch tốt nhất.
Chạy được trên CPU, GPU (CUDA) của Colab/Kaggle: thiết bị chọn tự động.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.models.rnn_data import FastWindows, batches
from src.utils import seed_everything


@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 60
    patience: int = 8                 # dừng nếu val không cải thiện sau `patience` epoch
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    lr_patience: int = 3              # giảm LR một nửa nếu val đứng yên `lr_patience` epoch
    tf_ratio: float = 0.5             # teacher forcing ban đầu (chỉ seq2seq), giảm tuyến tính về 0
    epoch_samples: int | None = None  # số mẫu mỗi epoch khi lấy mẫu theo trọng số (S3)
    min_delta: float = 1e-4
    verbose: bool = True
    extra: dict = field(default_factory=dict)


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def masked_mse(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    m = m.float()
    return ((pred - y) ** 2 * m).sum() / m.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate_loss(model: nn.Module, fw: FastWindows, device, batch_size: int = 1024) -> float:
    model.eval()
    tot, cnt = 0.0, 0.0
    for x, y, m, f in batches(fw, batch_size, device=device):
        pred = model(x, fut=f)
        mf = m.float()
        tot += float(((pred - y) ** 2 * mf).sum())
        cnt += float(mf.sum())
    return tot / max(cnt, 1.0)


@torch.no_grad()
def predict_z(model: nn.Module, fw: FastWindows, device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    outs = [model(x, fut=f).cpu().numpy() for x, _, _, f in batches(fw, batch_size, device=device)]
    return np.concatenate(outs) if outs else np.zeros((0, fw.H), np.float32)


def fit(model: nn.Module, fw_train: FastWindows, fw_val: FastWindows, cfg: TrainConfig, device,
        ckpt_path: str | Path | None = None, seed: int = 42, params=None,
        train_weights: np.ndarray | None = None) -> dict:
    """Huấn luyện tới khi early stopping. params: chỉ train các tham số này (fine-tune lớp cuối).

    Trả về dict: best_epoch, best_val_loss, epochs_run, train_sec, history (list loss train/val).
    Trọng số tốt nhất được nạp lại vào model khi kết thúc và lưu vào ckpt_path (nếu có).
    """
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    params = list(params) if params is not None else [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=cfg.lr_patience)
    best, best_state, best_epoch, bad = float("inf"), None, 0, 0
    hist = []
    t0 = time.time()
    for ep in range(1, cfg.max_epochs + 1):
        model.train()
        tf = cfg.tf_ratio * max(0.0, 1 - (ep - 1) / max(cfg.max_epochs * 0.5, 1))
        tot, nb = 0.0, 0
        it = batches(fw_train, cfg.batch_size, shuffle=True, rng=rng, weights=train_weights,
                     num_samples=cfg.epoch_samples, device=device)
        for x, y, m, f in it:
            pred = model(x, fut=f, y=y, m=m, tf_ratio=tf)
            loss = masked_mse(pred, y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            tot += loss.item()
            nb += 1
        val = evaluate_loss(model, fw_val, device)
        sched.step(val)
        hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_loss": val, "lr": opt.param_groups[0]["lr"]})
        if cfg.verbose:
            print(f"  ep {ep:3d}  train {tot / max(nb, 1):.4f}  val {val:.4f}  lr {opt.param_groups[0]['lr']:.1e}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
        if val < best - cfg.min_delta:
            best, best_epoch, bad = val, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    res = {"best_epoch": best_epoch, "best_val_loss": best, "epochs_run": len(hist),
           "train_sec": round(time.time() - t0, 1), "history": hist}
    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "train_config": asdict(cfg), "result": res}, ckpt_path)
    return res
