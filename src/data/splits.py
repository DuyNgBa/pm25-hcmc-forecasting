"""G1-48, G1-49, G1-50, G1-56 — Mốc chia dữ liệu, fold rolling-origin, lấy mẫu con train.

Mọi mốc đọc từ configs/data.yaml (nguồn sự thật duy nhất). KHÔNG hard-code ngày trong notebook.

Ví dụ:
    from src.data.splits import get_split_ranges
    rng = get_split_ranges(cfg, "hcm")             # split chính
    rng = get_split_ranges(cfg, "hcm", fold=2)     # fold rolling-origin số 2
    rng = get_split_ranges(cfg, "hcm", train_frac=0.25)   # train chỉ còn 25% (đoạn liên tục)
    rng["train"] -> (Timestamp start, Timestamp end), inclusive
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Range = tuple[pd.Timestamp, pd.Timestamp]


def _to_range(pair) -> Range:
    return pd.Timestamp(pair[0]), pd.Timestamp(pair[1])


def get_split_ranges(cfg: dict, city: str, fold: int | None = None,
                     train_frac: float = 1.0, mode: str | None = None, seed: int | None = None) -> dict[str, Range]:
    """Trả về dict {'train': (s, e), 'val': (s, e), ['test': (s, e)]}.

    fold=None -> split chính; fold=k -> fold rolling-origin thứ k (chỉ TP.HCM).
    train_frac < 1 -> thu hẹp train thành 1 đoạn liên tục chiếm train_frac số giờ (G1-56).
    """
    if fold is None:
        rng = {k: _to_range(v) for k, v in cfg["splits"][city].items()}
    else:
        folds = {f["fold"]: f for f in cfg["rolling_origin"][city]}
        if fold not in folds:
            raise KeyError(f"Không có fold {fold} cho {city}; có: {sorted(folds)}")
        rng = {k: _to_range(folds[fold][k]) for k in ("train", "val", "test")}
    if train_frac < 1.0:
        mode = mode or cfg["train_subsample"]["mode"]
        seed = cfg["seed"] if seed is None else seed
        rng["train"] = subsample_train_range(rng["train"], train_frac, mode, seed)
    return rng


def subsample_train_range(train: Range, frac: float, mode: str = "last", seed: int = 42) -> Range:
    """G1-56: lấy một ĐOẠN THỜI GIAN LIÊN TỤC chiếm `frac` tổng số giờ của train.

    Vì sao liên tục mà không lấy ngẫu nhiên từng giờ: mô hình chuỗi cần cửa sổ liền mạch,
    và kịch bản thật ("tỉnh mới lắp trạm") là có ÍT THÁNG dữ liệu chứ không phải dữ liệu lỗ chỗ.
      mode="last"  : đoạn cuối của train (sát val) — mặc định, tất định.
      mode="random": điểm bắt đầu chọn ngẫu nhiên bằng seed (cố định giữa các thành viên).
    """
    if not 0 < frac <= 1:
        raise ValueError("frac phải trong (0, 1]")
    s, e = train
    hours = pd.date_range(s, e, freq="h")
    n_keep = int(round(len(hours) * frac))
    if mode == "last":
        start_idx = len(hours) - n_keep
    elif mode == "random":
        start_idx = int(np.random.default_rng(seed).integers(0, len(hours) - n_keep + 1))
    else:
        raise ValueError(f"mode không hợp lệ: {mode}")
    return hours[start_idx], hours[start_idx + n_keep - 1]


def in_range(times: pd.Series | pd.DatetimeIndex, rng: Range) -> np.ndarray:
    t = pd.DatetimeIndex(times)
    return np.asarray((t >= rng[0]) & (t <= rng[1]))


def plot_folds(cfg: dict, out_path, city: str = "hcm") -> None:
    """Hình minh hoạ các fold rolling-origin (G1-50)."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    colors = {"train": "#2a78d6", "val": "#eda100", "test": "#1baf7a"}
    rows = [("Split chính", get_split_ranges(cfg, city))]
    rows += [(f"Fold {f['fold']}", get_split_ranges(cfg, city, fold=f["fold"])) for f in cfg["rolling_origin"][city]]
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(rows) + 1.2))
    for i, (name, rng) in enumerate(rows[::-1]):
        for part, (s, e) in rng.items():
            ax.barh(i, (e - s).total_seconds() / 86400, left=mdates.date2num(s), color=colors[part],
                    edgecolor="white", height=0.6, label=part if i == 0 else None)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.set_title("TP.HCM: split chính và 4 fold rolling-origin (expanding window)")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(k) for k in ("train", "val", "test")]
    ax.legend([handles[i] for i in order], ["train", "val", "test"], loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)   # tự tạo thư mục nếu chưa có
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    from src.utils import ROOT, load_config

    cfg = load_config()
    out = ROOT / "reports" / "figures" / "rolling_origin_folds.png"
    plot_folds(cfg, out)
    for f in [None, 1, 2, 3, 4]:
        r = get_split_ranges(cfg, "hcm", fold=f)
        print("main" if f is None else f"fold{f}", {k: (str(a.date()), str(b.date())) for k, (a, b) in r.items()})
    print("train 25%:", get_split_ranges(cfg, "hcm", train_frac=0.25)["train"])
    print(f"Đã lưu {out}")
