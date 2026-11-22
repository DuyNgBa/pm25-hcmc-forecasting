"""Tiện ích dùng chung: đọc config, cố định seed, đường dẫn dự án."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

# Thư mục gốc của repo (src/.. ). Mọi đường dẫn trong config là tương đối so với đây.
ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    """Đọc file YAML. Đường dẫn tương đối được hiểu là tính từ gốc repo.

    Mặc định configs/data.yaml; có thể đổi bằng biến môi trường PM25_CONFIG (dùng cho test).
    """
    p = Path(path or os.environ.get("PM25_CONFIG", "configs/data.yaml"))
    if not p.is_absolute():
        p = ROOT / p
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path_of(cfg: dict, key: str) -> Path:
    """Lấy đường dẫn tuyệt đối của một mục trong cfg['paths']."""
    return ROOT / cfg["paths"][key]


def results_file(cfg: dict, method: str) -> Path:
    """Quy ước của nhóm (README repo): mỗi phương pháp một file results/<tên-phương-pháp>.csv, cùng mẫu cột.

    Ví dụ: results_file(cfg, "patchtst") -> results/patchtst.csv
    """
    return ROOT / cfg["paths"]["results_dir"] / f"{method}.csv"


def seed_everything(seed: int = 42) -> None:
    """Cố định mọi nguồn ngẫu nhiên để chạy lại ra cùng kết quả.

    Lưu ý: trên GPU vẫn có thể lệch rất nhỏ do thuật toán cuDNN không tất định;
    vì vậy ta chạy 3 seed và báo cáo trung bình ± độ lệch chuẩn.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # torch không bắt buộc cho phần xử lý dữ liệu
        pass
