"""Nguồn gốc của một lần chạy RNN: dữ liệu nào, cấu hình huấn luyện nào, code nào.

Vì sao cần: `data_version` chỉ là một chuỗi do người gõ vào `configs/data.yaml`. Nếu dữ liệu đổi mà quên đổi chuỗi này
(ví dụ đổi nguồn khí tượng sang `era5_seamless` ngày 23/11/2026, `data_version` vẫn là `v0.9-dev`), thì:
  - checkpoint pretrain Bắc Kinh cũ vẫn bị dùng lại (tên file chỉ dựa trên cấu hình + data_version),
  - Optuna nối tiếp các lần thử cũ trong `results/hpo/*.db`,
  - `configs/rnn.yaml` -> `best` được dùng dù được tối ưu trên dữ liệu khác,
  - `rnn_experiments` bỏ qua lần chạy vì "đã có" trong `results/rnn_runs.csv`,
  - `rnn_report` trộn số của hai bộ dữ liệu.

Module này tính DẤU VÂN TAY từ NỘI DUNG dữ liệu (hai bảng lược đồ chung `data/processed/{hcm,beijing}_schema.parquet`)
và từ các mục cấu hình huấn luyện. Dấu vân tay được đưa vào run_id, tên checkpoint, tên nghiên cứu Optuna,
mục `best` và nhật ký `results/rnn_runs.csv`. Dữ liệu đổi một giá trị là dấu vân tay đổi, nên không thể dùng nhầm.

    from src.models.provenance import data_provenance, train_fingerprint
    prov = data_provenance()   # {"data_version", "meteo_model", "raw_fp", "hcm_fp", "beijing_fp", "data_fp"}

Đối chiếu với bản dữ liệu NHÓM ĐÃ KHOÁ (G1-66): nhãn "CHÍNH THỨC" của rnn_report không chỉ dựa vào chuỗi
data_version = "v1.0". Nó yêu cầu file khoá `data/DATA_LOCK.json` (nhóm commit cùng tag data-v1.0) và so từng dấu vân
tay: dữ liệu thô `raw_fp` + hai bảng lược đồ `hcm_fp`, `beijing_fp`. Module này CHỈ ĐỌC file khoá, không tự tạo.
Bản nháp nội dung file khoá (để nhóm xem và tự commit):  python -m src.models.provenance --lock-draft
"""
from __future__ import annotations

import functools
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from src.utils import ROOT, load_config, path_of

FP_LEN = 12
LEGACY_FP = "legacy"          # các lần chạy trước khi có dấu vân tay (v0.9-dev, khí tượng best_match)
KEY = ["city", "station_id", "time"]


def short_hash(obj) -> str:
    s = obj if isinstance(obj, (bytes, bytearray)) else json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.md5(s).hexdigest()[:FP_LEN]


def frame_fingerprint(df: pd.DataFrame, decimals: int = 6) -> str:
    """Dấu vân tay theo NỘI DUNG bảng: không phụ thuộc thứ tự dòng/cột, kiểu dữ liệu hay phiên bản pyarrow.

    Số thực làm tròn `decimals` chữ số để hai máy tính sin/cos lệch ở bit cuối vẫn ra cùng dấu vân tay.
    """
    d = df[sorted(df.columns)].copy()
    for c in d.columns:
        s = d[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            d[c] = s.astype("int64")
        elif pd.api.types.is_bool_dtype(s) or pd.api.types.is_integer_dtype(s):
            d[c] = s.astype("float64")
        elif pd.api.types.is_float_dtype(s):
            d[c] = s.astype("float64").round(decimals)
        else:
            d[c] = s.astype(str)
    for c in d.columns:                       # -0.0 và 0.0 phải như nhau
        if d[c].dtype == "float64":
            d[c] = d[c] + 0.0
    keys = [k for k in KEY if k in d.columns] or list(d.columns)
    d = d.sort_values(keys, kind="mergesort").reset_index(drop=True)
    h = pd.util.hash_pandas_object(d, index=False).to_numpy()
    return hashlib.md5(str(list(d.columns)).encode() + h.tobytes()).hexdigest()[:FP_LEN]


@functools.lru_cache(maxsize=8)
def _file_fp(path: str, mtime: float, size: int) -> str:
    return frame_fingerprint(pd.read_parquet(path))


def file_fingerprint(path: Path) -> str:
    st = Path(path).stat()
    return _file_fp(str(path), st.st_mtime, st.st_size)


def meteo_model() -> str:
    p = ROOT / "configs" / "openmeteo.yaml"
    if not p.exists():
        return "unknown"
    with open(p, encoding="utf-8") as f:
        return str((yaml.safe_load(f) or {}).get("model", "best_match"))


RAW_TEXT = {".csv", ".json", ".txt", ".md"}
RAW_SKIP = {"DOWNLOAD_INFO.json", ".gitkeep"}      # chỉ ghi ngày tải / giữ thư mục, không phải dữ liệu


def _raw_file_digest(f: Path) -> str:
    """sha256 nội dung một file thô. File văn bản chuẩn hoá CRLF -> LF (Windows autocrlf và Kaggle ra cùng kết quả).
    JSON Open-Meteo chỉ băm phần dữ liệu (bỏ generationtime_ms thay đổi mỗi lần gọi API)."""
    b = f.read_bytes()
    if f.suffix == ".json":
        try:
            js = json.loads(b.decode("utf-8"))
            if isinstance(js, dict) and "hourly" in js:
                js = {k: js.get(k) for k in ("latitude", "longitude", "timezone", "hourly_units", "hourly")}
            b = json.dumps(js, sort_keys=True, ensure_ascii=False).encode("utf-8")
        except ValueError:
            pass
    if f.suffix in RAW_TEXT:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


@functools.lru_cache(maxsize=4)
def _raw_fp(root: str, stamp: tuple) -> str:
    base = Path(root)
    items = [(f.relative_to(base).as_posix(), _raw_file_digest(f)) for f in sorted(base.rglob("*"))
             if f.is_file() and f.name not in RAW_SKIP]
    return short_hash(items)


def raw_fingerprint(raw_dir: Path | None = None) -> str:
    """Dấu vân tay toàn bộ dữ liệu thô đã commit (data/raw: HealthyAir, 12 file Bắc Kinh, JSON Open-Meteo)."""
    raw_dir = Path(raw_dir or ROOT / "data" / "raw")
    files = [f for f in sorted(raw_dir.rglob("*")) if f.is_file() and f.name not in RAW_SKIP]
    stamp = tuple((f.as_posix(), f.stat().st_mtime, f.stat().st_size) for f in files)
    return _raw_fp(str(raw_dir), stamp)


def data_provenance(cfg: dict | None = None, cities=("hcm", "beijing")) -> dict:
    """Dấu vân tay dữ liệu đầu vào của mọi thí nghiệm RNN (TP.HCM + Bắc Kinh vì RQ3 pretrain trên Bắc Kinh)."""
    cfg = cfg or load_config()
    out = {"data_version": str(cfg["data_version"]), "meteo_model": meteo_model(), "raw_fp": raw_fingerprint()}
    for c in cities:
        p = path_of(cfg, f"{c}_schema")
        if not p.exists():
            raise FileNotFoundError(f"Chưa có {p.relative_to(ROOT)}: chạy python -m src.data.schema trước")
        out[f"{c}_fp"] = file_fingerprint(p)
    out["data_fp"] = short_hash({k: out[k] for k in sorted(out)})
    return out


LOCK_PATH = ROOT / "data" / "DATA_LOCK.json"
LOCK_KEYS = ("data_version", "raw_fp", "hcm_fp", "beijing_fp")


def load_lock(path: Path | None = None) -> dict | None:
    p = Path(path or LOCK_PATH)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def compare_lock(prov: dict, lock: dict | None) -> list[str]:
    """Danh sách điểm KHÔNG khớp giữa dữ liệu của các lần chạy (prov) và bản khoá của nhóm. Rỗng = khớp."""
    if not lock:
        return [f"chưa có bản khoá dữ liệu của nhóm ({LOCK_PATH.relative_to(ROOT).as_posix()}, G1-66)"]
    out = []
    for k in LOCK_KEYS:
        if k not in lock:
            out.append(f"bản khoá thiếu trường {k}")
        elif str(prov.get(k)) != str(lock[k]):
            out.append(f"{k}: kết quả = {prov.get(k)}, bản khoá = {lock[k]}")
    return out


def lock_draft(cfg: dict | None = None) -> dict:
    """Nội dung đề xuất cho data/DATA_LOCK.json, tính từ dữ liệu local hiện tại. KHÔNG ghi file."""
    p = data_provenance(cfg)
    return {"data_version": p["data_version"], "git_tag": f"data-{p['data_version']}", "meteo_model": p["meteo_model"],
            "raw_fp": p["raw_fp"], "hcm_fp": p["hcm_fp"], "beijing_fp": p["beijing_fp"], "data_fp": p["data_fp"],
            "fingerprint_code": "src/models/provenance.py", "note": "Nhóm tạo khi gắn tag (G1-66); không sửa sau đó."}


def try_data_provenance(cfg: dict | None = None) -> dict | None:
    try:
        return data_provenance(cfg)
    except (FileNotFoundError, KeyError):
        return None


TRAIN_SECTIONS = ("residual", "train", "pretrain", "finetune", "s3")


def train_fingerprint(rcfg: dict) -> str:
    """Các mục của configs/rnn.yaml ảnh hưởng tới cách huấn luyện (không gồm best/seeds/hpo)."""
    return short_hash({k: rcfg.get(k) for k in TRAIN_SECTIONS})


def hpo_fingerprint(rcfg: dict) -> str:
    return short_hash({"hpo": rcfg.get("hpo"), "residual": rcfg.get("residual")})


def code_revision() -> str:
    """Commit git hiện tại (+ "-dirty" nếu có file chưa commit). Không có git thì "unknown"."""
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "src", "configs"], cwd=ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return (rev or "unknown") + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def check_best(rcfg: dict, cell: str, data_fp: str, hpo_fp: str | None = None) -> str | None:
    """Trả về thông báo lỗi nếu best.<cell> không được tối ưu trên ĐÚNG dữ liệu hiện tại, None nếu ổn."""
    best = (rcfg.get("best") or {}).get(cell)
    if not best:
        return f"configs/rnn.yaml chưa có best.{cell}: chạy python -m src.models.hpo_rnn --cell {cell} trước"
    got = best.get("data_fp")
    if got != data_fp:
        return (f"best.{cell} được tối ưu trên dữ liệu data_fp={got or 'không ghi (bản cũ)'}, "
                f"dữ liệu hiện tại là {data_fp}: chạy lại python -m src.models.hpo_rnn --cell {cell}")
    if hpo_fp and best.get("hpo_fp") not in (None, hpo_fp):
        return f"best.{cell} được tối ưu với mục hpo khác trong configs/rnn.yaml: chạy lại hpo_rnn --cell {cell}"
    return None


def runs_with_fp(runs: pd.DataFrame) -> pd.DataFrame:
    """Nhật ký cũ (trước khi có dấu vân tay) được gắn nhãn LEGACY_FP."""
    runs = runs.copy()
    if "data_fp" not in runs:
        runs["data_fp"] = LEGACY_FP
    runs["data_fp"] = runs["data_fp"].fillna(LEGACY_FP).astype(str)
    if "data_version" not in runs:
        runs["data_version"] = None
    return runs


def select_fp(runs: pd.DataFrame, requested: str | None = None, local_fp: str | None = None) -> str:
    """Chọn bộ dữ liệu để tổng hợp/kiểm tra. Không bao giờ trộn hai bộ."""
    fps = list(dict.fromkeys(runs["data_fp"]))
    if requested:
        if requested not in fps:
            raise SystemExit(f"Không có lần chạy nào với data_fp={requested}. Có: {fps}")
        return requested
    if local_fp in fps:
        return local_fp
    if len(fps) == 1:
        return fps[0]
    raise SystemExit(f"results/rnn_runs.csv có {len(fps)} bộ dữ liệu {fps} và không bộ nào khớp dữ liệu local "
                     f"({local_fp}). Chọn bằng --data-fp.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock-draft", action="store_true", help="in bản nháp data/DATA_LOCK.json (không ghi file)")
    a = ap.parse_args()
    if a.lock_draft:
        print(json.dumps(lock_draft(), indent=2, ensure_ascii=False))
    else:
        prov = data_provenance()
        print(json.dumps(prov, indent=2, ensure_ascii=False))
        diff = compare_lock(prov, load_lock())
        print("So với bản khoá của nhóm:", "KHỚP" if not diff else "; ".join(diff))
