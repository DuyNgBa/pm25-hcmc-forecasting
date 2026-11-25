"""G0-11 — Tải Beijing Multi-Site Air Quality từ UCI (DOI 10.24432/C5RK5G) vào data/raw/beijing.

Cách chạy (từ gốc repo):
    python -m src.data.download_beijing            # tải + giải nén + kiểm tra
    python -m src.data.download_beijing --verify   # chỉ kiểm tra 12 file đã có

Sau khi tải, script ghi data/raw/beijing/DOWNLOAD_INFO.json (ngày tải, nguồn, md5)
để biết chính xác phiên bản dữ liệu nhóm dùng. File gốc KHÔNG được chỉnh sửa.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pandas as pd

from src.utils import ROOT, load_config

UCI_URL = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
STATIONS = ["Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan", "Gucheng",
            "Huairou", "Nongzhanguan", "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong"]
EXPECTED_ROWS_PER_STATION = 35064   # 01/03/2013 00h -> 28/02/2017 23h, theo giờ
# "Dấu vân tay" NỘI DUNG của từng trạm: (số dòng, số giờ có PM2.5, tổng PM2.5).
# Không dùng md5 của file vì md5 đổi khi chỉ khác kiểu xuống dòng (LF/CRLF) dù số liệu y hệt.
EXPECTED_CONTENT = {
    "Aotizhongxin": (35064, 34139, 2825808.3), "Changping": (35064, 34290, 2438010.2),
    "Dingling": (35064, 34285, 2262449.9), "Dongsi": (35064, 34314, 2957671.1),
    "Guanyuan": (35064, 34448, 2856888.8), "Gucheng": (35064, 34418, 2886021.2),
    "Huairou": (35064, 34111, 2375025.0), "Nongzhanguan": (35064, 34436, 2921498.0),
    "Shunyi": (35064, 34151, 2714717.7), "Tiantan": (35064, 34387, 2825404.8),
    "Wanliu": (35064, 34682, 2891601.9), "Wanshouxigong": (35064, 34368, 2922109.5),
}


def fname(station: str) -> str:
    return f"PRSA_Data_{station}_20130301-20170228.csv"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(out_dir: Path) -> None:
    import requests  # chỉ cần khi tải

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Đang tải {UCI_URL} ...")
    r = requests.get(UCI_URL, timeout=120)
    r.raise_for_status()
    # File zip của UCI chứa một zip con / thư mục PRSA_Data_20130301-20170228/
    def extract(zbytes: bytes):
        with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
            for name in z.namelist():
                base = Path(name).name
                if base.startswith("PRSA_Data_") and base.endswith(".csv"):
                    (out_dir / base).write_bytes(z.read(name))
                elif base.endswith(".zip"):
                    extract(z.read(name))
    extract(r.content)


def verify(out_dir: Path) -> dict:
    """Kiểm tra đủ 12 file, đúng số dòng, và NỘI DUNG PM2.5 khớp bản nhóm dùng. Trả về thông tin để ghi log."""
    info, all_ok = {}, True
    for st in STATIONS:
        p = out_dir / fname(st)
        assert p.exists(), f"Thiếu file {p.name}"
        d = pd.read_csv(p, usecols=["No", "PM2.5"])
        got = (len(d), int(d["PM2.5"].notna().sum()), round(float(d["PM2.5"].sum()), 1))
        exp = EXPECTED_CONTENT[st]
        ok = got[0] == exp[0] and got[1] == exp[1] and abs(got[2] - exp[2]) < 0.5
        all_ok &= ok
        if not ok:
            print(f"[CẢNH BÁO] {st}: nội dung khác bản nhóm dùng (có {got}, kỳ vọng {exp}) -> báo TV3")
        info[st] = {"rows": got[0], "pm25_count": got[1], "pm25_sum": got[2],
                    "content_match": ok, "md5": md5(p)}
    assert all(v["rows"] == EXPECTED_ROWS_PER_STATION for v in info.values()), "Sai số dòng"
    if all_ok:
        print("Nội dung 12 file khớp bản nhóm dùng (số dòng, số giờ có PM2.5, tổng PM2.5).")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="chỉ kiểm tra, không tải")
    args = ap.parse_args()
    cfg = load_config()
    out_dir = ROOT / cfg["paths"]["raw_beijing"]
    if not args.verify:
        download(out_dir)
    info = verify(out_dir)
    meta = {
        "source": "UCI Machine Learning Repository, dataset 501",
        "doi": "10.24432/C5RK5G", "license": "CC BY 4.0", "url": UCI_URL,
        "checked_at": dt.datetime.now().isoformat(timespec="seconds"),
        "files": info,
    }
    (out_dir / "DOWNLOAD_INFO.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"OK: 12 file, mỗi file {EXPECTED_ROWS_PER_STATION} dòng. Đã ghi DOWNLOAD_INFO.json")


if __name__ == "__main__":
    main()
