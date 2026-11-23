"""G1-29, G1-30, G1-31, G1-32 — Tải và kiểm tra khí tượng theo giờ từ Open-Meteo Historical Weather API.

Cách chạy (từ gốc repo):
    python -m src.data.fetch_openmeteo                    # tải cả 2 thành phố (bỏ qua trạm đã có cache) + kiểm tra
    python -m src.data.fetch_openmeteo --city hcm         # chỉ TP.HCM (G1-30)
    python -m src.data.fetch_openmeteo --city beijing     # chỉ Bắc Kinh (G1-31)
    python -m src.data.fetch_openmeteo --force            # tải lại toàn bộ
    python -m src.data.fetch_openmeteo --verify           # chỉ kiểm tra cache đã có (G1-32)

Mọi tham số (toạ độ, khoảng ngày, timezone, danh sách biến, đơn vị gió) nằm trong configs/openmeteo.yaml.
Đầu ra:
  data/raw/openmeteo/<city>_<station_id>.json   phản hồi JSON THÔ của API, không chỉnh sửa (cache)
  data/raw/openmeteo/DOWNLOAD_INFO.json         ngày tải, URL request, ô lưới thật, md5 NỘI DUNG
  reports/meteo/g1_32_check_log.md              log kiểm tra: đủ giờ, không trùng, không thiếu, đơn vị đúng
Tái lập: md5 tính trên dữ liệu "hourly" (không tính generationtime_ms thay đổi mỗi lần gọi), nên tải lại
phải ra cùng md5. Lưu ý: ERA5 có thể được ECMWF hiệu chỉnh nhỏ theo thời gian -> luôn ghi lại md5 và ngày tải.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time

import pandas as pd
import requests

from src.data.openmeteo_common import ROOT, load_om_config, raw_json, rpath


def build_params(cfg: dict, city: str, lat: float, lon: float) -> dict:
    c = cfg["cities"][city]
    return {"latitude": lat, "longitude": lon, "start_date": c["start_date"], "end_date": c["end_date"],
            "hourly": ",".join(cfg["hourly_vars"]), "timezone": c["timezone"], "wind_speed_unit": "ms",
            "models": cfg["model"]}


def fetch(cfg: dict, params: dict, retries: int = 6) -> dict:
    """Gọi API, thử lại khi lỗi mạng / 429 (vượt giới hạn) / 5xx."""
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(cfg["api_url"], params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:150]}"
            wait = 65 if r.status_code == 429 else 5 * 2 ** attempt
        except requests.RequestException as e:
            last, wait = repr(e), 5 * 2 ** attempt
        print(f"   lỗi ({last}) -> thử lại sau {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"Không tải được sau {retries} lần: {last}")


def content_md5(js: dict) -> str:
    return hashlib.md5(json.dumps(js["hourly"], sort_keys=True).encode()).hexdigest()


def check_station(cfg: dict, city: str, sid: str, js: dict) -> dict:
    """G1-32: kiểm tra một trạm. Trả về dict kết quả (ok = True nếu qua hết)."""
    c = cfg["cities"][city]
    h = js["hourly"]
    t = pd.to_datetime(pd.Series(h["time"]))
    exp_start, exp_end = pd.Timestamp(c["start_date"]), pd.Timestamp(c["end_date"]) + pd.Timedelta(hours=23)
    n_exp = len(pd.date_range(exp_start, exp_end, freq="h"))
    units = js.get("hourly_units", {})
    bad_units = {v: units.get(v) for v, u in cfg["expected_units"].items() if units.get(v) != u}
    n_missing = {v: sum(x is None for x in h[v]) for v in cfg["hourly_vars"]}
    df = pd.DataFrame({v: h[v] for v in cfg["hourly_vars"]}, dtype=float)
    out_of_range = {v: int(((df[v] < lo) | (df[v] > hi)).sum())
                    for v, (lo, hi) in cfg["valid_range"].items() if v in df}
    res = {
        "city": city, "station_id": sid, "rows": len(t), "rows_expected": n_exp,
        "start_ok": t.iloc[0] == exp_start, "end_ok": t.iloc[-1] == exp_end,
        "hourly_step_ok": bool((t.diff().dropna() == pd.Timedelta(hours=1)).all()),
        "n_duplicated": int(t.duplicated().sum()), "n_missing_total": sum(n_missing.values()),
        "units_ok": not bad_units, "bad_units": bad_units,
        "n_out_of_range": sum(out_of_range.values()), "out_of_range": {k: v for k, v in out_of_range.items() if v},
        "timezone": js.get("timezone"), "tz_ok": js.get("timezone") == c["timezone"],
        "grid_lat": js.get("latitude"), "grid_lon": js.get("longitude"),
    }
    res["ok"] = (res["rows"] == n_exp and res["start_ok"] and res["end_ok"] and res["hourly_step_ok"]
                 and res["n_duplicated"] == 0 and res["n_missing_total"] == 0 and res["units_ok"]
                 and res["n_out_of_range"] == 0 and res["tz_ok"])
    return res


def write_check_log(cfg: dict, results: list[dict]) -> None:
    out = rpath(cfg, "report_dir") / "g1_32_check_log.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    cols = ["city", "station_id", "rows", "rows_expected", "start_ok", "end_ok", "hourly_step_ok",
            "n_duplicated", "n_missing_total", "units_ok", "n_out_of_range", "tz_ok", "grid_lat", "grid_lon", "ok"]
    units = "\n".join(f"| `{v}` | {u} |" for v, u in cfg["expected_units"].items())
    ranges = "\n".join(f"| `{v}` | {lo} | {hi} |" for v, (lo, hi) in cfg["valid_range"].items())
    grid = (df.groupby(["city", "grid_lat", "grid_lon"])["station_id"].apply(", ".join).reset_index())
    lines = [
        "# G1-32. Log kiểm tra dữ liệu Open-Meteo tải về", "",
        f"Sinh tự động bởi `python -m src.data.fetch_openmeteo --verify` lúc {dt.datetime.now():%Y-%m-%d %H:%M}.", "",
        "## Tiêu chí", "",
        "- **Đủ giờ:** số dòng = số giờ từ `start_date 00:00` đến `end_date 23:00`; bước thời gian luôn đúng 1 giờ.",
        "- **Không trùng:** không có mốc thời gian lặp.",
        "- **Không thiếu:** không có ô `null` ở 9 biến.",
        "- **Đơn vị đúng** (G1-28, đối chiếu tài liệu Historical Weather API, gió đặt `wind_speed_unit=ms`):", "",
        "| Biến | Đơn vị mong đợi |", "|---|---|", units, "",
        "- **Giá trị hợp lý về vật lý:**", "", "| Biến | Min | Max |", "|---|---|---|", ranges, "",
        "## Kết quả từng trạm", "", df[cols].to_markdown(index=False), "",
        f"**Tổng kết: {int(df['ok'].sum())}/{len(df)} trạm đạt mọi tiêu chí.**", "",
        "## Ô lưới ERA5 dùng chung", "",
        "Nhiều trạm gần nhau rơi vào cùng một ô lưới nên có khí tượng giống hệt nhau:", "",
        grid.to_markdown(index=False), "",
    ]
    if not df["ok"].all():
        bad = df[~df["ok"]][["city", "station_id", "bad_units", "out_of_range"]]
        lines += ["## Trạm KHÔNG đạt", "", bad.to_markdown(index=False), ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Log kiểm tra: {out.relative_to(ROOT)}  ({int(df['ok'].sum())}/{len(df)} trạm đạt)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=["hcm", "beijing", "all"], default="all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify", action="store_true", help="chỉ kiểm tra cache, không tải")
    args = ap.parse_args()

    cfg = load_om_config()
    out_dir = rpath(cfg, "raw_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    info_path = out_dir / "DOWNLOAD_INFO.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    cities = ["hcm", "beijing"] if args.city == "all" else [args.city]

    results = []
    for city in cities:
        for sid, st in cfg["cities"][city]["stations"].items():
            f = raw_json(cfg, city, sid)
            if not args.verify and (args.force or not f.exists()):
                params = build_params(cfg, city, st["lat"], st["lon"])
                print(f"[tải] {city}/{sid} ({st['lat']}, {st['lon']})")
                js = fetch(cfg, params)
                f.write_text(json.dumps(js, ensure_ascii=False), encoding="utf-8")
                info[f"{city}/{sid}"] = {
                    "downloaded_at": dt.datetime.now().isoformat(timespec="seconds"),
                    "request": {"url": cfg["api_url"], **params},
                    "grid_lat": js.get("latitude"), "grid_lon": js.get("longitude"),
                    "elevation": js.get("elevation"), "timezone": js.get("timezone"),
                    "utc_offset_seconds": js.get("utc_offset_seconds"), "rows": len(js["hourly"]["time"]),
                    "content_md5": content_md5(js),
                }
                info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
                time.sleep(1.0)  # lịch sự với API miễn phí
            if not f.exists():
                raise FileNotFoundError(f"Chưa có {f.name}; chạy lại không kèm --verify")
            js = json.loads(f.read_text(encoding="utf-8"))
            key = f"{city}/{sid}"
            if key in info and info[key].get("content_md5") != content_md5(js):
                raise ValueError(f"{f.name}: md5 nội dung khác DOWNLOAD_INFO -> file thô đã bị sửa?")
            results.append(check_station(cfg, city, sid, js))

    info["_source"] = {"api": cfg["api_url"], "doc": "https://open-meteo.com/en/docs/historical-weather-api",
                       "license": "CC BY 4.0 (Open-Meteo); dữ liệu nền ERA5/ERA5-Land (ECMWF)"}
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    write_check_log(cfg, results)
    if not all(r["ok"] for r in results):
        raise SystemExit("Có trạm không đạt kiểm tra, xem log.")


if __name__ == "__main__":
    main()
