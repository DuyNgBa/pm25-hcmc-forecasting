"""G1-54, G1-55 — Tạo cửa sổ trượt và đặc trưng trạm lân cận.

Bài toán (mục 5 của đề xuất): tại thời điểm gốc t của trạm s,
    input  = đặc trưng tại t-L+1 .. t          (L = 24 / 48 / 72)
    output = PM2.5 tại t+1 .. t+H              (H = 24; báo cáo chính ở h = 1, 6, 24)

Nguyên tắc:
  - Cắt cửa sổ BÊN TRONG từng (city, station_id): không cửa sổ nào vắt qua 2 trạm/2 thành phố.
  - Cửa sổ thuộc split X nếu MỌI thời điểm mục tiêu nằm trong X. Với train còn yêu cầu input
    cũng nằm trong train (truyền input_start = đầu train) -> không rò rỉ val/test vào train.
  - Không materialize toàn bộ mảng (Bắc Kinh: 12 trạm × 35k giờ × L × F rất lớn) mà lưu
    chỉ số (series_idx, t) và cắt khi cần -> WindowSet dùng trực tiếp làm Dataset cho PyTorch.

Ví dụ:
    feats = feature_columns("c")                          # ablation (c): PM2.5 + Open-Meteo + thời gian
    ws = make_windows(df_s, feats, L=48, H=24, target_range=rng["test"])
    x, y, m = ws[0]            # (L, F), (H,), (H,)  — m: mục tiêu có giá trị (dùng cho loss)
    X, Y, M = ws.materialize() # cho mô hình cây / sklearn
    meta = ws.meta()           # city, station_id, time_origin cho file dự đoán
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TIME_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
ISNA_COLS = ("temp_ha_s", "rh_ha_s")   # cột được thêm kênh cờ thiếu khi nan_policy="drop_target"
METEO_S = ["temp_s", "rh_s", "dew_s", "wind_speed_s", "wind_u_s", "wind_v_s",
           "precip_s", "precip_6h_s", "precip_24h_s", "pressure_s"]


def feature_columns(ablation: str, k_neighbors: int = 5, neighbor_isna: bool = True) -> list[str]:
    """Danh sách đặc trưng cho từng thiết lập ablation (RQ2) — dùng CHUNG cho cả nhóm.

    (a) chỉ PM2.5 quá khứ + đặc trưng thời gian
    (b) (a) + khí tượng gốc HealthyAir (nhiệt độ, độ ẩm)
    (c) (a) + khí tượng Open-Meteo (thay cho b)
    (d) (c) + PM2.5 của 5 trạm lân cận (gần -> xa)
    """
    a = ["pm25_z"] + TIME_COLS
    if ablation == "a":
        return a
    if ablation == "b":
        return a + ["temp_ha_s", "rh_ha_s"]
    if ablation == "c":
        return a + METEO_S
    if ablation == "d":
        nb = [f"pm25_nb{i}" for i in range(1, k_neighbors + 1)]
        nb_na = [f"pm25_nb{i}_isna" for i in range(1, k_neighbors + 1)] if neighbor_isna else []
        return a + METEO_S + nb + nb_na
    raise ValueError(f"ablation không hợp lệ: {ablation}")


# ---------------------------------------------------------------------------
# G1-55: PM2.5 của trạm lân cận
# ---------------------------------------------------------------------------
def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def neighbor_order(stations_cfg: dict) -> dict[str, list[str]]:
    """Với mỗi trạm, danh sách các trạm còn lại sắp theo khoảng cách tăng dần."""
    ids = list(stations_cfg)
    order = {}
    for s in ids:
        d = {o: _haversine_km(stations_cfg[s]["lat"], stations_cfg[s]["lon"],
                              stations_cfg[o]["lat"], stations_cfg[o]["lon"]) for o in ids if o != s}
        order[s] = sorted(d, key=d.get)
    return order


def add_neighbor_pm25(df: pd.DataFrame, stations_cfg: dict, city: str = "hcm", k: int = 5,
                      value_col: str = "pm25_z", enabled: bool = True, add_isna: bool = True) -> pd.DataFrame:
    """Thêm cột pm25_nb1..pm25_nbk = PM2.5 (đã chuẩn hoá) của k trạm gần nhất CÙNG thời điểm t.

    Tham số bật/tắt `enabled` (ablation d). Giá trị thiếu ở trạm lân cận -> điền 0 (= mức trung bình
    sau z-score) và bật cờ pm25_nb{i}_isna = 1, để không phải bỏ cả cửa sổ chỉ vì trạm khác mất dữ liệu.
    Chỉ dùng thông tin tại t và trước t (cửa sổ input), nên không rò rỉ tương lai.
    """
    out = df.copy()
    cols = [f"pm25_nb{i}" for i in range(1, k + 1)]
    if not enabled:
        return out
    m = out["city"] == city
    wide = out.loc[m].pivot(index="time", columns="station_id", values=value_col)
    order = neighbor_order(stations_cfg)
    for c in cols + [c + "_isna" for c in cols]:
        out[c] = np.nan
    for s, nbs in order.items():
        rows = m & (out["station_id"] == s)
        t = out.loc[rows, "time"]
        for i, nb in enumerate(nbs[:k], start=1):
            v = wide[nb].reindex(t).to_numpy()
            out.loc[rows, f"pm25_nb{i}"] = np.nan_to_num(v, nan=0.0)
            out.loc[rows, f"pm25_nb{i}_isna"] = np.isnan(v).astype(float)
    if not add_isna:
        out = out.drop(columns=[c + "_isna" for c in cols])
    return out


# ---------------------------------------------------------------------------
# G1-54: cửa sổ trượt
# ---------------------------------------------------------------------------
@dataclass
class WindowSet:
    """Tập cửa sổ lưu dạng chỉ số. Dùng như Dataset: len(ws), ws[i] -> (x, y, mask)."""
    L: int
    H: int
    feature_cols: list[str]
    series: list[dict] = field(default_factory=list)
    origins: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    future_cols: list[str] | None = None

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, i):
        si, t = self.origins[i]
        s = self.series[si]
        x = s["X"][t - self.L + 1: t + 1]
        y = s["y_z"][t + 1: t + 1 + self.H]
        mask = ~np.isnan(y)
        y = np.nan_to_num(y, nan=0.0)
        if self.future_cols:
            return x, y, mask, s["F"][t + 1: t + 1 + self.H]
        return x, y, mask

    def materialize(self):
        """Trả về mảng đầy đủ X (N,L,F), Y (N,H) [z], M (N,H) [có nhãn]."""
        n = len(self)
        F = len(self.feature_cols)
        X = np.empty((n, self.L, F), np.float32)
        Y = np.empty((n, self.H), np.float32)
        M = np.empty((n, self.H), bool)
        for i in range(n):
            X[i], Y[i], M[i] = self[i][:3]
        return X, Y, M

    def meta(self) -> pd.DataFrame:
        si, t = self.origins[:, 0], self.origins[:, 1]
        return pd.DataFrame({
            "city": [self.series[j]["city"] for j in si],
            "station_id": [self.series[j]["station_id"] for j in si],
            "time_origin": [self.series[j]["times"][k] for j, k in zip(si, t)],
        })

    def y_true_ug(self) -> np.ndarray:
        """PM2.5 thật (µg/m³) tại t+1..t+H; NaN nếu không phải giá trị ĐO THẬT (thiếu hoặc nội suy)."""
        out = np.full((len(self), self.H), np.nan)
        for i, (si, t) in enumerate(self.origins):
            s = self.series[si]
            y = s["y_raw"][t + 1: t + 1 + self.H].copy()
            y[~s["y_real"][t + 1: t + 1 + self.H]] = np.nan
            out[i] = y
        return out

    def last_observed_ug(self) -> np.ndarray:
        """PM2.5 (µg/m³) tại gốc t — dùng cho baseline persistence."""
        return np.array([self.series[si]["y_raw"][t] for si, t in self.origins])

    def target_times(self) -> np.ndarray:
        """Ma trận (N, H) các thời điểm mục tiêu — dùng cho kiểm tra rò rỉ."""
        return np.stack([self.series[si]["times"][t + 1: t + 1 + self.H] for si, t in self.origins]) \
            if len(self) else np.empty((0, self.H), dtype="datetime64[ns]")

    def input_times(self) -> np.ndarray:
        return np.stack([self.series[si]["times"][t - self.L + 1: t + 1] for si, t in self.origins]) \
            if len(self) else np.empty((0, self.L), dtype="datetime64[ns]")


def make_windows(df: pd.DataFrame, feature_cols: list[str], L: int, H: int = 24,
                 target_range: tuple | None = None, input_start=None,
                 target_col: str = "pm25_z", nan_policy: str = "drop_target", min_valid_targets: int = 1,
                 future_cols: list[str] | None = None, stride: int = 1,
                 isna_cols: tuple[str, ...] = ISNA_COLS, origin_whitelist: dict | None = None) -> WindowSet:
    """Tạo WindowSet từ bảng đã chuẩn hoá (có pm25, pm25_z, is_imputed và các cột đặc trưng).

    target_range: (start, end) inclusive — mọi t+1..t+H phải nằm trong khoảng này.
    input_start : nếu đặt, mọi t-L+1..t phải >= input_start (dùng cho TRAIN để không rò rỉ).
    nan_policy  : "drop_target" (mặc định) — chỉ bỏ cửa sổ khi PM2.5 (target_col) ở input có NaN; NaN ở
                  các đặc trưng khác điền 0 (= trung bình sau chuẩn hoá); với cột trong isna_cols thêm kênh
                  <col>_isna (khí tượng gốc HealthyAir thiếu nhiều, việc thiếu cũng là thông tin).
                  Nhờ vậy MỌI ablation có cùng tập cửa sổ -> so sánh công bằng.
                  "drop" — bỏ cửa sổ có NaN ở bất kỳ đặc trưng nào; "fill" — điền 0 + kênh _isna cho mọi cột có NaN.
    origin_whitelist: {(city, station_id): mảng time_origin được phép} — dùng để mọi mô hình/L/ablation
                  đánh giá trên CÙNG tập gốc t (xem Prepared.windows).
    min_valid_targets: số mục tiêu ĐO THẬT tối thiểu trong t+1..t+H để giữ cửa sổ.
    future_cols : (thiết lập phụ) khí tượng THỰC tại t+1..t+H — "dự báo thời tiết hoàn hảo".
    """
    feature_cols = list(feature_cols)
    missing = [c for c in feature_cols + [target_col, "pm25", "is_imputed"] if c not in df.columns]
    if missing:
        raise KeyError(f"Thiếu cột {missing}")
    data = df.sort_values(["city", "station_id", "time"])
    if nan_policy not in ("drop", "fill", "drop_target"):
        raise ValueError(f"nan_policy không hợp lệ: {nan_policy}")
    if target_col not in feature_cols and nan_policy == "drop_target":
        raise ValueError("drop_target cần target_col nằm trong feature_cols")
    fill_cols: list[str] = []
    if nan_policy == "fill":
        fill_cols = [c for c in feature_cols if data[c].isna().any()]
    elif nan_policy == "drop_target":
        fill_cols = [c for c in feature_cols if c in isna_cols]   # danh sách CỐ ĐỊNH -> số kênh tất định
    out_cols = feature_cols + [c + "_isna" for c in fill_cols]
    ws = WindowSet(L=L, H=H, feature_cols=out_cols, future_cols=future_cols)
    origins = []
    for (city, st), g in data.groupby(["city", "station_id"], sort=True):
        times = pd.DatetimeIndex(g["time"])
        if len(times) < L + H:
            continue
        if not ((times[1:] - times[:-1]) == pd.Timedelta(hours=1)).all():
            raise ValueError(f"{city}/{st}: thời gian không liên tục từng giờ -> chạy complete_hourly_grid trước")
        X = g[feature_cols].to_numpy(np.float32)
        tgt_nan = np.isnan(g[target_col].to_numpy(np.float32))
        if nan_policy in ("fill", "drop_target"):
            isna = g[fill_cols].isna().to_numpy(np.float32)
            X = np.concatenate([np.nan_to_num(X, nan=0.0), isna], axis=1)
        y_z = g[target_col].to_numpy(np.float32)
        y_raw = g["pm25"].to_numpy(np.float64)
        y_real = (~np.isnan(y_raw)) & (g["is_imputed"].to_numpy() == 0)
        T = len(g)
        t = np.arange(L - 1, T - H, stride)
        ok = np.ones(len(t), bool)
        if target_range is not None:
            rs, re = pd.Timestamp(target_range[0]), pd.Timestamp(target_range[1])
            ok &= (times[t + 1] >= rs) & (times[t + H] <= re)
        if input_start is not None:
            ok &= times[t - L + 1] >= pd.Timestamp(input_start)
        if nan_policy in ("drop", "drop_target"):
            row_bad = np.isnan(X).any(axis=1) if nan_policy == "drop" else tgt_nan
            bad = np.concatenate([[0], np.cumsum(row_bad)])
            ok &= (bad[t + 1] - bad[t - L + 1]) == 0
        if origin_whitelist is not None:
            allowed = origin_whitelist.get((city, st))
            ok &= np.isin(times.values[t], allowed) if allowed is not None else False
        cr = np.concatenate([[0], np.cumsum(y_real)])
        ok &= (cr[t + 1 + H] - cr[t + 1]) >= min_valid_targets
        t = t[ok]
        if len(t) == 0:
            continue
        s = {"city": city, "station_id": st, "times": times.values, "X": X, "y_z": y_z,
             "y_raw": y_raw, "y_real": y_real}
        if future_cols:
            s["F"] = np.nan_to_num(g[future_cols].to_numpy(np.float32), nan=0.0)
        ws.series.append(s)
        origins.append(np.stack([np.full(len(t), len(ws.series) - 1), t], axis=1))
    ws.origins = np.concatenate(origins).astype(np.int64) if origins else np.zeros((0, 2), np.int64)
    return ws


def origin_set(ws: WindowSet) -> dict:
    """{(city, station_id): mảng time_origin} của một WindowSet — làm whitelist cho tập đánh giá chung."""
    m = ws.meta()
    return {k: g["time_origin"].to_numpy() for k, g in m.groupby(["city", "station_id"])}


def windows_for_split(df_s: pd.DataFrame, feature_cols, L, H, rng: dict, split: str, **kw) -> WindowSet:
    """Tiện ích: cửa sổ cho 'train' (input cũng trong train) hoặc 'val'/'test'."""
    input_start = rng["train"][0] if split == "train" else None
    return make_windows(df_s, feature_cols, L, H, target_range=rng[split], input_start=input_start, **kw)
