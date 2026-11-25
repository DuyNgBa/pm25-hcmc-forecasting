"""Các bước làm sạch DÙNG CHUNG cho cả HealthyAir (TV1) và Bắc Kinh (TV3).

Mục đích: hai thành phố phải qua CÙNG quy tắc, nếu không mô hình sẽ học cả
sự khác biệt trong cách xử lý dữ liệu thay vì quy luật vật lý.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def complete_hourly_grid(df: pd.DataFrame, start=None, end=None,
                         keys=("city", "station_id"), time_col="time") -> pd.DataFrame:
    """Tạo lưới giờ đầy đủ cho từng trạm; giờ không có dữ liệu -> NaN.

    Mô hình chuỗi thời gian giả định các bước cách đều nhau 1 giờ. Nếu thiếu dòng
    (chứ không phải NaN) thì phép "dịch 24 dòng" sẽ KHÔNG còn là "lùi 24 giờ".
    """
    out = []
    for key_vals, g in df.groupby(list(keys), sort=True):
        g = g.sort_values(time_col)
        s = pd.Timestamp(start) if start is not None else g[time_col].min()
        e = pd.Timestamp(end) if end is not None else g[time_col].max()
        full = pd.DataFrame({time_col: pd.date_range(s, e, freq="h")})
        g2 = full.merge(g, on=time_col, how="left")
        for k, v in zip(keys, key_vals if isinstance(key_vals, tuple) else (key_vals,)):
            g2[k] = v
        out.append(g2)
    res = pd.concat(out, ignore_index=True)
    cols = list(keys) + [time_col] + [c for c in res.columns if c not in (*keys, time_col)]
    return res[cols]


def interpolate_short_gaps(s: pd.Series, max_gap: int) -> tuple[pd.Series, pd.Series]:
    """Nội suy tuyến tính CHỈ những đoạn NaN dài <= max_gap và có giá trị thật ở cả 2 đầu.

    Trả về (chuỗi sau nội suy, cờ is_imputed). Đoạn dài hơn max_gap giữ NaN
    (không bịa dữ liệu cho khối mất dài như giai đoạn giãn cách).
    Đầu vào phải là chuỗi của MỘT trạm trên lưới giờ đầy đủ, đã sắp theo thời gian.
    """
    s = s.astype(float).reset_index(drop=True)
    isna = s.isna().to_numpy()
    fill_mask = np.zeros(len(s), dtype=bool)
    # Tìm các "run" NaN liên tiếp: [start, end)
    i, n = 0, len(s)
    while i < n:
        if isna[i]:
            j = i
            while j < n and isna[j]:
                j += 1
            run_len = j - i
            bounded = i > 0 and j < n          # có giá trị thật ở cả hai phía
            if bounded and run_len <= max_gap:
                fill_mask[i:j] = True
            i = j
        else:
            i += 1
    interp = s.interpolate(method="linear", limit_area="inside")
    out = s.copy()
    out[fill_mask] = interp[fill_mask]
    return out, pd.Series(fill_mask.astype(np.int8))


def missing_table(df: pd.DataFrame, cols, by="station_id") -> pd.DataFrame:
    """Bảng % thiếu theo biến × trạm (dùng cho bảng trước/sau làm sạch)."""
    return (df.groupby(by)[list(cols)]
              .apply(lambda g: g.isna().mean() * 100)
              .round(2))
