"""G1-58 — Quy đổi nồng độ PM2.5 (µg/m³) sang mức chất lượng không khí.

THANG CHÍNH: US EPA, bảng điểm gãy PM2.5 cập nhật 2024 (hiệu lực 06/05/2024).
THANG BỔ SUNG: VN_AQI theo Quyết định 1459/QĐ-TCMT (12/11/2019) của Tổng cục Môi trường.

Vì sao chọn US EPA 2024 làm thang chính?
  1. Ngưỡng "giờ ô nhiễm cao" 35,4 µg/m³ của đề xuất chính là ranh giới Moderate/USG của US EPA
     -> thang AQI và độ đo MAE_high dùng CÙNG một mốc, dễ diễn giải.
  2. Các bài so sánh (gồm [2] về TP.HCM) chủ yếu dùng thang US EPA.
  3. VN_AQI giờ của PM2.5 tính trên giá trị NowCast (trung bình có trọng số 12 giờ), không phải
     nồng độ 1 giờ; áp thẳng lên dự báo từng giờ sẽ KHÔNG đúng định nghĩa chính thức.

LƯU Ý PHƯƠNG PHÁP (ghi vào báo cáo): US EPA định nghĩa AQI PM2.5 trên trung bình 24 giờ
(NowCast cho báo cáo theo giờ). Ở đây ta áp các ĐIỂM GÃY NỒNG ĐỘ lên giá trị từng giờ để có
"mức nồng độ theo giờ" — một cách quy đổi phục vụ đánh giá phân loại, không phải AQI chính thức.
"""
from __future__ import annotations

import numpy as np

# Cận trên (inclusive) của mỗi mức, đơn vị µg/m³. Mức cuối là +inf.
SCALES = {
    "us_epa_2024": {
        "upper": [9.0, 35.4, 55.4, 125.4, 225.4, np.inf],
        "labels": ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy",
                   "Very Unhealthy", "Hazardous"],
    },
    # Bảng BP_i của PM2.5 trong QĐ 1459/QĐ-TCMT: 0, 25, 50, 80, 150, 250, 350, 500 ứng với I = 0..500.
    # Mức chất lượng: 0–50 Tốt, 51–100 Trung bình, 101–150 Kém, 151–200 Xấu, 201–300 Rất xấu, >300 Nguy hại
    # -> theo nồng độ: <=25, <=50, <=80, <=150, <=250, >250.
    "vn_aqi_2019": {
        "upper": [25.0, 50.0, 80.0, 150.0, 250.0, np.inf],
        "labels": ["Tốt", "Trung bình", "Kém", "Xấu", "Rất xấu", "Nguy hại"],
    },
}


def pm25_to_level(pm25, scale: str = "us_epa_2024") -> np.ndarray:
    """Trả về mã mức 0..5 (NaN giữ NaN -> -1). Giá trị âm (dự đoán lỗi) được coi là 0."""
    x = np.asarray(pm25, dtype=float)
    upper = np.array(SCALES[scale]["upper"])
    # Làm tròn 1 chữ số thập phân như quy ước báo cáo của EPA (vd. 35.44 -> 35.4 thuộc Moderate)
    xr = np.round(np.clip(x, 0, None), 1)
    lvl = np.searchsorted(upper, xr, side="left")
    lvl = np.where(np.isnan(x), -1, lvl)
    return lvl.astype(int)


def level_names(scale: str = "us_epa_2024") -> list[str]:
    return SCALES[scale]["labels"]


def aqi_classification_scores(y_true, y_pred, scale: str = "us_epa_2024") -> dict:
    """Accuracy và macro-F1 trên mức AQI quy đổi (bỏ các dòng y_true NaN).

    macro-F1 lấy trung bình trên các mức XUẤT HIỆN trong y_true (mức không có mẫu thật nào sẽ
    làm F1 không xác định). Ghi rõ điều này trong báo cáo vì TP.HCM hiếm khi vượt mức 3.
    """
    from sklearn.metrics import accuracy_score, f1_score

    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = ~np.isnan(yt) & ~np.isnan(yp)
    if ok.sum() == 0:
        return {"AQI_acc": np.nan, "AQI_macroF1": np.nan}
    lt, lp = pm25_to_level(yt[ok], scale), pm25_to_level(yp[ok], scale)
    labels = np.unique(lt)
    return {"AQI_acc": float(accuracy_score(lt, lp)),
            "AQI_macroF1": float(f1_score(lt, lp, labels=labels, average="macro", zero_division=0))}
