"""src/evaluate.py — điểm vào đánh giá CHUNG theo quy ước README nhóm ("sửa src/evaluate.py phải báo cả nhóm").

Code thật nằm ở src/eval/ (metrics, aqi, results); file này gom lại để cả nhóm chỉ cần:
    from src.evaluate import evaluate, evaluate_grouped, wide_to_long, results_rows, append_results, save_predictions
"""
from src.eval.aqi import aqi_classification_scores, pm25_to_level  # noqa: F401
from src.eval.metrics import (add_area_type, evaluate, evaluate_grouped,  # noqa: F401
                              regression_scores, wide_to_long)
from src.eval.results import (PRED_COLUMNS, RESULT_COLUMNS, append_results,  # noqa: F401
                              make_run_id, results_rows, save_predictions)


# ---------------------------------------------------------------------------
# Chạy trực tiếp:  python -m src.evaluate      (minh hoạ G1-57, G1-58 trên ví dụ nhỏ)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    from src.eval.aqi import level_names

    print("=== G1-58: quy đổi nồng độ PM2.5 -> mức AQI ===")
    xs = [5, 9.0, 20, 35.4, 35.5, 50, 100, 150, 300]
    print(pd.DataFrame({"PM2.5 (µg/m³)": xs,
                        "US EPA 2024 (chính)": [level_names("us_epa_2024")[i] for i in pm25_to_level(xs)],
                        "VN_AQI (bổ sung)": [level_names("vn_aqi_2019")[i] for i in pm25_to_level(xs, "vn_aqi_2019")]
                        }).to_string(index=False))

    print("\n=== G1-57: tính độ đo trên một bảng dự đoán nhỏ ===")
    preds = pd.DataFrame({
        "city": "hcm", "station_id": ["hcm_01"] * 4 + ["hcm_02"] * 4,
        "time_origin": pd.Timestamp("2022-03-01 07:00"), "horizon": [1, 1, 24, 24] * 2,
        "y_true": [20, 40, np.nan, 30, 15, 50, 25, 10],     # NaN = giờ không phải đo thật -> bị bỏ qua
        "y_pred": [22, 30, 99, 25, 15, 38, 30, 14], "model": "demo", "config": "demo"})
    print(preds[["station_id", "horizon", "y_true", "y_pred"]].to_string(index=False))
    print("\nTheo tầm dự báo:")
    print(evaluate_grouped(preds, ["horizon"]).round(2).to_string(index=False))
    print("\nGhi chú: dòng y_true = NaN không được tính (n = 7 chứ không phải 8);"
          " MAE_high chỉ tính trên giờ y_true > 35,4 (có 2 giờ: 40 và 50).")
