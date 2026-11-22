# Mẫu bảng kết quả

Mỗi phương pháp một file `results/<tên-phương-pháp>.csv`, copy từ `results_template.csv`, giữ đúng thứ tự cột.
Mỗi dòng = 1 mô hình × 1 cấu hình × 1 tầm dự báo × 1 trạm (hoặc ALL) × 1 tập đánh giá × 1 seed.

| Cột | Giá trị hợp lệ | Ý nghĩa |
|---|---|---|
| member | Duy Ngô / Minh Trí / Bo bo | Ai chạy |
| model | persistence, hourly_mean, Ridge, RandomForest, LightGBM, XGBoost, LSTM, GRU, Seq2Seq, PatchTST | Tên mô hình |
| strategy | S0, S1, S2_full, S2_head, S3 (cây ghi S2) | Cách dùng dữ liệu Bắc Kinh; S0 = chỉ TP.HCM |
| ablation | a, b, c, d, c_oracle; baseline ghi `-` | Bộ đặc trưng (RQ2) |
| train_frac | 1.0 / 0.5 / 0.25 | Tỉ lệ dữ liệu train TP.HCM được dùng |
| horizon | 1 / 6 / 24 | Dự báo trước bao nhiêu giờ |
| station | ALL hoặc S1..S6 | Trạm (ALL = gộp 6 trạm) |
| L | 24 / 48 / 72; baseline ghi 0 | Số giờ lịch sử đưa vào mô hình |
| seed | 42 / 43 / 44; baseline ghi 0 | Seed ngẫu nhiên |
| MAE, RMSE | µg/m³ | Sai số tuyệt đối TB / căn của sai số bình phương TB |
| R2 | ≤ 1 | Tỉ lệ biến động giải thích được |
| MAE_high | µg/m³ | MAE chỉ trên giờ PM2.5 thật > 35,4 µg/m³ |
| AQI_acc, AQI_macroF1 | 0–1 | Độ chính xác khi quy dự báo ra mức AQI |
| notes | tự do | Ghi chú: thời gian train, số tham số, phiên bản dữ liệu |
| eval_set | test / fold1..fold4 | Đánh giá trên test chính hay fold nào |
| n | số nguyên | Số giờ dùng để tính độ đo; các mô hình phải bằng nhau |
| run_id | chuỗi | Mã lần chạy, nối sang file dự đoán `results/preds/` |

Không tự tính độ đo bằng tay: dùng hàm trong `src/evaluate.py`.

---


# Format file dự đoán

Mỗi lần chạy mô hình, lưu dự đoán trên tập đánh giá vào:

`results/preds/<run_id>__<eval_set>.parquet`   
(ví dụ: `results/preds/patchtst_1a2b3c4d__test.parquet`)

`run_id` trùng với cột `run_id` trong file kết quả, để nối được 2 file với nhau.

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| city | chuỗi | `hcm` |
| station_id | chuỗi | `S1` .. `S6` |
| time_origin | thời gian | Thời điểm gốc t (giờ cuối cùng của input), giờ địa phương, không kèm múi giờ |
| horizon | số nguyên | Tầm dự báo h ∈ {1, 6, 24}. Thời điểm được dự báo = `time_origin + h` giờ |
| y_true | số thực | PM2.5 thật (µg/m³). **Để NaN** nếu giờ đó không phải giá trị đo thật (bị thiếu hoặc là giá trị nội suy) |
| y_pred | số thực | PM2.5 dự đoán (µg/m³, đã đổi ngược về đơn vị gốc) |
| model | chuỗi | Tên mô hình, giống cột `model` trong file kết quả |
| config | chuỗi | Bằng `run_id` |

Quy tắc:
- Dạng **dài**: mỗi dòng = 1 trạm × 1 thời điểm gốc × 1 tầm. Mỗi lần chạy có đủ 3 tầm 1, 6, 24.
- Mọi mô hình dùng **cùng tập (station_id, time_origin)** trên test, để so sánh được từng giờ một.
- Lưu bằng **parquet**: nhẹ hơn CSV nhiều lần và giữ đúng kiểu thời gian.

Ví dụ lưu bằng pandas:

```python
preds.to_parquet(f"results/preds/{run_id}__test.parquet", index=False)   # cần cài pyarrow
```