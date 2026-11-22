# G0-07 – Tóm tắt bài [2] và bảng kết quả tham chiếu

**Bài:** Rakholia R., Le Q., Vu K., Ho B.Q., Carbajo R.S. (2022). *AI-based air quality PM2.5 forecasting models for developing countries: A case study of Ho Chi Minh City, Vietnam.* Urban Climate 46, 101315. doi:10.1016/j.uclim.2022.101315 (open access, CC BY-NC-ND).
**File:** `overview/AI-based-air-quality-PM2.5-forecasting-models-for-developing-countries.pdf`
**Dùng cho:** G3-10 (so sánh với bài [2]), G3-14 (tổng quan tài liệu).

> Đây là bài "gốc" của dữ liệu nhóm mình: cùng 6 trạm **HealthyAir**, cùng TP.HCM, cùng năm 2021. Vì vậy đây là mốc so sánh quan trọng nhất trong báo cáo.

---

## 1. Dữ liệu

| Mục | Bài [2] |
|---|---|
| Nguồn | 6 trạm HealthyAir (CeADAR + Viện Môi trường & Tài nguyên ĐHQG-HCM) |
| Thời gian | 02/2021 → 12/2021 (gồm giai đoạn giãn cách 5–9/2021) |
| Tần suất | Đo mỗi phút → lấy trung bình theo giờ |
| Biến đo | PM2.5, TSP, SO2, O3, NO2, CO, nhiệt độ, độ ẩm |
| **Biến đầu vào mô hình** | **Chỉ PM2.5 quá khứ của chính trạm** (đơn biến). Khí tượng và chất khác chưa dùng, để "future work" |
| Trạm | 1 nền đô thị (ĐHQG, Thủ Đức) · 2 giao thông (Bình Tân) · 3 công nghiệp (KCN Tân Bình) · 4 dân cư (Bình Thạnh) · 5 giao thông (Q.3) · 6 giao thông + dân cư (Q.10) |
| Đối chứng thêm | PM2.5 theo giờ của Lãnh sự quán Mỹ (gần trạm 5) |

## 2. Tiền xử lý

- Giá trị hợp lệ: 0–500 µg/m³; ngoài khoảng → NaN.
- Ngoại lai: > 200 µg/m³ **và** gấp 3 lần giờ trước → bỏ.
- Điền thiếu: thiếu 1 giờ → lấy giờ trước; thiếu nhiều giờ liền → lấy cùng giờ của ngày hôm trước.
- **Chỉ chấm điểm trên mục tiêu là giá trị đo thật** (bỏ cặp có mục tiêu là giá trị đã điền). Nhóm mình cũng làm vậy (`y_true = NaN` nếu nội suy).
- Chuẩn hoá min-max về [0, 1].

## 3. Bài toán và cách chia dữ liệu

- **One-shot multi-step:** input 60 giờ PM2.5 (thử L ∈ {12, 24, …, 96}, 60 là tốt nhất) → output 24 giờ tiếp theo (t+1..t+24).
- Hai nhiệm vụ: (i) PM2.5 theo giờ; (ii) trung bình trượt 24 giờ (theo QCVN 05:2013 và WHO).
- **Một mô hình riêng cho mỗi trạm.**
- **Giao thức "dữ liệu ngắn":** chuỗi không dừng nên thay vì dùng cả năm, họ lấy mẫu đều ra **nhiều bộ dữ liệu con dài khoảng 3 tháng** mỗi trạm. Mỗi bộ chia **train/val/test = 80/10/10 theo thời gian**. Nghĩa là mỗi tập test chỉ khoảng 9 ngày, nằm sát ngay sau train.
- Kết quả báo cáo = **trung bình trên mọi tầm 1..24 giờ, mọi trạm** (và các bộ con).

## 4. Mô hình và siêu tham số

| Mô hình | Cấu hình |
|---|---|
| SGDRegressor (bọc `MultiOutputRegressor`) | loss = squared_error, eta0 = 0.02, alpha = 0.003, penalty = l2, max_iter = 800 |
| XGBoost (bọc `MultiOutputRegressor`) | n_estimators = 100, max_depth = 5, learning_rate = 0.1, min_child_weight = 6, colsample_bytree = 0.5, booster = gbtree |
| 1D CNN-LSTM | Conv1D(32, k=2, stride=2, tanh) → MaxPool(2) → Conv1D(16, k=2, tanh) → Dropout 0.1 → MaxPool(2) → LSTM(16, relu) → Dense(24). Adam, MSE, batch 4, 100 epoch |
| Prophet | changepoint_prior_scale = 0.1, seasonality_prior_scale = 5.0, không dùng ngày lễ; không cần chuẩn hoá/cửa sổ |

Chọn siêu tham số: grid search rồi chỉnh tay trên val.

## 5. Độ đo và kết quả

Độ đo: **RMSE, MAE** (µg/m³). Không có R², không có độ đo mức AQI, không có kiểm định thống kê, không báo nhiều seed.

**Bảng tham chiếu (trung bình 6 trạm, trung bình tầm 1..24 giờ):**

| Nhiệm vụ | Độ đo | XGBoost | 1D CNN-LSTM | **SGDRegressor** | Prophet |
|---|---|---|---|---|---|
| PM2.5 theo giờ (Table 5) | RMSE | 8.73 | 8.86 | **7.74** | 8.47 |
| | MAE | 6.28 | 6.65 | **5.75** | 5.89 |
| TB trượt 24 h (Table 6) | RMSE | 4.31 | 4.12 | **3.38** | 5.55 |
| | MAE | 3.36 | 3.25 | **2.64** | 4.21 |

Bản CSV: [G0-07_bai2_reference.csv](G0-07_bai2_reference.csv).

Nhận xét chính của tác giả:
- SGDRegressor (tuyến tính) tốt nhất ở cả 2 nhiệm vụ. Lý do họ đưa ra: train mỗi bộ quá nhỏ (80% của 3 tháng), nên mô hình phức tạp (XGBoost, CNN-LSTM) thua.
- Sai số tăng dần theo tầm dự báo (h1–h3 thấp nhất), trừ trạm 5 (mất điện, thiếu dữ liệu dài).
- Kết quả theo trạm khác nhau rõ: vị trí trạm ảnh hưởng mạnh.
- Đặc điểm theo thời gian: cao nhất ở trạm 4 (dân cư, TB năm 23,1 µg/m³, gấp 4,6 lần WHO), rồi trạm 3 (KCN), trạm 1. Đỉnh trong ngày 6–8 giờ sáng và 17–18 giờ; cao nhất thứ Tư, thứ Năm; mùa khô (11–1) cao, mùa mưa (5–8) thấp; năm 2021 bị giãn cách làm lệch xu hướng.

## 6. So sánh với thiết kế của nhóm (chuẩn bị cho G3-10)

| Khía cạnh | Bài [2] | Nhóm mình |
|---|---|---|
| Dữ liệu | HealthyAir 02–12/2021 | HealthyAir 02/2021–06/2022 + Open-Meteo + Bắc Kinh |
| Đầu vào | Chỉ PM2.5 của trạm | Ablation a/b/c/d: PM2.5 + thời gian, + khí tượng HealthyAir, + Open-Meteo, + 5 trạm lân cận |
| L / H | 60 / 24 | 24, 48, 72 / 24 |
| Chia dữ liệu | Nhiều bộ con 3 tháng, 80/10/10 mỗi bộ (test khoảng 9 ngày, sát train) | Train 02–12/2021, val 01–02/2022, test 03–06/2022 (gần 4 tháng) + rolling-origin 4 fold |
| Mô hình | 1 mô hình / trạm | Mô hình chung cho 6 trạm (+ chuyển giao từ Bắc Kinh, RQ3) |
| Báo cáo | TB tầm 1..24 | Riêng h = 1, 6, 24; thêm R², MAE_high, AQI; 3 seed; luôn so với persistence |
| Mốc ngây thơ | Không có persistence | Có persistence + TB theo giờ |

**Lưu ý khi so số (rất quan trọng cho G3-10):**
1. **Hai cách tính không cùng loại.** Bài [2] lấy trung bình trên mọi tầm 1..24, còn mình báo h = 1, 6, 24. Muốn đặt cạnh nhau phải tính thêm "TB tầm 1..24, gộp 6 trạm" từ dự đoán của mình. Hiện format `results/preds/` chỉ lưu h ∈ {1, 6, 24}, nên **LSTM/GRU cần lưu thêm dự đoán đủ 24 tầm** (xem plan GĐ2, bước G2-49).
2. **Tập test của bài [2] dễ hơn:** test chỉ khoảng 9 ngày ngay sau train và nằm trong năm 2021 (có giãn cách, PM2.5 thấp). Test của mình là 4 tháng của năm 2022, tách khỏi train bởi 2 tháng val. Số của mình cao hơn chưa chắc là mô hình kém hơn.
3. Bài [2] không có persistence. Tham khảo persistence trên test của mình (Bobo, data v0.9-dev): RMSE 5.49 / 12.03 / 12.66 ở h = 1 / 6 / 24. Nên báo thêm "persistence TB 1..24" để người đọc thấy mức khó của mỗi tập test.
4. Có thể (tuỳ chọn) chạy lại SGDRegressor theo đúng giao thức của bài [2] trên dữ liệu của mình để so sánh công bằng ("reproduce rồi mới so").

Điểm mình có thể nhấn mạnh trong báo cáo: (i) thêm khí tượng và trạm lân cận, đúng phần bài [2] để "future work"; (ii) test trên giai đoạn dài, khác năm, có rolling-origin; (iii) có persistence, nhiều seed, độ đo theo mức AQI; (iv) thử chuyển giao từ Bắc Kinh.

## 7. Trích dẫn (BibTeX, cho G0-09)

```bibtex
@article{rakholia2022pm25hcmc,
  title   = {AI-based air quality PM2.5 forecasting models for developing countries: A case study of Ho Chi Minh City, Vietnam},
  author  = {Rakholia, Rajnish and Le, Quan and Vu, Khue and Ho, Bang Quoc and Carbajo, Ricardo Simon},
  journal = {Urban Climate},
  volume  = {46},
  pages   = {101315},
  year    = {2022},
  doi     = {10.1016/j.uclim.2022.101315}
}
```
