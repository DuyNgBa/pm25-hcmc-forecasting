# [Bản nháp] Thiết lập thí nghiệm (G3-16)

*Người viết: Bo bo. Mọi con số về mốc thời gian và tham số đều lấy từ `configs/data.yaml`. Nếu sửa config thì sửa cả mục này.*

## 1. Phát biểu bài toán

Với mỗi trạm s và thời điểm gốc t, mô hình nhận cửa sổ đầu vào gồm L giờ gần nhất, X_{t−L+1:t}, và dự báo nồng độ PM2.5 tại t+1, …, t+24. Cửa sổ đầu vào chứa:

- PM2.5 đã chuẩn hoá;
- đặc trưng thời gian: sin/cos của giờ trong ngày và thứ trong tuần;
- tuỳ thiết lập: khí tượng và PM2.5 của các trạm lân cận.

Ta báo cáo ba tầm h ∈ {1, 6, 24} giờ, thử L ∈ {24, 48, 72} và chọn L = 48 cho cấu hình chính. Mô hình cây huấn luyện riêng một mô hình cho mỗi tầm. Mô hình học sâu dự báo cả 24 bước cùng lúc.

Trong thiết lập chính, khí tượng chỉ được dùng đến thời điểm t. Thiết lập phụ `c_oracle` dùng thêm khí tượng *thực đo* tại t+1..t+24. Thiết lập này đóng vai trò giới hạn trên cho trường hợp có dự báo thời tiết hoàn hảo.

## 2. Chia dữ liệu

**Chia theo thời gian, không xáo trộn.**

| | Train | Validation | Test |
|---|---|---|---|
| TP.HCM | 23/02/2021 – 31/12/2021 | 01/01 – 28/02/2022 | 01/03 – 21/06/2022 |
| Bắc Kinh (chỉ dùng để pretrain) | 01/03/2013 – 29/02/2016 | 01/03/2016 – 28/02/2017 | không có |

Validation chỉ dùng cho early stopping và chọn siêu tham số. Dữ liệu Bắc Kinh **không bao giờ** xuất hiện trong tập đánh giá.

**Gán cửa sổ vào tập.** Một cửa sổ thuộc tập X khi mọi thời điểm mục tiêu t+1..t+24 đều nằm trong X. Với train, ta yêu cầu thêm phần đầu vào cũng nằm trong train. Với val và test, đầu vào được phép lấy lịch sử trước đó, giống như khi dự báo thật.

**Đánh giá rolling-origin.** Tập test chính chỉ rơi vào giai đoạn chuyển mùa khô sang mùa mưa. Để bớt phụ thuộc vào điều đó, ta thêm 4 fold kiểu cửa sổ mở rộng (expanding window). Mỗi fold có val 1 tháng và test 2 tháng, nối tiếp nhau, phủ liên tục từ 11/2021 đến 06/2022 và gồm cả đỉnh mùa khô (Hình X: `reports/figures/rolling_origin_folds.png`). Ở mỗi fold, mọi bước tiền xử lý có "fit" đều được fit lại trên train của fold đó.

## 3. Tiền xử lý dùng chung và chống rò rỉ dữ liệu

1. **Chuẩn hoá PM2.5**: z = (log(1 + x) − μ_s) / σ_s. μ_s và σ_s là trung bình và độ lệch chuẩn của log(1 + PM2.5) tại trạm s, tính trên train và chỉ trên giá trị đo thật. Mô hình dự đoán z, sau đó z được đổi ngược về µg/m³ để đánh giá.
2. **Chuẩn hoá khí tượng**: tham số tính theo thành phố, trên train. Thành phần gió u, v chỉ chia cho độ lệch chuẩn, không trừ trung bình, để giữ hướng gió. Lượng mưa được biến đổi log(1 + x) rồi chia cho độ lệch chuẩn, nên 0 mm vẫn là 0.
3. **Nội suy và outlier**: nội suy tuyến tính khoảng trống ≤ 3 giờ trong từng trạm. PM2.5 **không bị cắt outlier** ở cả hai thành phố (giá trị cao là sự kiện cần dự báo); các biến phụ của HealthyAir được cắt ở phân vị 99,9 tính trên train (TV1).
4. **Giá trị thiếu ở đầu vào**: cửa sổ bị loại khi PM2.5 đầu vào có giá trị thiếu. Giá trị thiếu ở các đặc trưng khác được điền 0 (bằng trung bình sau chuẩn hoá). Riêng khí tượng gốc HealthyAir có thêm kênh cờ "thiếu".
5. **Kiểm thử tự động** (`tests/test_pipeline.py`):
   - không cửa sổ train nào chứa thời điểm thuộc val hoặc test;
   - tham số scaler không đổi khi dữ liệu ngoài train bị thay đổi;
   - phép đảo chuẩn hoá khôi phục đúng giá trị gốc trên dữ liệu thật.

## 4. Tập đánh giá chung

Các mô hình dùng L khác nhau và tập đặc trưng khác nhau, nên số cửa sổ hợp lệ có thể khác nhau. Để so sánh công bằng, ta cố định **một tập gốc t duy nhất** cho mỗi tập val/test. Tập này gồm các gốc t có đủ 72 giờ PM2.5 trước t và có ít nhất một nhãn đo thật trong 24 giờ tới. Mọi mô hình, mọi L và mọi ablation đều được chấm trên đúng tập này (cột `n` trong bảng kết quả bằng nhau).

**Nhãn đánh giá chỉ gồm giá trị đo thật.** Giờ bị thiếu và giờ được nội suy đều bị loại khỏi phép tính độ đo.

## 5. Độ đo

- **Hồi quy (µg/m³)**: MAE, RMSE và R² tính theo từng tầm, từng trạm và từng loại khu vực (giao thông, dân cư, công nghiệp, nền đô thị). **MAE_high** là MAE chỉ trên các giờ có PM2.5 thật > 35,4 µg/m³; đây là những giờ cần cảnh báo.
- **Phân loại mức chất lượng không khí**: giá trị thật và giá trị dự báo được quy đổi theo bảng điểm gãy PM2.5 của US EPA (cập nhật 2024): ≤ 9,0 / 35,4 / 55,4 / 125,4 / 225,4 / > 225,4 µg/m³. Sau đó tính accuracy và macro-F1. macro-F1 lấy trung bình trên các mức có xuất hiện trong nhãn thật. Lưu ý: ta áp điểm gãy nồng độ cho từng giờ, nên đây là "mức nồng độ theo giờ", không phải AQI chính thức (AQI chính thức dùng trung bình 24 giờ hoặc NowCast). Thang VN_AQI (QĐ 1459/QĐ-TCMT) được báo cáo bổ sung.

## 6. Baseline

Mọi mô hình bắt buộc phải được so với hai baseline sau:

1. **Persistence**: ŷ(t+h) = y(t).
2. **Trung bình theo giờ trong ngày**: lấy trung bình PM2.5 trên train của trạm s tại cùng giờ trong ngày với t+h.

Cả hai được tính trên cùng tập đánh giá chung (mục 4).

## 7. Quy ước so sánh công bằng giữa ba nhóm phương pháp

- Cùng phiên bản dữ liệu (tag `data-v1.0`), cùng mốc chia, cùng hàm đánh giá (`src/eval/metrics.py`).
- Cùng ngân sách tối ưu siêu tham số: **30 lần thử** trên validation (Optuna, bộ lấy mẫu TPE, seed 42), mục tiêu là MAE trên val tính bằng µg/m³.
- Mô hình học sâu chạy **3 seed** (42, 43, 44) và báo cáo trung bình ± độ lệch chuẩn. Early stopping theo loss trên val.
- Kết quả ghi vào một bảng chung theo cùng mẫu cột. Dự đoán trên test được lưu theo cùng format để phân tích lỗi ở giai đoạn 3.

## 8. Ma trận thí nghiệm

| Câu hỏi | Biến thay đổi | Giữ cố định |
|---|---|---|
| RQ1 | mô hình (baseline, Ridge, RF, LightGBM, XGBoost, LSTM, GRU, PatchTST) × tầm | ablation (c), L = 48, S0 |
| RQ2 | ablation (a) chỉ PM2.5; (b) + khí tượng gốc HealthyAir; (c) thay bằng Open-Meteo; (d) + PM2.5 của 5 trạm gần nhất | L = 48, S0 |
| RQ3 | chiến lược S0 / S1 / S2 (fine-tune toàn bộ hoặc chỉ đầu ra) / S3 × train_frac ∈ {100, 50, 25}% | ablation (c), L = 48 |
| Phụ | L ∈ {24, 48, 72} (PatchTST); seq2seq so với dự báo trực tiếp (RNN); `c_oracle` | |

Với `train_frac` < 1, ta lấy **đoạn thời gian liên tục cuối** của train (gần val nhất). Cách này mô phỏng một trạm mới lắp chỉ có vài tháng dữ liệu.
