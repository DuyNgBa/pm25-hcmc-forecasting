# G0-08 — Ghi chú: bộ dữ liệu Beijing Multi-Site Air Quality [3][4] và các mốc so sánh

*Người viết: Bo bo (TV3). Trạng thái: bản 1. Các mục "⚠ cần xác minh" là những chỗ chưa đọc được toàn văn.*

## 1. Bài giới thiệu dữ liệu — Zhang et al. (2017) [3]

- **Trích dẫn**: Zhang, S., Guo, B., Dong, A., He, J., Xu, Z., Chen, S.X. (2017). *Cautionary tales on air-quality improvement in Beijing*. Proc. R. Soc. A 473(2205): 20170457. DOI 10.1098/rspa.2017.0457 (đã kiểm tra trên PMC5627385).
- **Bài này KHÔNG phải bài dự báo.** Đây là một nghiên cứu thống kê: số liệu chính thức nói PM2.5 năm 2016 của Bắc Kinh giảm 9,9%, nhóm tác giả phân tích PM2.5 bốn năm ở 36 trạm và cho rằng sau khi điều chỉnh theo khí tượng thì mức cải thiện "không đẹp như con số 9,9%" và còn rất bất định. Số liệu chính thức được tính bằng trung bình đơn giản các giá trị theo giờ của 11 trạm cấp quốc gia (Guokong).
- **Ý nghĩa với đồ án**:
  1. *Khí tượng là biến gây nhiễu lớn* (meteorological confounding) đối với PM2.5. Đây là căn cứ để ta bổ sung khí tượng (RQ2) và dùng CÙNG một nguồn khí tượng cho cả hai thành phố (mục 4.3 của đề xuất).
  2. Bộ UCI là một **tập con 12 trạm** trong mạng 36 trạm mà bài báo dùng.

## 2. Trang mô tả UCI [4]

| Mục | Giá trị |
|---|---|
| Tên / ID | Beijing Multi-Site Air Quality, UCI dataset 501, DOI 10.24432/C5RK5G |
| Ngày đóng góp | 19/09/2019 |
| Quy mô | 420.768 dòng = 12 trạm × 35.064 giờ (01/03/2013 00h – 28/02/2017 23h) |
| Nguồn ô nhiễm | Beijing Municipal Environmental Monitoring Center, 12 trạm cấp quốc gia |
| Nguồn khí tượng | Mỗi trạm được ghép với trạm khí tượng **gần nhất** của Cục Khí tượng Trung Quốc (CMA) |
| Biến | `No, year, month, day, hour, PM2.5, PM10, SO2, NO2, CO, O3` (µg/m³), `TEMP` (°C), `PRES` (hPa), `DEWP` (°C), `RAIN` (mm), `wd` (16 hướng), `WSPM` (m/s), `station` |
| Giấy phép | CC BY 4.0 |

**Nhóm tự kiểm tra lại** (`src/data/beijing.py`, md5 ghi trong `src/data/download_beijing.py`):

- Đủ 12 file × 35.064 dòng, không trùng khoá, lưới giờ liên tục.
- PM2.5 thiếu 1,1–2,7% mỗi trạm (sau khi nội suy khoảng trống ≤ 3 giờ: 0,5–1,9%) → khớp con số "1–3%" trong đề xuất.
- PM2.5 có giá trị lớn nhất 999 µg/m³ (khả năng là giá trị trần của thiết bị). Nhóm giữ nguyên (không cắt), giống PM2.5 TP.HCM.
- `wd` không có giá trị "calm/cv"; 1.822 giờ thiếu hướng gió.
- **Không có độ ẩm tương đối.** Nếu cần (chỉ khi chưa có Open-Meteo) thì tính từ TEMP và DEWP bằng công thức Magnus.
- Trung bình PM2.5 = 79,7 µg/m³, trung vị 55 → khớp bảng 4.1.

## 3. Nghiên cứu dự báo dùng bộ này làm mốc

> ⚠ **Bẫy hay gặp**: UCI có HAI bộ Bắc Kinh. **Beijing PM2.5** (UCI 381) chỉ có 1 điểm đo ở Đại sứ quán Mỹ, 2010–2014, khí tượng từ sân bay. **Beijing Multi-Site** (UCI 501) là bộ của ta. Nhiều bài (ví dụ Du et al., IEEE TKDE 2021, "Deep Air Quality Forecasting Using Hybrid Deep Learning Framework") dùng bộ 381. **Không được** đặt số liệu của các bài đó cạnh kết quả của ta.

| # | Bài | Thiết lập | Kết quả / ghi chú | So được với ta? |
|---|---|---|---|---|
| B1 | Garg & Jindal (2021), *Evaluation of Time Series Forecasting Models for Estimation of PM2.5 Levels in Air*, I2CT 2021, arXiv:2104.03226 | 12 trạm; chia theo thời gian 75/25 (train 03/2013–02/2016, test 03/2016–02/2017); ARIMA, Prophet, LSTM, 1D-CNN | LSTM tốt nhất: RMSE TB 20,8, MAE 13,2. **Bài không nêu rõ tầm dự báo**, và có chỗ mâu thuẫn về kích thước dữ liệu (35.064 dòng giờ so với 1.421 dòng ngày) | ✗ Chỉ để trích dẫn. Lưu ý: test của bài ≈ val Bắc Kinh của ta |
| B2 | Li, D., Liu, J., Zhao, Y. (2022), *Forecasting of PM2.5 Concentration in Beijing Using Hybrid Deep Learning Framework Based on Attention Mechanism*, Applied Sciences 12(21): 11155, DOI 10.3390/app122111155 | 12 trạm UCI, 01/03/2013–28/02/2017; mô hình FPHFA: 1D-CNN đa kênh (không gian) + BiLSTM + attention; tầm 1–12h và 13–96h | ⚠ Chưa đọc được bảng số liệu (Bảng 3), cần đọc toàn văn | △ Cùng dữ liệu, nhưng cách chia và độ đo cần kiểm tra |
| B3 | *Interpretable short-term PM2.5 forecasting using meteorology–pollution coupling across multiple Beijing monitoring stations*, Environmental Monitoring and Assessment (2026), DOI 10.1007/s10661-026-15799-9 | 420.768 giờ, 12 trạm; **kiểm định thời gian chống rò rỉ** (leakage-safe), **chuyển giao giữa các trạm**, khả năng giải thích; tầm 1h và 6h (6h là thiết lập chính) | ⚠ Chưa lấy được toàn văn (tên tác giả, số liệu). Rất gần với cách làm của ta (chống rò rỉ + chuyển giao), **nên đọc kỹ** | △ Có thể so ở tầm 6h nếu cùng cách chia |
| B4 | Tan, Bergmeir, Petitjean, Webb (2021), *Time series extrinsic regression*, Data Mining and Knowledge Discovery 35(3): 1032–1060, DOI 10.1007/s10618-021-00745-9 | Kho TSER Monash/UEA/UCR có bộ BeijingPM25Quality / BeijingPM10Quality tạo từ UCI 501 (Zenodo 3902671) | Bài toán **hồi quy ngoại sinh** (dự đoán PM từ chuỗi các biến khác), KHÔNG phải dự báo tương lai | ✗ Chỉ để trích dẫn rằng bộ này là benchmark chuẩn |
| B5 | Bekkar et al. (2021), *Air-pollution prediction in smart city, deep learning approach*, Journal of Big Data (2021), DOI 10.1186/s40537-021-00548-1 | CNN-LSTM dự báo PM2.5 theo giờ ở Bắc Kinh, có dùng PM2.5 của các trạm lân cận | ⚠ Chưa xác minh đây là bộ 501 hay bộ khác (trang PMC bị chặn khi tải). Ý tưởng "trạm lân cận" tương ứng ablation (d) của ta | ? |

**Mốc nội bộ (cùng giao thức với đồ án, tự tính, dùng chắc chắn được)** — baseline trên tập val Bắc Kinh (03/2016–02/2017), chỉ tính giờ đo thật, cửa sổ L_ref = 72 (`results/beijing_reference.csv`):

| Tầm | Persistence MAE / RMSE (µg/m³) | Trung bình theo giờ MAE / RMSE |
|---|---|---|
| t+1 | 10,2 / 19,3 | 59,9 / 82,5 |
| t+6 | 32,0 / 54,5 | 60,0 / 82,6 |
| t+24 | 58,5 / 88,2 | 60,0 / 82,8 |

→ Đây là "thước đo độ khó": ở Bắc Kinh persistence rất mạnh ở t+1 (ACF lag 1h = 0,97) nhưng sụp ở t+24. Nếu PatchTST pretrain trên Bắc Kinh mà không thắng các con số này trên val Bắc Kinh thì phần pretrain có vấn đề.

## 4. Truy vấn để tìm thêm (Google Scholar / ACL Anthology không có bài môi trường, dùng Scholar)

- `"Beijing Multi-Site Air-Quality" forecasting PM2.5 hourly`
- `"Beijing multi-site" PM2.5 transformer OR PatchTST OR informer`
- `PM2.5 forecasting "transfer learning" Beijing source target city`
- `"10.24432/C5RK5G"` (các bài trích dẫn DOI của bộ dữ liệu)
- Trên Scholar: mở trang bài Zhang et al. 2017 → "Cited by" → lọc "forecast".

## 5. Việc tiếp theo

- [ ] Đọc toàn văn B2 và B3; điền tầm dự báo, cách chia và MAE/RMSE vào bảng.
- [ ] Thêm các mục đã xác minh vào `references.bib` (đã thêm B1, B2, B4 và các mục có DOI chắc chắn).
