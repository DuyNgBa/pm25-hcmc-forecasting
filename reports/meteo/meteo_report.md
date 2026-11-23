# Báo cáo dữ liệu khí tượng Open-Meteo (TV2 – Minh Trí)

Phạm vi: các task **G1-26 → G1-44** trong checklist phân công (GĐ1, Minh Trí). Mọi con số dưới đây do các script trong `src/` sinh tự động. Chạy lại theo mục 8 sẽ ra đúng các số này.

**Nguồn khí tượng:** Open-Meteo Historical Weather API với **`models=era5_seamless`** (ERA5-Land 0,1° ghép ERA5 0,25°), dùng **cùng một nguồn cho cả TP.HCM và Bắc Kinh**. Bản tải ngày 22/11/2026 không ghi `models` nên API dùng `best_match`; với TP.HCM giai đoạn 2021–2022, `best_match` trộn thêm mô hình phân giải cao khác, còn Bắc Kinh 2013–2017 vốn đã là ERA5. Sau recheck của nhóm, nguồn được cố định và tải lại ngày 23/11/2026. Bắc Kinh gần như không đổi (tương quan với khí tượng gốc lệch < 0,001); TP.HCM đổi rõ, nhất là mưa và gió (tương quan giữa bản cũ và bản mới: nhiệt độ 0,95, gió 0,75, mưa từng giờ chỉ 0,29). Mọi số liệu dưới đây là của bản `era5_seamless`.

| ID | Việc | Đầu ra | Trạng thái |
|---|---|---|---|
| G1-26 | Toạ độ 6 trạm TP.HCM, đối chiếu đề xuất | `data/stations/stations_hcm.csv` | ✔ |
| G1-27 | Toạ độ 12 trạm Bắc Kinh + nguồn | `data/stations/stations_beijing.csv` | ✔ |
| G1-28 | Chốt 9 biến + đơn vị theo tài liệu API | `configs/openmeteo.yaml` (`hourly_vars`, `expected_units`) | ✔ |
| G1-29 | Script `fetch_openmeteo.py` (tham số hoá, cache JSON thô, retry, cố định `models`) | `src/data/fetch_openmeteo.py` | ✔ tải lại ra dữ liệu giống hệt |
| G1-30 | Tải 6 trạm TP.HCM | `data/raw/openmeteo/hcm_*.json` | ✔ |
| G1-31 | Tải 12 trạm Bắc Kinh | `data/raw/openmeteo/beijing_*.json` | ✔ |
| G1-32 | Kiểm tra dữ liệu tải về | `reports/meteo/g1_32_check_log.md` | ✔ 18/18 trạm đạt |
| G1-33 | Left join với HealthyAir sạch | `data/interim/hcm_merged.parquet` | ✔ |
| G1-34 | Kiểm tra khớp giờ (tương quan chéo) | `g1_34_lag_hcm.csv` + hình | ✔ |
| G1-35 | Kiểm chứng bằng khí tượng gốc Bắc Kinh | `g1_35_beijing_validation.csv` | ✔ |
| G1-36 | `wind_u`, `wind_v`, sin/cos hướng gió | cột trong `meteo_*.parquet`; test ví dụ tay | ✔ |
| G1-37 | Mưa 6h, 24h, số giờ kể từ trận mưa gần nhất (ngưỡng 0,1 mm/h) | cột trong bảng merged | ✔ |
| G1-38 | Tốc độ gió trung bình 6h | cột trong bảng merged | ✔ |
| G1-39 | Test đặc trưng cuộn không nhìn trước | `tests/test_meteo_features.py` | ✔ 18 test đạt |
| G1-40 | Khí tượng tương lai (thiết lập "dự báo thời tiết hoàn hảo") | `src/data/meteo_future.py` + `meteo_hcm_future.parquet` | ✔ |
| G1-41 | Xuất `meteo_hcm`, `meteo_beijing`, `hcm_merged` | `data/interim/` | ✔ đã chạy thử với `schema.py` của TV3 |
| G1-42 | Tương quan PM2.5 với từng biến, theo trạm | `g1_42_*` | ✔ |
| G1-43 | PM2.5 khi mưa / không mưa; theo khoảng gió | `g1_43_*` | ✔ |
| G1-44 | Hoa gió từng trạm | `g1_44_*` | ✔ |

## 1. Toạ độ trạm (G1-26, G1-27)

- **TP.HCM:** toạ độ lấy từ Bảng 1 bài Data in Brief [1]. Hai cột latitude/longitude bị ghi ngược tên trong bài gốc, đã đổi lại. Lệch so với bảng trong đề xuất chỉ 2–6 m, do đề xuất làm tròn 4 chữ số.
- **Bắc Kinh:** bộ UCI chỉ có tên trạm. Toạ độ (kèm độ cao, mô tả vị trí) lấy từ **Bảng 1 của Wu, Xie & Kang (2022), *npj Urban Sustainability* 2:23, https://doi.org/10.1038/s42949-022-00070-0**. Bảng này liệt kê 32 trạm quan trắc Bắc Kinh và đủ cả 12 trạm của bộ UCI. Đã đối chiếu thêm với các nguồn khác cho Huairou, Wanshouxigong, Aotizhongxin, Dingling: các nguồn khớp nhau trong phạm vi vài trăm mét, nhỏ hơn nhiều so với ô lưới ERA5-Land (0,1°, khoảng 11 km).

## 2. Tải dữ liệu (G1-28 → G1-32)

| Mục | Giá trị |
|---|---|
| API | `https://archive-api.open-meteo.com/v1/archive`, tham số `models=era5_seamless` (tái phân tích ERA5-Land + ERA5, ECMWF), khai báo ở `configs/openmeteo.yaml` (`model`) |
| 9 biến (G1-28) | `temperature_2m` °C, `relative_humidity_2m` %, `dew_point_2m` °C, `precipitation` mm, `surface_pressure` hPa, `cloud_cover` %, `wind_speed_10m` m/s, `wind_direction_10m` °, `wind_gusts_10m` m/s |
| TP.HCM (G1-30) | 20/02/2021 → 25/06/2022, `Asia/Bangkok`, 11.784 giờ/trạm |
| Bắc Kinh (G1-31) | 26/02/2013 → 02/03/2017 (đệm 3 ngày mỗi đầu), `Asia/Shanghai`, 35.184 giờ/trạm |
| Cache | `data/raw/openmeteo/<city>_<trạm>.json` = phản hồi JSON **thô** của API; `DOWNLOAD_INFO.json` ghi URL request, ô lưới thật, md5 nội dung |
| Tái lập (G1-29) | Tải lại 6 trạm TP.HCM ngay sau lần tải đầu (23/11/2026): **md5 nội dung giống hệt**. Lần tải 22/11 (chưa cố định `models`) cũng đã tải lại 18/18 trạm ra số liệu giống hệt |

**Kiểm tra (G1-32, `g1_32_check_log.md`):** cả 18/18 trạm đạt mọi tiêu chí. Đủ giờ từ `start 00:00` đến `end 23:00`, bước đúng 1 giờ, 0 mốc trùng, 0 ô thiếu, đơn vị khớp `expected_units`, giá trị nằm trong khoảng vật lý hợp lý, timezone đúng.

**Ô lưới dùng chung** (lưới 0,1°): 6 trạm TP.HCM chỉ rơi vào **4 ô lưới**: S4, S5 và S6 trùng một ô (10,8° N, 106,7° E); S1, S2, S3 mỗi trạm một ô. Ở Bắc Kinh, Dongsi, Tiantan và Wanshouxigong trùng một ô; 9 trạm còn lại mỗi trạm một ô. Khí tượng các trạm trùng ô là giống hệt nhau. Đề xuất (mục 3.2) đã lường trước điều này: khác biệt giữa các trạm nằm ở PM2.5 đo trực tiếp.

## 3. Ghép với HealthyAir (G1-33)

Left join `healthyair_clean.parquet` (TV1) với `meteo_hcm.parquet` theo `(city, station_id, time)`, trong đó `station_id = hcm_01..hcm_06` ứng với `Station_No = 1..6`, kiểm tra quan hệ 1–1. Kết quả `hcm_merged.parquet`: **69.534 dòng × 31 cột**, 0 dòng thiếu khí tượng. Trong bảng ghép, tên biến đã đổi sang tên ngắn (`temp, rh, dew, precip, pressure, wind_speed, wind_dir, ...`).

Đối chiếu nhiệt độ hai nguồn (`g1_33_merge_check.csv`): r = 0,80–0,85 ở trạm 1, 2, 3, 6. Open-Meteo **thấp hơn cảm biến 1,6–2,8 °C**, phù hợp với hiệu ứng đảo nhiệt đô thị và việc cảm biến đặt sát mặt đường. Đây không phải lỗi lệch giờ.

## 4. Kiểm tra khớp giờ (G1-34)

Tương quan giữa nhiệt độ/độ ẩm cảm biến HealthyAir (đã bỏ giá trị 0 và đoạn đóng băng ≥ 24 giờ) và Open-Meteo dịch k giờ, k từ −8 đến +8 (`g1_34_lag_hcm.csv`, `figures/g1_34_lag_hcm.png`):

| Trạm | n | k tốt nhất (nhiệt độ / độ ẩm) | r tại k tốt nhất | r tại k = 0 |
|---|---|---|---|---|
| hcm_01 | 7.892 | +1 / +1 | 0,852 | 0,838 |
| hcm_02 | 9.357 | +1 / +1 | 0,849 | 0,813 |
| hcm_03 | 8.418 | +1 / +1 | 0,819 | 0,797 |
| hcm_06 | 9.499 | 0 / 0 | 0,854 | 0,854 |
| hcm_04, hcm_05 | 1.837 / 2.994 | cảm biến kém (r ≈ 0,48–0,49) | | |

**Kết luận:** đỉnh tương quan ở k = 0 hoặc k = +1; ở k = +1 thì chênh với k = 0 không quá 0,04 (bảng ghi r của nhiệt độ). Lệch +1 là đúng về vật lý: HealthyAir dòng T là trung bình T:00–T:59 (tâm ở T+30 phút), còn Open-Meteo là giá trị tức thời lúc T. Đường cong tương quan đối xứng quanh 0 và giảm rất nhanh khi |k| ≥ 2. Không có lệch múi giờ: **HealthyAir dùng giờ Việt Nam (UTC+7), ghép thẳng với Open-Meteo `Asia/Bangkok`.**

## 5. Kiểm chứng bằng khí tượng gốc Bắc Kinh (G1-35)

Trung bình 12 trạm (`g1_35_beijing_validation.csv`):

| Gốc PRSA | Open-Meteo | r | Bias (OM − gốc) | MAE |
|---|---|---|---|---|
| TEMP | temperature_2m | **0,983** | −0,81 °C | 1,81 °C |
| DEWP | dew_point_2m | **0,979** | −0,39 °C | 2,16 °C |
| PRES | surface_pressure | **0,990** | −1,45 hPa | 1,99 hPa |
| WSPM | wind_speed_10m | 0,651 | +0,30 m/s | 0,82 m/s |
| RAIN | precipitation | 0,335 | ≈ 0 | 0,09 mm |

Lệch giờ tốt nhất của nhiệt độ bằng 0 ở cả 12/12 trạm. Nhiệt độ, điểm sương và áp suất khớp rất tốt; gió khớp vừa phải, vì gió tại trạm chịu ảnh hưởng của công trình xung quanh. Mưa **từng giờ** khớp kém (mưa đối lưu cục bộ, lưới thô) nên ưu tiên mưa **tích luỹ**. Kết luận: Open-Meteo đủ tin cậy làm **nguồn khí tượng chung** cho hai thành phố, dẫn chứng cho mục 4.3 của đề xuất. Ghi chú thêm: trong bộ PRSA, Changping và Dingling có cột RAIN giống hệt 100% (cùng trạm thời tiết gần nhất).

## 6. Đặc trưng (G1-36 → G1-41)

| Cột | Cách tính | Test |
|---|---|---|
| `wind_u`, `wind_v` | `u = −ws·sin(dir)`, `v = −ws·cos(dir)`; dir là hướng gió thổi **tới từ** | Ví dụ tay: gió Bắc 5 m/s → (0, −5); gió Tây → (+5, 0); gió Nam → (0, +5); gió Tây Nam 2√2 → (2, 2). Khớp công thức `wind_uv` trong `schema.py` của TV3 |
| `wind_dir_sin/cos` | sin, cos của hướng gió | |
| `precip_6h`, `precip_24h` | tổng mưa 6/24 giờ qua, gồm giờ hiện tại | Tính tay trên chuỗi tự tạo và trên dữ liệu thật |
| `hours_since_rain` | số giờ kể từ giờ có mưa ≥ **0,1 mm/h** (độ phân giải của ERA5/ERA5-Land), chặn trên 168 | Ví dụ tay, kiểm tra ngưỡng |
| `wind_speed_6h` | trung bình tốc độ gió 6 giờ qua | Tính tay |

- **G1-39, không nhìn trước:** test đổi ngẫu nhiên toàn bộ dữ liệu sau mốc t (t = 6, 24, 40, 60), kiểm tra mọi đặc trưng tại các giờ ≤ t giữ nguyên. Test cũng báo lỗi nếu chuỗi thiếu giờ, vì khi đó "dịch 1 dòng" không còn là "dịch 1 giờ".
- **Quy ước thời gian:** Open-Meteo tại mốc T là giá trị tức thời; riêng `precipitation` là tổng của giờ T−1 → T. Tại gốc dự báo t, dùng khí tượng có mốc ≤ t là hợp lệ.
- **G1-40, khí tượng tương lai:** `add_future_meteo(df, cols, horizons)` thêm cột `<biến>_fut{h}` (cho mô hình cây, mỗi tầm một mô hình). `future_block(df, origins, cols, H)` trả khối (n, H, n_biến) khí tượng tại t+1..t+H (cho mô hình DL dự báo cả chuỗi). File dựng sẵn `meteo_hcm_future.parquet` có h = 1, 6, 24. **Chỉ dùng cho thiết lập phụ (G2-48)**, không dùng trong thiết lập chính.
- **G1-41:** trong `meteo_hcm.parquet` / `meteo_beijing.parquet`, 9 biến giữ **tên gốc như API**, vì `src/data/schema.py` của TV3 đọc đúng các tên này. Đã chạy thử `schema.py` và `pipeline.prepare("hcm")` của TV3 trên các file này: cả hai thành phố có `meteo_source = open_meteo` và tạo được cửa sổ cho ablation (c).

## 7. EDA: khí tượng và PM2.5 tại TP.HCM (G1-42 → G1-44)

Chỉ dùng **52.329 giờ có PM2.5 đo thật** (`is_imputed = 0`).

### G1-42. Tương quan theo trạm (`figures/g1_42_spearman_heatmap.png`)

| Đặc trưng | ρ (6 trạm) | Khoảng theo trạm | Diễn giải |
|---|---|---|---|
| `wind_speed_6h` | **−0,31** | −0,26 … −0,42 | Gió mạnh làm bụi phát tán; yếu tố mạnh nhất, ổn định ở mọi trạm |
| `wind_v` | −0,28 | −0,19 … −0,36 | Gió mùa Nam/Tây Nam (v > 0, mùa mưa) đi kèm PM2.5 thấp |
| `pressure` | +0,24 | +0,20 … +0,35 | Áp cao (mùa khô, khí quyển ổn định) đi kèm PM2.5 cao |
| `hours_since_rain` | +0,22 | +0,16 … +0,32 | Càng lâu không mưa, bụi càng tích tụ |
| `precip_24h` | −0,22 | −0,12 … −0,36 | Mưa tích luỹ mạnh hơn mưa từng giờ (−0,17) |
| `temp`, `rh`, `cloud_cover` | ≈ −0,05 | đổi dấu theo trạm | Gần như không liên quan trực tiếp |

Trạm 5 (Q.3, giao thông) phản ứng với khí tượng yếu nhất ở mọi biến. Đây là trạm có PM2.5 thấp nhất và thiếu dữ liệu nhiều nhất (62% số giờ có PM2.5). Khí tượng tại t vẫn tương quan với PM2.5 tại t+24 ở mức |ρ| ≈ 0,2 (`g1_42_lead_corr.csv`), nên có ích cho cả tầm dự báo xa. Theo tháng (`g1_42_monthly.png`), PM2.5 cao nhất ở 12/2021–01/2022 (27,5–29,9 µg/m³), khi gần như không mưa; thấp nhất ở 07–09/2021 (13–15 µg/m³), khi mưa khoảng 280–350 mm/tháng. Riêng giai đoạn này trùng giãn cách COVID-19 nên không tách được hoàn toàn tác động của mưa.

### G1-43. Mưa / không mưa và tốc độ gió (`g1_43_rain_vs_dry.png`, `g1_43_wind_rain.png`)

- Có mưa trong 6 giờ qua làm trung vị PM2.5 giảm **18,3%** (19,2 → 15,7 µg/m³), và tỉ lệ giờ vượt 35,4 µg/m³ giảm từ 14,1% xuống 8,2%. Mức giảm ở 4/6 trạm là 18–22%; trạm 4 giảm 14,7%, trạm 5 chỉ 12,8%.
- Theo tốc độ gió TB 6h: trung vị giảm đều từ 24,8 µg/m³ (gió < 1 m/s) xuống 12,3 µg/m³ (≥ 4 m/s); tỉ lệ giờ vượt ngưỡng giảm từ 29% xuống 1%.
- Theo mưa 24h: trung vị 21,1 µg/m³ khi không mưa, 13,5 µg/m³ khi mưa ≥ 20 mm. Từ 10 mm trở lên, mức giảm chững lại.

### G1-44. Hoa gió từng trạm (`g1_44_wind_rose.png`)

- Hướng gió phổ biến nhất ở mọi trạm là **Đông Nam** (24–31% số giờ), tiếp theo là Tây Nam (gió mùa).
- Ở **cả 6 trạm**, gió từ **Bắc, Đông Bắc, Tây Bắc** (gió mùa khô) mang PM2.5 cao nhất; gió từ Tây Nam và Tây thấp nhất. Ví dụ trạm 3 (KCN Tân Bình): 28–29 µg/m³ khi gió Bắc/Tây Bắc so với 15 µg/m³ khi gió Tây.
- Trạm 4 (Thanh Đa) có PM2.5 cao ở hầu hết các hướng, kể cả Đông/Đông Nam (22–26 µg/m³), cho thấy ảnh hưởng của nguồn phát thải cục bộ hoặc vận chuyển từ trung tâm thành phố.
- Tỉ lệ lặng gió (< 0,5 m/s) chỉ 2,4–2,8%.

**Gợi ý cho mô hình (RQ2):** ưu tiên `wind_speed_6h`, `wind_u/v`, `precip_24h`, `hours_since_rain`, `pressure`, `dew`. **Đề xuất với TV3:** thêm `hours_since_rain` và `wind_speed_6h` vào danh sách `METEO` của `schema.py`. Hiện lược đồ chung chưa có hai cột này, dù chúng thuộc nhóm tương quan mạnh nhất.

## 8. Cách chạy lại

```bash
pip install -r requirements.txt
set PYTHONUTF8=1                                  # Windows: để in tiếng Việt
python -m src.data.stations                       # G1-26/27
python -m src.data.fetch_openmeteo                # G1-29..32: tải (bỏ qua trạm đã có cache) + log kiểm tra; --force để tải lại
python -m src.data.meteo_features                 # G1-36..38, G1-41
python -m src.data.meteo_future                   # G1-40
python -m src.data.check_meteo                    # G1-34, G1-35 (cần data/raw/beijing)
python -m src.data.merge_meteo                    # G1-33 (cần data/interim/healthyair_clean.parquet của TV1)
python -m src.eda.eda_meteo                       # G1-42..44
python -m pytest tests -q                         # G1-36, G1-39, G1-40
```

## Tài liệu tham khảo

1. Rakholia, R., Le, Q., Vu, K.H.N., Ho, B.Q., Simon Carbajo, R. (2023). Outdoor air quality data for spatiotemporal analysis and air quality modelling in Ho Chi Minh City, Vietnam: A part of HealthyAir Project. *Data in Brief*, 46, 108774. https://doi.org/10.1016/j.dib.2022.108774
2. Wu, L., Xie, J., Kang, K. (2022). Changing weekend effects of air pollutants in Beijing under 2020 COVID-19 lockdown controls. *npj Urban Sustainability*, 2, 23. https://doi.org/10.1038/s42949-022-00070-0
3. Zippenfenig, P. Open-Meteo.com Weather API. https://open-meteo.com. Dữ liệu nền (`models=era5_seamless`): Hersbach, H. et al. (2020). The ERA5 global reanalysis. *QJRMS*, 146(730), 1999–2049; Muñoz-Sabater, J. et al. (2021). ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. *Earth System Science Data*, 13, 4349–4383.
