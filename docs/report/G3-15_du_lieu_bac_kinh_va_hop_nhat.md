# [Bản nháp] Chương Dữ liệu — phần Bắc Kinh và hợp nhất (G3-15, phần của TV3)

*Số liệu lấy từ `reports/eda/g1_64_beijing_vs_hcm.csv`, `reports/beijing_missing_before_after.csv`. **Phần TP.HCM của bảng hiện tính trên bản làm sạch tạm; khi có data-v1.0 phải chạy lại `python -m src.eda.compare_cities` và cập nhật các con số đánh dấu (\*).***

## X.3 Bộ dữ liệu Beijing Multi-Site Air Quality

Bộ dữ liệu gồm số liệu theo giờ của 12 trạm quan trắc cấp quốc gia tại Bắc Kinh, từ 01/03/2013 đến 28/02/2017 (35.064 giờ mỗi trạm, tổng cộng 420.768 dòng) [chen2017beijing; zhang2017cautionary]. Mỗi dòng có:

- 6 chất ô nhiễm;
- 6 biến khí tượng, lấy từ trạm khí tượng gần nhất của Cục Khí tượng Trung Quốc.

Bộ dữ liệu không có độ ẩm tương đối, và hướng gió chỉ được ghi theo 16 hướng rời rạc.

**Chuẩn bị** (`src/data/beijing.py`):

- Gộp 12 file và tạo khoá `(city, station_id, time)`, với `time` là giờ địa phương UTC+8. Kiểm tra không có khoá trùng và lưới giờ đầy đủ.
- PM2.5 thiếu 1,1–2,7% mỗi trạm.
- Áp dụng **đúng các quy tắc làm sạch đã dùng cho HealthyAir**:
  - giá trị ≤ 0 được coi là thiếu (Bắc Kinh thực tế không có);
  - PM2.5 **không bị cắt outlier** (giá trị lớn nhất 999 µg/m³ được giữ), giống cách xử lý PM2.5 TP.HCM;
  - nội suy tuyến tính khoảng trống ≤ 3 giờ và đánh dấu bằng cờ `is_imputed`.
- Sau làm sạch, tỉ lệ thiếu còn 0,5–1,9% (Bảng Y).

## X.4 Khác biệt giữa hai bộ dữ liệu (Bảng 4.1 cập nhật)

| Tiêu chí | Bắc Kinh | TP.HCM\* | Hệ quả khi hợp nhất |
|---|---|---|---|
| Quy mô | 12 trạm × 35.064 giờ | 6 trạm × 11.589 giờ (lưới) | Bắc Kinh nhiều hơn khoảng 8 lần, dễ lấn át nếu train chung → S3 cân bằng mẫu |
| Giờ thiếu sau làm sạch | 2,1% | 24,7% | TP.HCM có nhiều khối mất dài |
| PM2.5 trung bình / trung vị | 79,7 / 55,0 µg/m³ | 21,1 / 17,5 µg/m³ | Chênh khoảng 3–4 lần → log rồi z-score theo trạm |
| Phân vị 95 | 242 µg/m³ | 46,1 µg/m³ | |
| % giờ > 35,4 µg/m³ | 62,7% | 11,3% | Ở Bắc Kinh "ô nhiễm cao" là trạng thái thường gặp |
| Đỉnh / đáy chu kỳ ngày | 22h / 7h | 7h / 16h | Hai chu kỳ gần như **ngược pha** → quan hệ giờ–PM2.5 không chuyển giao được |
| Tháng cao nhất / thấp nhất | 12 / 8 | 1 / 9 | Cả hai cao vào cuối năm, nhưng cơ chế khác nhau (sưởi ấm mùa đông so với mùa khô) |
| Tương quan TB giữa các trạm | 0,89 | 0,59 | Bắc Kinh do ô nhiễm vùng chi phối; TP.HCM chủ yếu do nguồn cục bộ |
| ACF lag 1h / 24h | 0,97 / 0,40 | 0,88 / 0,43 | Chuỗi TP.HCM nhiễu hơn ở ngắn hạn |

Hình ACF cho thấy một khác biệt mà bảng không thể hiện được. ACF của TP.HCM **dao động theo chu kỳ 24 giờ**, với các đỉnh ở độ trễ 24 và 48 giờ. ACF của Bắc Kinh thì giảm gần như đơn điệu. Như vậy PM2.5 ở TP.HCM bị chi phối mạnh bởi nhịp sinh hoạt hằng ngày, còn ở Bắc Kinh là các đợt ô nhiễm kéo dài nhiều ngày. Đây là lý do nên thử cửa sổ L ≥ 48 giờ để bao trọn hai chu kỳ ngày.

So với bảng trong đề xuất (tính trên dữ liệu thô), tương quan giữa các trạm TP.HCM tăng từ 0,53 lên 0,59 và ACF lag 1h từ 0,81 lên 0,88. Lý do: khi làm sạch, các điểm nhảy bất thường một giờ (lỗi đo) đã bị loại, nên chuỗi mượt hơn. Kết luận định tính không đổi.

Hình gợi ý:

- `g1_64_cycles.png`: chu kỳ ngày và năm, chuẩn hoá theo trung bình của từng thành phố để hai thành phố dùng chung một trục.
- `g1_64_distribution.png`
- `g1_64_acf.png`

## X.5 Lược đồ chung

Ta không gộp trực tiếp dữ liệu thô của hai bộ. Thay vào đó, cả hai được đưa về **một lược đồ** (Bảng Z, docs/PIPELINE_GUIDE.md §2):

- cùng tập biến, cùng đơn vị;
- khí tượng của **cả hai** thành phố đều lấy từ Open-Meteo, để mô hình không học sự khác biệt giữa hai nguồn đo;
- gió được tách thành hai thành phần u, v theo quy ước khí tượng;
- mưa tích luỹ 6 giờ và 24 giờ chỉ dùng dữ liệu quá khứ;
- giờ và thứ trong tuần được mã hoá sin/cos theo giờ địa phương;
- **không dùng** biến tháng và các chất ô nhiễm khác.

Dữ liệu được nối theo chiều dọc: mỗi trạm là một chuỗi riêng. Cửa sổ huấn luyện được cắt bên trong từng trạm, nên không bao giờ vắt qua hai thành phố. Hai file `hcm_schema.parquet` và `beijing_schema.parquet` có cùng danh sách cột; điều này được kiểm tra tự động.
