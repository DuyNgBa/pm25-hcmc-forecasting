# Data dictionary: healthyair_clean.parquet

Khoá: (`city`, `station_id`, `time`). Giá trị thiếu: `NaN`. Sinh bởi `src/data/clean_healthyair.py` với cấu hình `configs/cleaning.json`.

| Cột | Kiểu | Đơn vị | Ý nghĩa | Cách xử lý |
|---|---|---|---|---|
| `city` | string |  | Thành phố, luôn là `hcm` | Thêm mới |
| `station_id` | string |  | Mã trạm `hcm_01`…`hcm_06` (từ `Station_No`) | Đổi tên |
| `time` | datetime64 | giờ địa phương (UTC+7) | Mốc đầu giờ; giá trị là trung bình 01:00–01:59 cho dòng 01:00 | Lưới giờ đầy đủ; giờ thiếu = NaN |
| `pm25` | float | µg/m³ | PM2.5, biến mục tiêu | 0 → NaN; điểm nhảy bị đánh dấu loại → NaN; KHÔNG cắt outlier; nội suy khoảng trống ≤ 3 giờ |
| `tsp` | float | µg/m³ | Tổng bụi lơ lửng | ≤ 0 → NaN; cắt outlier; nội suy khoảng ngắn. Không dùng làm đặc trưng chính |
| `o3` | float | µg/m³ | Ozone (gốc ppb × 1,963) | 0 → NaN; cắt outlier; nội suy khoảng ngắn |
| `co` | float | µg/m³ | CO (gốc ppm × 1146) | 0 → NaN; cắt outlier; nội suy khoảng ngắn |
| `no2` | float | µg/m³ | NO2 (gốc ppm × 1882) | 0 → NaN; trạm 5 trong khoảng bão hoà → NaN; cắt outlier; nội suy khoảng ngắn |
| `so2` | float | µg/m³ | SO2 (gốc ppm × 2620) | Cắt outlier; nội suy khoảng ngắn. Không dùng làm đặc trưng chính (nghi sai lệch cảm biến) |
| `temp_sensor` | float | °C | Nhiệt độ cảm biến tại trạm | Trạm 4 đóng băng → NaN; cắt outlier; nội suy khoảng ngắn. Chỉ dùng để kiểm tra khớp giờ với Open-Meteo |
| `rh_sensor` | float | % | Độ ẩm cảm biến tại trạm | Như temp_sensor |
| `is_lockdown` | int8 | 0/1 | 1 trong thời gian giãn cách theo Chỉ thị 16 toàn TP.HCM | 2021-07-09 00:00 đến trước 2021-10-01 00:00 |
| `is_imputed` | int8 | 0/1 | 1 nếu `pm25` được nội suy (không phải giá trị đo thật) | Không dùng các dòng is_imputed = 1 làm nhãn khi đánh giá |
