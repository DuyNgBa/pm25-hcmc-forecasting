# Data dictionary: khí tượng Open-Meteo (TV2)

Sinh bởi `src/data/meteo_features.py`, `src/data/merge_meteo.py`, `src/data/meteo_future.py`; cấu hình `configs/openmeteo.yaml`.
Khoá: (`city`, `station_id`, `time`); `time` là giờ địa phương (HCM UTC+7, Bắc Kinh UTC+8), không kèm múi giờ.

## meteo_hcm.parquet, meteo_beijing.parquet

Tên 9 biến gốc giữ nguyên như API (để `src/data/schema.py` của TV3 đọc được). Trong `hcm_merged.parquet`, các tên này được đổi sang tên ngắn ở cột thứ hai.

| Cột (API) | Tên ngắn | Đơn vị | Ý nghĩa |
|---|---|---|---|
| `city` | | | `hcm` hoặc `beijing` |
| `station_id` | | | HCM: `hcm_01`..`hcm_06` (= Station_No 1..6); Bắc Kinh: tên trạm PRSA |
| `time` | | | Mốc giờ địa phương |
| `temperature_2m` | `temp` | °C | Nhiệt độ 2 m, tức thời tại mốc giờ |
| `relative_humidity_2m` | `rh` | % | Độ ẩm tương đối 2 m |
| `dew_point_2m` | `dew` | °C | Điểm sương 2 m |
| `precipitation` | `precip` | mm | Tổng mưa của giờ **trước** mốc (T−1 → T) |
| `surface_pressure` | `pressure` | hPa | Áp suất bề mặt |
| `cloud_cover` | `cloud_cover` | % | Tổng mây |
| `wind_speed_10m` | `wind_speed` | m/s | Tốc độ gió 10 m |
| `wind_direction_10m` | `wind_dir` | độ | Hướng gió thổi **tới từ** (360 = Bắc, 90 = Đông) |
| `wind_gusts_10m` | `wind_gusts` | m/s | Gió giật 10 m |
| `wind_u` | | m/s | `−ws·sin(dir)`; dương = gió thổi về phía Đông |
| `wind_v` | | m/s | `−ws·cos(dir)`; dương = gió thổi về phía Bắc |
| `wind_dir_sin`, `wind_dir_cos` | | | Hướng gió mã hoá vòng tròn |
| `precip_6h`, `precip_24h` | | mm | Tổng mưa 6 / 24 giờ qua, gồm giờ hiện tại (NaN ở 5/23 giờ đầu chuỗi) |
| `hours_since_rain` | | giờ | Số giờ kể từ giờ gần nhất có mưa ≥ 0,1 mm, chặn trên 168 |
| `wind_speed_6h` | | m/s | Tốc độ gió trung bình 6 giờ qua (NaN ở 5 giờ đầu chuỗi) |

## hcm_merged.parquet

= `healthyair_clean.parquet` của TV1 (xem `data/data_dictionary.md` trên nhánh của TV1) + các cột khí tượng ở trên (tên ngắn) + cột `area_type` (urban_background, traffic, industrial, residential, traffic_residential).

## meteo_hcm_future.parquet (G1-40, chỉ dùng cho thiết lập phụ)

`<biến>_fut{h}` = giá trị của biến tại t+h giờ, cùng trạm, với h ∈ {1, 6, 24} và biến ∈ {temperature_2m, relative_humidity_2m, dew_point_2m, precipitation, surface_pressure, wind_speed_10m, wind_u, wind_v}. Cần tầm khác thì dùng `add_future_meteo()` / `future_block()` trong `src/data/meteo_future.py`.

## data/stations/

| File | Cột |
|---|---|
| `stations_hcm.csv` | station_id, station_no, lat, lon, area_type, lat_proposal, lon_proposal, diff_m (lệch so với đề xuất), source |
| `stations_beijing.csv` | station_id, abbr, lat, lon, elevation_m, location, source (Wu, Xie & Kang 2022, Table 1) |
