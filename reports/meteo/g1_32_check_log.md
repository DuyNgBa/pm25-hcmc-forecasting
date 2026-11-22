# G1-32. Log kiểm tra dữ liệu Open-Meteo tải về

Sinh tự động bởi `python -m src.data.fetch_openmeteo --verify` lúc 2026-11-22 19:06.

## Tiêu chí

- **Đủ giờ:** số dòng = số giờ từ `start_date 00:00` đến `end_date 23:00`; bước thời gian luôn đúng 1 giờ.
- **Không trùng:** không có mốc thời gian lặp.
- **Không thiếu:** không có ô `null` ở 9 biến.
- **Đơn vị đúng** (G1-28, đối chiếu tài liệu Historical Weather API, gió đặt `wind_speed_unit=ms`):

| Biến | Đơn vị mong đợi |
|---|---|
| `temperature_2m` | °C |
| `relative_humidity_2m` | % |
| `dew_point_2m` | °C |
| `precipitation` | mm |
| `surface_pressure` | hPa |
| `cloud_cover` | % |
| `wind_speed_10m` | m/s |
| `wind_direction_10m` | ° |
| `wind_gusts_10m` | m/s |

- **Giá trị hợp lý về vật lý:**

| Biến | Min | Max |
|---|---|---|
| `temperature_2m` | -40 | 50 |
| `relative_humidity_2m` | 0 | 100 |
| `dew_point_2m` | -60 | 35 |
| `precipitation` | 0 | 200 |
| `surface_pressure` | 900 | 1100 |
| `cloud_cover` | 0 | 100 |
| `wind_speed_10m` | 0 | 60 |
| `wind_direction_10m` | 0 | 360 |
| `wind_gusts_10m` | 0 | 80 |

## Kết quả từng trạm

| city    | station_id    |   rows |   rows_expected | start_ok   | end_ok   | hourly_step_ok   |   n_duplicated |   n_missing_total | units_ok   |   n_out_of_range | tz_ok   |   grid_lat |   grid_lon | ok   |
|:--------|:--------------|-------:|----------------:|:-----------|:---------|:-----------------|---------------:|------------------:|:-----------|-----------------:|:--------|-----------:|-----------:|:-----|
| hcm     | hcm_01        |  11784 |           11784 | True       | True     | True             |              0 |                 0 | True       |                0 | True    |    10.8612 |    106.805 | True |
| hcm     | hcm_02        |  11784 |           11784 | True       | True     | True             |              0 |                 0 | True       |                0 | True    |    10.7206 |    106.617 | True |
| hcm     | hcm_03        |  11784 |           11784 | True       | True     | True             |              0 |                 0 | True       |                0 | True    |    10.7909 |    106.631 | True |
| hcm     | hcm_04        |  11784 |           11784 | True       | True     | True             |              0 |                 0 | True       |                0 | True    |    10.7909 |    106.711 | True |
| hcm     | hcm_05        |  11784 |           11784 | True       | True     | True             |              0 |                 0 | True       |                0 | True    |    10.7909 |    106.711 | True |
| hcm     | hcm_06        |  11784 |           11784 | True       | True     | True             |              0 |                 0 | True       |                0 | True    |    10.7909 |    106.631 | True |
| beijing | Aotizhongxin  |  35184 |           35184 | True       | True     | True             |              0 |                 0 | True       |                0 | True    |    39.9648 |    116.397 | True |
| beijing | Changping     |  35184 |           35184 | True       | True     | True             |              0 |                 0 | True       |                0 | True    |    40.1757 |    116.255 | True |
| beijing | Dingling      |  35184 |           35184 | True       | True     | True             |              0 |                 0 | True       |                0 | True    |    40.246  |    116.292 | True |
| beijing | Dongsi        |  35184 |           35184 | True       | True     | True             |              0 |                 0 | True       |                0 | True    |    39.9648 |    116.397 | True |
| beijing | Guanyuan      |  35184 |           35184 | True       | True     | True             |              0 |                 0 | True       |                0 | True    |    39.8946 |    116.36  | True |
| beijing | Gucheng       |  35184 |           35184 | True       | True     | True             |              0 |                 0 | True       |                0 | True    |    39.8946 |    116.234 | True |
| beijing | Huairou       |  35184 |           35184 | True       | True     | True             |              0 |                 0 | True       |                0 | True    |    40.3163 |    116.582 | True |
| beijing | Nongzhanguan  |  35184 |           35184 | True       | True     | True             |              0 |                 0 | True       |                0 | True    |    39.8946 |    116.485 | True |
| beijing | Shunyi        |  35184 |           35184 | True       | True     | True             |              0 |                 0 | True       |                0 | True    |    40.1757 |    116.634 | True |
| beijing | Tiantan       |  35184 |           35184 | True       | True     | True             |              0 |                 0 | True       |                0 | True    |    39.8946 |    116.36  | True |
| beijing | Wanliu        |  35184 |           35184 | True       | True     | True             |              0 |                 0 | True       |                0 | True    |    39.9648 |    116.271 | True |
| beijing | Wanshouxigong |  35184 |           35184 | True       | True     | True             |              0 |                 0 | True       |                0 | True    |    39.8946 |    116.36  | True |

**Tổng kết: 18/18 trạm đạt mọi tiêu chí.**

## Ô lưới ERA5 dùng chung

Nhiều trạm gần nhau rơi vào cùng một ô lưới nên có khí tượng giống hệt nhau:

| city    |   grid_lat |   grid_lon | station_id                       |
|:--------|-----------:|-----------:|:---------------------------------|
| beijing |    39.8946 |    116.234 | Gucheng                          |
| beijing |    39.8946 |    116.36  | Guanyuan, Tiantan, Wanshouxigong |
| beijing |    39.8946 |    116.485 | Nongzhanguan                     |
| beijing |    39.9648 |    116.271 | Wanliu                           |
| beijing |    39.9648 |    116.397 | Aotizhongxin, Dongsi             |
| beijing |    40.1757 |    116.255 | Changping                        |
| beijing |    40.1757 |    116.634 | Shunyi                           |
| beijing |    40.246  |    116.292 | Dingling                         |
| beijing |    40.3163 |    116.582 | Huairou                          |
| hcm     |    10.7206 |    106.617 | hcm_02                           |
| hcm     |    10.7909 |    106.631 | hcm_03, hcm_06                   |
| hcm     |    10.7909 |    106.711 | hcm_04, hcm_05                   |
| hcm     |    10.8612 |    106.805 | hcm_01                           |
