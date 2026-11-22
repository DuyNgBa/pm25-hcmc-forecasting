# Mô tả bộ dữ liệu Beijing Multi-Site Air Quality

## 1. Tổng quan

| Mục | Nội dung |
|---|---|
| Nội dung | Nồng độ chất ô nhiễm và thời tiết **theo từng giờ** tại 12 trạm quan trắc ở Bắc Kinh |
| Thời gian | 01/03/2013 00:00 → 28/02/2017 23:00 (đúng 4 năm) |
| Quy mô | 12 trạm × 35.064 giờ = **420.768 dòng** |
| Định dạng | 12 file CSV, mỗi trạm 1 file, tên dạng `PRSA_Data_<Trạm>_20130301-20170228.csv` |
| Nguồn | UCI Machine Learning Repository (bộ số 501), giấy phép CC BY 4.0 |
| Vai trò trong đồ án | **Chỉ dùng để pretrain** (học chuyển giao, RQ3). Không bao giờ dùng để đánh giá kết quả |

Mỗi trạm có **đủ mọi giờ**: không thiếu dòng nào, không trùng thời gian. Chỗ nào thiếu thì là ô trống (NaN) trong dòng đó.

## 2. Một dòng dữ liệu trông như thế nào

```
No,year,month,day,hour,PM2.5,PM10,SO2,NO2,CO,O3,TEMP,PRES,DEWP,RAIN,wd,WSPM,station
1,2013,3,1,0,4,4,4,7,300,77,-0.7,1023,-18.8,0,NNW,4.4,Aotizhongxin
```

Đọc là: 0 giờ ngày 01/03/2013 tại trạm Aotizhongxin, PM2.5 = 4 µg/m³, nhiệt độ −0,7 °C, gió hướng Bắc-Tây-Bắc 4,4 m/s.

## 3. Các cột

| Cột gốc | Tên trong code | Ý nghĩa | Đơn vị | Khoảng giá trị |
|---|---|---|---|---|
| No | — | Số thứ tự dòng | — | — |
| year, month, day, hour | `time` | Thời điểm đo, **giờ địa phương Bắc Kinh** (UTC+8) | — | — |
| **PM2.5** | `pm25` | **Bụi mịn ≤ 2,5 µm, biến cần dự báo** | µg/m³ | 2 – 999 (trung vị 55) |
| PM10 | `pm10` | Bụi ≤ 10 µm | µg/m³ | 2 – 999 |
| SO2, NO2, CO, O3 | `so2`, `no2`, `co`, `o3` | Các khí ô nhiễm | µg/m³ | — |
| TEMP | `temp_native` | Nhiệt độ | °C | −19,9 – 41,6 |
| PRES | `pres_native` | Áp suất không khí | hPa | 982 – 1043 |
| DEWP | `dewp_native` | Nhiệt độ điểm sương | °C | −43,4 – 29,1 |
| RAIN | `rain_native` | Lượng mưa trong giờ | mm | 0 – 72,5 (đa số là 0) |
| wd | `wd_native` | Hướng gió, ghi bằng **chữ**, 16 hướng (N, NNE, NE, …) | — | — |
| WSPM | `wspm_native` | Tốc độ gió | m/s | 0 – 13,2 |
| station | `station_id` | Tên trạm | — | 12 giá trị |

Khí tượng ở đây lấy từ trạm thời tiết **gần nhất** với mỗi trạm đo ô nhiễm, không đo tại chỗ.

**Đồ án chỉ dùng PM2.5** của bộ này. Các khí ô nhiễm khác bị bỏ (TP.HCM đo không đầy đủ). Khí tượng gốc cũng được thay bằng Open-Meteo để hai thành phố dùng cùng một nguồn.

## 4. Mười hai trạm

| Trạm | Khu vực | PM2.5 TB (µg/m³) | Trung vị | % giờ thiếu PM2.5 |
|---|---|---|---|---|
| Aotizhongxin | nội thành | 82,8 | 58 | 2,6 |
| Dongsi | nội thành | 86,2 | 61 | 2,1 |
| Guanyuan | nội thành | 82,9 | 59 | 1,8 |
| Gucheng | nội thành | 83,9 | 60 | 1,8 |
| Nongzhanguan | nội thành | 84,8 | 59 | 1,8 |
| Tiantan | nội thành | 82,2 | 59 | 1,9 |
| Wanliu | nội thành | 83,4 | 59 | 1,1 |
| Wanshouxigong | nội thành | 85,0 | 60 | 2,0 |
| Changping | ngoại thành | 71,1 | 46 | 2,2 |
| Dingling | ngoại thành | 66,0 | 41 | 2,2 |
| Huairou | ngoại thành | 69,6 | 47 | 2,7 |
| Shunyi | ngoại thành | 79,5 | 55 | 2,6 |

Các trạm ngoại thành có PM2.5 thấp hơn rõ rệt so với nội thành.

## 5. Chất lượng dữ liệu

| Vấn đề | Chi tiết | Cách xử lý (giống hệt HealthyAir) |
|---|---|---|
| Thiếu giá trị | PM2.5 thiếu 1,1–2,7% mỗi trạm; khí tượng thiếu < 0,5% | Nội suy tuyến tính khoảng trống **≤ 3 giờ**, đánh dấu `is_imputed = 1`. Khoảng trống dài hơn giữ nguyên NaN. Sau bước này PM2.5 còn thiếu 0,5–1,9% |
| Giá trị rất cao | PM2.5 lên tới 999 µg/m³ (các đợt ô nhiễm nặng mùa đông) | **Giữ nguyên**, không cắt, giống cách nhóm xử lý PM2.5 TP.HCM: đây là biến cần dự báo, giá trị cao là sự kiện thật. Khi chuẩn hoá, PM2.5 được lấy log nên các giá trị lớn được nén lại |
| Số nguyên | PM2.5 được ghi dạng số nguyên (4, 8, 55…) | Không cần xử lý |
| Không có độ ẩm | Bộ này không có độ ẩm tương đối | Dùng độ ẩm của Open-Meteo |
| Hướng gió dạng chữ | `wd` là chữ, không dùng trực tiếp được | Đổi chữ sang độ, rồi tách thành 2 thành phần gió `wind_u`, `wind_v` |

Nhìn chung bộ Bắc Kinh **sạch hơn HealthyAir rất nhiều**: thiếu ~2% so với ~25% ở TP.HCM.

## 6. Bắc Kinh khác TP.HCM ở đâu

| | Bắc Kinh | TP.HCM |
|---|---|---|
| Quy mô | 12 trạm × 4 năm | 6 trạm × ~16 tháng |
| PM2.5 trung bình / trung vị | 79,7 / 55 µg/m³ | 21,1 / 17,5 µg/m³ |
| % giờ > 35,4 µg/m³ (mức cần cảnh báo) | 62,7% | 11,3% |
| Giờ ô nhiễm nhất trong ngày | khoảng 22h (ban đêm) | khoảng 7h (giờ đi làm buổi sáng) |
| Mùa ô nhiễm | mùa đông (tháng 12, do sưởi ấm) | mùa khô (tháng 1) |
| Các trạm có lên xuống cùng nhau không | rất giống nhau (tương quan 0,89) | ít giống hơn (0,60), do nguồn ô nhiễm tại chỗ |

Vì hai thành phố chênh nhau khoảng 4 lần về mức nồng độ, PM2.5 được **lấy log rồi chuẩn hoá riêng theo từng trạm** trước khi đưa vào mô hình. Nhờ đó mô hình học **hình dạng lên xuống** của chuỗi chứ không học mức tuyệt đối.

## 7. Cách nhóm chia dữ liệu Bắc Kinh

| Tập | Thời gian | Dùng để |
|---|---|---|
| Train | 01/03/2013 → 29/02/2016 (3 năm) | Pretrain mô hình |
| Validation | 01/03/2016 → 28/02/2017 (1 năm) | Dừng huấn luyện sớm (early stopping) |
| Test | không có | Kết quả chỉ chấm trên TP.HCM |
