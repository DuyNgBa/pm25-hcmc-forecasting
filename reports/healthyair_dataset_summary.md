# Tóm tắt bộ dữ liệu HealthyAir TP.HCM

**Nguồn:** Rakholia, R., Le, Q., Vu, K.H.N., Ho, B.Q., Simon Carbajo, R. (2023). *Outdoor air quality data for spatiotemporal analysis and air quality modelling in Ho Chi Minh City, Vietnam: A part of HealthyAir Project.* Data in Brief, 46, 108774. https://doi.org/10.1016/j.dib.2022.108774

**Dữ liệu:** Mendeley Data, https://data.mendeley.com/datasets/pk6tzrjks8/1 (DOI 10.17632/pk6tzrjks8.1). Giấy phép CC BY 4.0.

**Tổng quan:** 6 trạm quan trắc, đo từ giữa 02/2021 đến giữa 06/2022, gộp theo giờ, 52.549 bản ghi. Biến gồm PM2.5, TSP, O3, CO, NO2, SO2, nhiệt độ và độ ẩm.

---

## 1. Vị trí

| Trạm | Vĩ độ | Kinh độ | Địa chỉ |
|---|---|---|---|
| 1 | 10,8699 | 106,7960 | ĐHQG TP.HCM, P. Linh Trung, TP. Thủ Đức |
| 2 | 10,7410 | 106,6171 | 20 Nguyễn Trọng Trí, P. An Lạc, Q. Bình Tân |
| 3 | 10,8162 | 106,6204 | KCN Tân Bình, P. Tây Thạnh, Q. Tân Phú |
| 4 | 10,8158 | 106,7174 | 49 Thanh Đa, P. 27, Q. Bình Thạnh |
| 5 | 10,7764 | 106,6878 | 268 Nguyễn Đình Chiểu, P. 6, Q. 3 |
| 6 | 10,7805 | 106,6595 | MM18 Trường Sơn, P. 14, Q. 10 |

> ⚠️ **Bảng 1 trong bài báo ghi ngược tên cột:** cột "longitude" chứa giá trị 10,87… (thực ra là vĩ độ), còn cột "latitude" chứa 106,79… (thực ra là kinh độ). Bảng trên đã sửa lại. Khi gọi Open-Meteo phải dùng `latitude=10.87`, `longitude=106.79`.

## 2. Loại trạm

| Loại | Trạm |
|---|---|
| Giao thông | 2, 5 |
| Dân cư | 4 |
| Công nghiệp | 3 |
| Giao thông + dân cư | 6 |
| Nền đô thị (công nghiệp + giao thông + dân cư) | 1 |

**Thiết bị:** máy phân tích chất lượng không khí ngoài trời **PM SCAN** của hãng Sensoronic Co., Ltd (Hàn Quốc). Mọi trạm đo cùng một bộ chỉ số.

## 3. Cách gộp từ phút sang giờ

Nguồn: bài báo (mục 3) và notebook [`data integration_codes.ipynb`](https://github.com/RajnishUCD/HealthyAir/blob/main/data%20integration_codes.ipynb) của nhóm tác giả.

**Quy trình trong code:**

1. Mỗi trạm đo **mỗi 60 giây**, gửi lên cloud server; mỗi trạm mỗi ngày lưu thành một file CSV.
2. Gộp tất cả file CSV. Thời điểm của mỗi bản ghi = **ngày lấy từ tên file** + giờ:phút:giây lấy từ cột `Date` gốc.
3. Giữ các thiết bị có `DeviceID` từ 1 đến 6.
4. Gộp theo giờ cho từng trạm: `resample('60T', on='date').agg(np.nanmean)`.
5. Quy đổi đơn vị các chất khí sang µg/m³ (xem mục 4).
6. Thay các giá trị **âm** bằng NaN.
7. (Không có trong notebook) Bỏ các giờ không có dữ liệu, cắt đầu chuỗi, đổi `DeviceID` thành `Station_No`, xuất ra CSV.

**Đã xác định:**

| Câu hỏi | Trả lời |
|---|---|
| Gộp bằng hàm gì | **Trung bình** (`np.nanmean`) |
| Mốc thời gian | **Đầu giờ**: dòng `01:00` là trung bình của khoảng 01:00:00–01:59:59 |
| Số phút tối thiểu mỗi giờ | **Không có ngưỡng**: chỉ cần 1 lần đo là có giá trị cho giờ đó |
| Múi giờ | **Không xử lý**: giữ nguyên giờ của thiết bị/server. Nhiều khả năng là giờ Việt Nam, cần kiểm tra bằng tương quan nhiệt độ với Open-Meteo |
| Định dạng thời gian trong file | `dd-mm-yyyy HH:00` |

> ⚠️ **Vấn đề trong quy trình gốc:**
> - **Giá trị âm được lọc *sau* khi tính trung bình:** các giá trị âm ở mức phút đã bị tính vào trung bình, kéo giá trị theo giờ xuống thấp. Chỉ những giờ có trung bình âm mới bị loại. Bài báo viết "đã loại bỏ giá trị âm" là không chính xác.
> - **Ngày lấy từ tên file, không phải từ thời điểm đo:** bản ghi gần nửa đêm bị lưu nhầm file có thể bị dời sai 1 ngày.
> - **Không loại dòng trùng** trong dữ liệu thô trước khi tính trung bình.
> - **Giờ thiếu dữ liệu bị xoá khỏi file** thay vì để NaN, ở một bước không có trong notebook. File công bố không có dòng nào thiếu PM2.5, và bắt đầu từ 23/02/2021 21:00 thay vì 20/02/2021 như dữ liệu thô. Đây là lý do bài báo nói "không thiếu timestamp".
> - **Một giờ có thể chỉ dựa trên 1 lần đo**, nên độ tin cậy giữa các giờ không đồng đều.
> - **Không thể chạy lại:** dữ liệu theo phút không được công bố; code dùng `DataFrame.append` (đã bị xoá từ pandas 2.0).

## 4. Đơn vị

| Chất | Đơn vị gốc | Hệ số quy đổi sang µg/m³ |
|---|---|---|
| PM2.5, TSP | µg/m³ | Không cần quy đổi |
| CO | ppm | × 1146 |
| NO2 | ppm | × 1882 |
| SO2 | ppm | × 2620 |
| O3 | ppb | × 1,963 |
| Nhiệt độ | °C | |
| Độ ẩm | % | |

Các hệ số này là hệ số chuẩn ở 25°C và 1 atm, nên phép quy đổi là đúng. Vì vậy SO2 cao bất thường (trung bình khoảng 225 µg/m³) **không phải do quy đổi sai**, mà do chính cảm biến đọc ra như vậy: output mẫu trong notebook có SO2 gốc 0,14 ppm, quy đổi thành 366,8 µg/m³, một mức rất cao so với không khí đô thị. Khả năng cao là cảm biến bị lệch, nên **không dùng SO2 làm đặc trưng chính**.

## 5. Hạn chế

### 5.1. Bài báo tự thừa nhận

- Có outlier ở một số thời điểm do "sự kiện bất thường".
- Một số chất ô nhiễm bị thiếu ở một số trạm, chủ yếu trong **giai đoạn giãn cách COVID-19**, do mất điện và các yếu tố khách quan.
- Thời gian chỉ khoảng 16 tháng.

### 5.2. Bài báo nói chưa đúng, hoặc có lỗi nằm ngay trong bảng của bài báo

| Bài báo nói / cho thấy | Thực tế |
|---|---|
| "Không thiếu timestamp nào ở mọi trạm" | Sai. Các giờ thiếu dữ liệu đã bị **xoá khỏi file** chứ không để trống. Trạm 1 chỉ có khoảng 68% số giờ, trạm 5 khoảng 64%; trạm 1 và 5 mất trắng tháng 8–9/2021 |
| "Đã loại bỏ các giá trị âm" | Chỉ đúng một phần: giá trị âm được lọc **sau** khi tính trung bình theo giờ, nên các giá trị âm ở mức phút vẫn bị tính vào |
| Bảng 2 (dữ liệu mẫu): nhiệt độ trạm 4 = **20,40**, độ ẩm = **12,50** ở cả 5 dòng | Đây là lỗi cảm biến đóng băng, kéo dài 8.123 giờ (05/2021–06/2022), nhưng bài báo dùng làm ví dụ mà không nhận ra |
| Bảng 3: TSP trạm 3 có trung vị và phân vị 75 đều bằng **0** | Cảm biến TSP trạm 3 hỏng (8.044 giờ bằng 0), dẫn đến 8.107 dòng có PM2.5 > TSP, vô lý về mặt vật lý |
| Bảng 3: NO2 trạm 5 có trung vị **272,9**, phân vị 75 là 276,2 | Cảm biến NO2 bị bão hoà, kẹt ở mức trần trong 911 giờ |
| Bảng 3: CO trạm 2 chỉ có 358 giá trị; O3, NO2, SO2 trạm 1 và 5 chỉ khoảng 2.200–2.500 giá trị | Trạm thiếu hẳn cảm biến trong phần lớn thời gian |
| Chỉ thay giá trị **âm** bằng NaN | Các giá trị **0** vẫn được giữ, phần lớn là lỗi cảm biến (PM2.5 = 0 ở 207 giờ, CO = 0 ở 430 giờ, NO2 = 0 ở 326 giờ) |
| Gọi thiết bị là "cảm biến chất lượng cao" | PM SCAN là máy đo dùng cảm biến, không phải thiết bị đo chuẩn (reference-grade), nên độ chính xác của các chất khí (đặc biệt SO2) cần dè dặt |

### 5.3. Các hạn chế khác

- **Không có gió, mưa, áp suất**: đây là các yếu tố quyết định sự phát tán và lắng đọng bụi mịn. Cần bổ sung từ nguồn ngoài (Open-Meteo).
- **Phiên bản dữ liệu không thống nhất:** phần mô tả dẫn link Mendeley **V1** (`pk6tzrjks8/1`), nhưng phần tài liệu tham khảo ghi **V2** (`pk6tzrjks8.2`). Cần kiểm tra file đang dùng là phiên bản nào và ghi rõ trong báo cáo.

---

## Hướng xử lý đề xuất

| Vấn đề | Xử lý |
|---|---|
| Toạ độ bị ghi ngược trong Bảng 1 | Dùng toạ độ đã sửa ở mục 1 |
| Nhiệt độ, độ ẩm trạm 4 đóng băng | Bỏ, thay bằng Open-Meteo cho tất cả các trạm |
| TSP trạm 3 bằng 0 | Chuyển thành NaN; không dùng TSP làm đặc trưng chính |
| NO2 trạm 5 bão hoà | Chuyển đoạn bị kẹt thành NaN |
| Giá trị 0 vô lý (PM2.5, CO, NO2) | Chuyển thành NaN |
| SO2 cao bất thường | Không dùng làm đặc trưng chính; nêu rõ trong phần hạn chế |
| Giờ thiếu bị xoá khỏi file | **Bắt buộc** tạo lại lưới giờ đầy đủ cho từng trạm (giờ thiếu để NaN) trước khi tạo cửa sổ trượt |
| Khoảng trống dài (trạm 1, 5 trong tháng 8–9/2021) | Không nội suy; thêm biến chỉ báo giãn cách |
| Múi giờ chưa được xác nhận | Kiểm tra khớp giờ bằng tương quan nhiệt độ với Open-Meteo (thử lệch từ −8 đến +8 giờ) |
| Mốc thời gian là đầu giờ | Khi ghép với Open-Meteo, lưu ý giá trị Open-Meteo tại `01:00` là giá trị tức thời lúc 01:00, còn HealthyAir là trung bình 01:00–01:59 |
