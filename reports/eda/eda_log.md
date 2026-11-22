# EDA PM2.5 HealthyAir

> Tóm tắt nhanh: PM2.5 ở TP.HCM **cao nhất vào buổi sáng (6–9h) và vào mùa khô (tháng 12–1)**, thấp nhất vào đầu giờ chiều và mùa mưa. Giờ cao điểm buổi sáng chủ yếu do giao thông: khi giãn cách, đỉnh này gần như biến mất. Mỗi trạm có đặc điểm riêng khá rõ, các trạm gần nhau thì giống nhau hơn.

Chỉ dùng giá trị đo thật (`is_imputed == 0`, `pm25` không NaN).

Số giờ dùng cho EDA: 52329 (6 trạm)

## G1-20. Phân phối PM2.5 theo trạm
| station_id   | loai_tram           |   so_gio |   tb |   trung_vi |   do_lech |   p95 |   lon_nhat |   pct_gio_vuot_35.4 |
|:-------------|:--------------------|---------:|-----:|-----------:|----------:|------:|-----------:|--------------------:|
| hcm_01       | Nền đô thị          |     7889 | 20.8 |       18.6 |      11.4 |  42.3 |      112   |                10.4 |
| hcm_02       | Giao thông          |     9357 | 19.2 |       16.1 |      10.6 |  40.7 |       96   |                 7.8 |
| hcm_03       | Công nghiệp         |     8418 | 23.5 |       18.4 |      17.7 |  57.3 |      207.8 |                17.1 |
| hcm_04       | Dân cư              |     9946 | 26.4 |       22.5 |      16.2 |  54.3 |      403.7 |                19.5 |
| hcm_05       | Giao thông          |     7224 | 15.5 |       13.3 |       8.1 |  31.7 |       77.4 |                 2.7 |
| hcm_06       | Giao thông + dân cư |     9495 | 20.1 |       16.8 |      12.5 |  42   |      120.3 |                 8.4 |

**Nhận xét G1-20** (hình: `figures/g1_20_histogram.png`, `figures/g1_20_boxplot.png`)
- **Trạm 4 ô nhiễm nhất**: trung bình 26,4 µg/m³, khoảng 1/5 số giờ vượt 35,4. **Trạm 5 sạch nhất**: trung bình 15,5, chỉ 2,7% số giờ vượt.
- Ở mọi trạm, trung bình cao hơn trung vị: phần lớn thời gian không khí ở mức vừa phải, nhưng thỉnh thoảng có những đợt bụi rất cao kéo trung bình lên.
- Trạm 5 nằm trên đường Nguyễn Đình Chiểu (khu giao thông) mà lại thấp nhất, khá lạ. Trạm này cũng là trạm nhiều lỗi cảm biến nhất, nên cần dè dặt với số liệu của nó.
- Ngưỡng 35,4 µg/m³ là ngưỡng của Mỹ (US EPA). Quy chuẩn Việt Nam (QCVN 05:2023) dùng ngưỡng 50 µg/m³ cho trung bình 24 giờ.

## G1-21. Chu kỳ ngày và chu kỳ tuần
Giờ cao nhất / thấp nhất theo trạm:
| station_id   |   gio_cao_nhat |   gia_tri_cao_nhat |   gio_thap_nhat |   gia_tri_thap_nhat |
|:-------------|---------------:|-------------------:|----------------:|--------------------:|
| hcm_01       |              6 |               28.9 |              15 |                16.7 |
| hcm_02       |              7 |               23.2 |              22 |                17   |
| hcm_03       |              6 |               28.6 |              16 |                19.5 |
| hcm_04       |              6 |               34   |              13 |                23.2 |
| hcm_05       |              9 |               20.8 |               0 |                12.7 |
| hcm_06       |              9 |               24.4 |               0 |                17.7 |

Ngày thường so với cuối tuần:
| station_id   |   ngay_thuong |   cuoi_tuan |   chenh_lech_pct |
|:-------------|--------------:|------------:|-----------------:|
| hcm_01       |          20.9 |        20.6 |             -1.4 |
| hcm_02       |          19.5 |        18.3 |             -6.2 |
| hcm_03       |          24.1 |        22.1 |             -8.3 |
| hcm_04       |          26.9 |        25.3 |             -5.9 |
| hcm_05       |          15.8 |        14.6 |             -7.6 |
| hcm_06       |          20.3 |        19.5 |             -3.9 |

**Nhận xét G1-21** (hình: `figures/g1_21_cycles.png`)
- **Mọi trạm đều cao nhất vào buổi sáng (6–9h)**, đúng giờ đi làm. Buổi sáng lớp không khí sát mặt đất còn mỏng nên bụi bị giữ lại, cộng thêm lượng xe đông.
- Phần lớn các trạm **thấp nhất vào đầu giờ chiều (13–16h)**, khi trời nắng làm không khí xáo trộn mạnh và bụi phát tán lên cao.
- Trạm 5 và 6 (Q.3, Q.10, khu trung tâm) có đỉnh muộn hơn (9h) và thấp nhất lúc nửa đêm.
- **Cuối tuần chỉ thấp hơn ngày thường một chút (1–8%)**, nên chu kỳ tuần không mạnh. Trạm 3 (khu công nghiệp) giảm nhiều nhất, có thể vì nhà máy làm việc ít hơn vào cuối tuần.

## G1-22. Chu kỳ năm: theo tháng, mùa khô và mùa mưa
PM2.5 trung bình theo tháng (gộp các năm):
|   month |   hcm_01 |   hcm_02 |   hcm_03 |   hcm_04 |   hcm_05 |   hcm_06 |
|--------:|---------:|---------:|---------:|---------:|---------:|---------:|
|       1 |     26.8 |     29.1 |     38.7 |     36.6 |     18.6 |     29.5 |
|       2 |     18.1 |     18.1 |     20.3 |     23.9 |     12.5 |     19   |
|       3 |     21.9 |     19.7 |     20.7 |     25.7 |     15.7 |     25.7 |
|       4 |     20.5 |     19.5 |     23.3 |     29.2 |     18.8 |     21.3 |
|       5 |     16.8 |     16.3 |     18.4 |     23.8 |     14   |     16.3 |
|       6 |     19.4 |     17.8 |     20.6 |     23.3 |     13.5 |     17.2 |
|       7 |     18   |     12.9 |     16.8 |     17.4 |     12.8 |     12.6 |
|       8 |    nan   |     13.2 |     13.5 |     18.3 |    nan   |     12.7 |
|       9 |    nan   |     12.5 |     12.4 |     17.5 |    nan   |     12   |
|      10 |     20.8 |     18.5 |     23.7 |     30.1 |      9.9 |     25.7 |
|      11 |     23.5 |     24.3 |     31.4 |     34.1 |     14.4 |     22.3 |
|      12 |     23.1 |     28.7 |     36.9 |     34.3 |     15.7 |     26.3 |

Mùa khô (tháng 12–4) so với mùa mưa (tháng 5–11):
| station_id   |   Mùa khô |   Mùa mưa |   kho_chia_mua |
|:-------------|----------:|----------:|---------------:|
| hcm_01       |      21.9 |      19.3 |           1.13 |
| hcm_02       |      22.7 |      16.6 |           1.37 |
| hcm_03       |      27.8 |      20   |           1.39 |
| hcm_04       |      29.5 |      23.8 |           1.24 |
| hcm_05       |      16.5 |      13.6 |           1.21 |
| hcm_06       |      24.2 |      16.9 |           1.43 |

Lưu ý: dữ liệu từ 02/2021 đến 06/2022, nên các tháng 3–6 có 2 năm, các tháng 7–11 chỉ có năm 2021 (trùng giai đoạn giãn cách).

**Nhận xét G1-22** (hình: `figures/g1_22_annual.png`, `figures/g1_22_timeseries.png`)
- **Mùa khô bụi nhiều hơn mùa mưa**, gấp khoảng 1,1–1,4 lần tuỳ trạm. Cao nhất vào **tháng 12 và tháng 1**.
- Thấp nhất vào **tháng 7–9**. Tuy nhiên đây vừa là mùa mưa, vừa là giai đoạn giãn cách năm 2021, nên không tách được bao nhiêu phần là do mưa và bao nhiêu phần là do giãn cách.
- Từ tháng 10, bụi tăng dần trở lại khi bước sang mùa khô.

## G1-23. Tương quan giữa các trạm và tự tương quan
Tương quan PM2.5 giữa các trạm (Pearson, theo từng cặp giờ cùng có dữ liệu):
| station_id   |   hcm_01 |   hcm_02 |   hcm_03 |   hcm_04 |   hcm_05 |   hcm_06 |
|:-------------|---------:|---------:|---------:|---------:|---------:|---------:|
| hcm_01       |     1    |     0.57 |     0.58 |     0.51 |     0.5  |     0.48 |
| hcm_02       |     0.57 |     1    |     0.76 |     0.6  |     0.68 |     0.71 |
| hcm_03       |     0.58 |     0.76 |     1    |     0.56 |     0.57 |     0.6  |
| hcm_04       |     0.51 |     0.6  |     0.56 |     1    |     0.51 |     0.49 |
| hcm_05       |     0.5  |     0.68 |     0.57 |     0.51 |     1    |     0.71 |
| hcm_06       |     0.48 |     0.71 |     0.6  |     0.49 |     0.71 |     1    |

- Trung bình các cặp: 0.59; thấp nhất: 0.48; cao nhất: 0.76

Tự tương quan (ACF) theo trạm:
|    |   hcm_01 |   hcm_02 |   hcm_03 |   hcm_04 |   hcm_05 |   hcm_06 |
|---:|---------:|---------:|---------:|---------:|---------:|---------:|
|  1 |     0.87 |     0.88 |     0.85 |     0.85 |     0.89 |     0.93 |
|  3 |     0.61 |     0.64 |     0.58 |     0.61 |     0.67 |     0.78 |
|  6 |     0.37 |     0.43 |     0.36 |     0.39 |     0.42 |     0.61 |
| 12 |     0.16 |     0.26 |     0.21 |     0.17 |     0.2  |     0.43 |
| 24 |     0.38 |     0.43 |     0.4  |     0.3  |     0.52 |     0.57 |
| 48 |     0.28 |     0.37 |     0.33 |     0.25 |     0.42 |     0.47 |

**Nhận xét G1-23** (hình: `figures/g1_23_correlation_acf.png`)
- **Các trạm chỉ giống nhau ở mức vừa phải** (tương quan trung bình 0,59). Ô nhiễm ở TP.HCM chủ yếu do nguồn tại chỗ (xe cộ, công trình, đốt rác…), chứ không phải cả thành phố cùng lên cùng xuống.
- **Trạm càng gần nhau càng giống nhau**: trạm 2 và 3 (Bình Tân và Tân Phú) giống nhau nhất (0,76), tiếp theo là trạm 5 và 6 (khu trung tâm, 0,71). Trạm 1 (Thủ Đức, xa nhất) ít giống các trạm khác nhất.
  → Dùng dữ liệu của trạm lân cận có thể giúp dự báo, nhất là với các cặp trạm gần nhau.
- **Tự tương quan (ACF):**
  - Giá trị giờ này rất giống giờ trước (0,85–0,93), nên dự báo 1 giờ tới khá dễ.
  - Sau 12 giờ thì gần như không còn giống (0,16–0,43).
  - Nhưng sau **24 giờ lại giống hơn** (0,30–0,57): bụi lặp lại theo nhịp mỗi ngày.
  → Để dự báo ngày mai, nên dùng **giá trị cùng giờ hôm nay** và **giờ trong ngày** làm đặc trưng.

## G1-24. So sánh theo loại khu vực
| type                |   so_tram |   so_gio |   tb |   trung_vi |   p95 |   pct_vuot |
|:--------------------|----------:|---------:|-----:|-----------:|------:|-----------:|
| Công nghiệp         |         1 |     8418 | 23.5 |       18.4 |  57.3 |       17.1 |
| Dân cư              |         1 |     9946 | 26.4 |       22.5 |  54.3 |       19.5 |
| Giao thông          |         2 |    16581 | 17.6 |       14.9 |  36.6 |        5.6 |
| Giao thông + dân cư |         1 |     9495 | 20.1 |       16.8 |  42   |        8.4 |
| Nền đô thị          |         1 |     7889 | 20.8 |       18.6 |  42.3 |       10.4 |

Biên độ chu kỳ ngày (cao nhất − thấp nhất trong ngày) theo loại khu vực:
| type                |   bien_do |
|:--------------------|----------:|
| Công nghiệp         |       9   |
| Dân cư              |      10.8 |
| Giao thông          |       6.6 |
| Giao thông + dân cư |       6.7 |
| Nền đô thị          |      12.1 |

Lưu ý: mỗi loại chỉ có 1–2 trạm, nên khác biệt giữa các loại cũng chính là khác biệt giữa từng trạm cụ thể.

**Nhận xét G1-24** (hình: `figures/g1_24_station_type.png`)
- Kết quả **ngược với dự đoán**: khu dân cư (trạm 4) và khu công nghiệp (trạm 3) bụi nhiều nhất, còn khu giao thông lại ít nhất.
- Mỗi loại khu vực chỉ có 1–2 trạm, nên khác biệt này phản ánh **vị trí cụ thể của từng trạm** (gần công trình, chỗ đốt rác…) hơn là đặc điểm chung của cả loại khu vực. Ví dụ trạm 4 từng có một đợt bụi rất cao kéo dài khoảng 8 giờ vào ngày 14/10/2021.
- Vì vậy **không nên kết luận** "khu dân cư ô nhiễm hơn khu giao thông" chỉ dựa trên dữ liệu này.

## G1-25. Ảnh hưởng của giai đoạn giãn cách
PM2.5 trung bình trước / trong / sau giãn cách (cùng mùa mưa để hạn chế ảnh hưởng của mùa):
| station_id   |   tb_gian_cach (09/07–30/09/2021) |   tb_sau (01/10–31/10/2021) |   tb_truoc (09/06–08/07/2021) |   so_gio_gian_cach (09/07–30/09/2021) |   so_gio_sau (01/10–31/10/2021) |   so_gio_truoc (09/06–08/07/2021) |
|:-------------|----------------------------------:|----------------------------:|------------------------------:|--------------------------------------:|--------------------------------:|----------------------------------:|
| hcm_01       |                              19.6 |                        20.8 |                          19   |                                   102 |                             561 |                               228 |
| hcm_02       |                              12.8 |                        18.5 |                          17.2 |                                  1714 |                             529 |                               524 |
| hcm_03       |                              14.1 |                        23.7 |                          21.7 |                                  1196 |                             540 |                               529 |
| hcm_04       |                              17.8 |                        30.1 |                          19.4 |                                  1523 |                             591 |                               554 |
| hcm_05       |                              13.1 |                         9.9 |                          14.6 |                                   381 |                              77 |                               470 |
| hcm_06       |                              12.4 |                        25.7 |                          16.1 |                                  1677 |                             554 |                               542 |

Trạm có dưới 200 giờ dữ liệu trong một giai đoạn thì kết quả giai đoạn đó không đáng tin.

Chu kỳ ngày (gộp các trạm): giãn cách so với các tháng mùa mưa khác:
- Giãn cách: cao nhất 16.1 lúc 7h, thấp nhất 12.9, biên độ 3.1
- Mùa mưa không giãn cách: cao nhất 25.6 lúc 6h, thấp nhất 16.2, biên độ 9.5

**Nhận xét G1-25** (hình: `figures/g1_25_lockdown.png`)
- Chỉ nên xét các trạm có đủ dữ liệu (trạm 2, 3, 4, 6). Trạm 1 (chỉ 102 giờ trong giãn cách) và trạm 5 (chỉ 77 giờ sau giãn cách) có quá ít số liệu.
- **Giãn cách làm bụi giảm rõ rệt** so với tháng trước đó:

| Trạm | Trước | Giãn cách | Thay đổi |
|---|---:|---:|---:|
| hcm_02 | 17,2 | 12,8 | −26% |
| hcm_03 | 21,7 | 14,1 | −35% |
| hcm_04 | 19,4 | 17,8 | −8% |
| hcm_06 | 16,1 | 12,4 | −23% |

- **Đỉnh bụi buổi sáng gần như biến mất** khi giãn cách: chênh lệch giữa giờ cao nhất và thấp nhất trong ngày giảm từ 9,5 xuống còn 3,1 µg/m³. Đây là bằng chứng rõ nhất cho thấy **đỉnh buổi sáng chủ yếu do xe cộ**.
- Sau giãn cách (tháng 10), bụi tăng lên **cao hơn cả trước giãn cách**, vì xe cộ hoạt động trở lại và thời tiết bắt đầu chuyển sang mùa khô.

## Gợi ý cho bước xây dựng mô hình
1. Thêm đặc trưng **giờ trong ngày** và **PM2.5 cùng giờ hôm trước** (lag 24 giờ).
2. Giữ biến **`is_lockdown`**, vì giai đoạn giãn cách có mức bụi và nhịp trong ngày khác hẳn bình thường.
3. Thử dùng **PM2.5 của trạm lân cận**, nhất là các cặp 2–3 và 5–6.
4. **Không nên để tập test rơi vào giai đoạn giãn cách**, vì dữ liệu lúc đó không giống thời gian bình thường.
5. Kết quả của trạm 5 cần dè dặt, vì trạm này vừa thiếu nhiều dữ liệu vừa có nhiều lỗi cảm biến.