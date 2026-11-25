# [Khung] Giải thích kết quả chuyển giao bằng các khác biệt ở mục 4.1 (G3-05)

*Mục này viết SAU khi có bảng RQ3 (`reports/rq3_table.md`, `figures/rq3_data_curve_*.png`). Dưới đây là các giả thuyết đặt ra từ trước, cùng cách kiểm tra từng giả thuyết. Khi có số liệu, giữ những giả thuyết được dữ liệu ủng hộ và nói rõ những giả thuyết bị bác bỏ. Không viết kết luận trước khi có số.*

| # | Khác biệt (Bảng 4.1) | Dự đoán nếu khác biệt này quan trọng | Cách kiểm tra bằng dữ liệu đã có |
|---|---|---|---|
| H1 | Chu kỳ ngày ngược pha (đỉnh 22h ở Bắc Kinh, 7h ở TP.HCM) | S1 (zero-shot) sai nhiều nhất vào giờ cao điểm sáng 6–9h. S2_full tốt hơn S2_head, vì phần nắm quan hệ giờ nằm trong backbone | Lấy MAE theo giờ trong ngày của t+h từ file dự đoán (`results/preds`), so S1 với S0. So S2_full với S2_head |
| H2 | Tương quan giữa trạm 0,89 so với 0,59 (vùng so với cục bộ) | Mô hình pretrain dựa vào "trạng thái chung của thành phố". Lợi ích chuyển giao lớn hơn ở trạm nền đô thị (S1) so với trạm giao thông | Tách MAE theo `area_type` cho S0 và S2 |
| H3 | Chuyển giao chủ yếu là "hình dạng động học" sau chuẩn hoá log-z, chẳng hạn PM2.5 giảm khi mưa và gió mạnh | Lợi ích của S2/S3 so với S0 tăng khi `train_frac` giảm (100 → 25%) | Đường cong dữ liệu: khoảng cách giữa S0 và S2/S3 ở 25% so với ở 100% |
| H4 | ACF lag 1h: 0,97 so với 0,88 (TP.HCM nhiễu hơn) | Ở t+1, pretrain gần như không giúp gì, vì persistence đã mạnh và TP.HCM khó dự báo ngắn hạn. Nếu có lợi thì nằm ở t+6 và t+24 | So mức cải thiện tương đối theo tầm |
| H5 | Mức ô nhiễm: 62,7% giờ > 35,4 ở Bắc Kinh so với 11,3% ở TP.HCM | Mô hình học trên Bắc Kinh dự báo đỉnh "mạnh tay" hơn, nên MAE_high có thể tốt hơn dù MAE tổng thể kém hơn | So MAE_high với MAE của S1/S3 so với S0 |
| H6 | Khí tượng: cùng nguồn Open-Meteo, nhưng khí hậu khác nhau (4 mùa so với nhiệt đới) | Nếu S1 tệ hơn cả persistence thì mô hình đã học quan hệ khí tượng–PM2.5 riêng của Bắc Kinh, chẳng hạn nghịch nhiệt mùa đông | Tham chiếu SHAP của TV1: so vai trò của gió và mưa giữa mô hình train trên Bắc Kinh và mô hình train trên TP.HCM (G2-24) |

**Nếu chuyển giao không hiệu quả** (S2 và S3 không thắng S0 một cách có ý nghĩa qua 3 seed và các fold), đó vẫn là một kết luận có giá trị, đúng như mục 10 của đề xuất. Khi đó, phần thảo luận dựa vào H1, H2 và H6 để giải thích.
