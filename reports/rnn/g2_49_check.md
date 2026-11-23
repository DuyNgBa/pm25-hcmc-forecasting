# G2-49. Kiểm tra kết quả LSTM / GRU / Seq2Seq

Sinh bởi `python -m src.models.rnn_check` lúc 2026-11-23T16:00:46.

- Bộ dữ liệu: `data_version=None`, `data_fp=legacy`, khí tượng `None`
- Lần chạy: 60 (cần 60 cho các nhóm rq1, rolling, rq2, seq2seq, rq3, curve, oracle), file dự đoán đọc được: 60
- Thiết bị: {'cpu': 60}
- Đối chiếu y_true với hcm_schema local: có
- Khớp bản dữ liệu nhóm đã khoá (`data/DATA_LOCK.json`): KHÔNG – dữ liệu cũ
- **Kết luận: ĐẠT** (0 lỗi, 1 cảnh báo)

## Độ phủ cấu hình

| group   | model   |   cần |   có |
|:--------|:--------|------:|-----:|
| curve   | GRU     |     6 |    6 |
| curve   | LSTM    |     6 |    6 |
| oracle  | GRU     |     1 |    1 |
| oracle  | LSTM    |     1 |    1 |
| rolling | GRU     |     4 |    4 |
| rolling | LSTM    |     4 |    4 |
| rq1     | GRU     |     3 |    3 |
| rq1     | LSTM    |     3 |    3 |
| rq2     | GRU     |     9 |    9 |
| rq2     | LSTM    |     9 |    9 |
| rq3     | GRU     |     4 |    4 |
| rq3     | LSTM    |     4 |    4 |
| seq2seq | Seq2Seq |     6 |    6 |

## Cảnh báo

- Đây là các lần chạy CŨ (trước khi có dấu vân tay dữ liệu: v0.9-dev, khí tượng best_match). Không đối chiếu được với dữ liệu hiện tại; chỉ là kết quả thử nghiệm.
