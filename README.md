# Dự báo PM2.5 tại TP.HCM

Đồ án môn học: dự báo đa bước nồng độ bụi mịn PM2.5 theo giờ tại 6 trạm quan trắc ở TP. Hồ Chí Minh. Dữ liệu chính là bộ quan trắc HealthyAir, kết hợp dữ liệu khí tượng Open-Meteo và học chuyển giao từ dữ liệu Bắc Kinh.

## Mục tiêu

- Dự báo PM2.5 ở 3 tầm: **t+1h, t+6h, t+24h**
- So sánh 3 nhóm phương pháp: **Gradient Boosting, RNN (LSTM/GRU), Transformer (PatchTST)**
- Mọi mô hình đều phải so sánh với baseline persistence (giữ nguyên giá trị hiện tại)

## Câu hỏi nghiên cứu

| | Câu hỏi |
|---|---|
| **RQ1** | Nhóm mô hình nào dự báo tốt nhất ở từng tầm dự báo? |
| **RQ2** | Dữ liệu khí tượng Open-Meteo và PM2.5 các trạm lân cận đóng góp bao nhiêu? |
| **RQ3** | Pretrain trên dữ liệu Bắc Kinh có cải thiện dự báo cho TP.HCM không? |

## Dữ liệu

| Bộ dữ liệu | Vai trò | Nguồn |
|---|---|---|
| HealthyAir TP.HCM (6 trạm, 02/2021–06/2022) | Dữ liệu chính | [Mendeley Data](https://data.mendeley.com/datasets/pk6tzrjks8/1) |
| Open-Meteo Historical Weather (ERA5) | Bổ sung gió, mưa, áp suất… | [Open-Meteo API](https://open-meteo.com/en/docs/historical-weather-api) |
| Beijing Multi-Site Air Quality (12 trạm, 2013–2017) | Học chuyển giao | [UCI ML Repository](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) |

Lưu ý: dữ liệu HealthyAir có nhiều lỗi cảm biến (nhiệt độ và độ ẩm trạm 4 bị đóng băng, TSP trạm 3 bằng 0, NO2 trạm 5 bị bão hoà…). Xem chi tiết trong `reports/`.

## Cấu trúc thư mục

```
├── data/
│   ├── raw/          # Dữ liệu gốc, KHÔNG chỉnh sửa
│   ├── interim/      # Dữ liệu trung gian (không commit)
│   └── processed/    # Dữ liệu đã làm sạch, sẵn sàng train (không commit)
├── src/              # Code dùng chung: xử lý dữ liệu, mô hình, đánh giá
├── notebooks/        # Notebook thử nghiệm, mỗi người một thư mục riêng
├── results/          # Kết quả thí nghiệm (.csv), mỗi phương pháp một file
└── reports/          # Proposal, báo cáo, hình ảnh
```

## Cài đặt

```bash
git clone https://github.com/<tên-tài-khoản>/pm25-hcmc-forecasting.git
cd pm25-hcmc-forecasting
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## Phân công

| Thành viên | Chuẩn bị dữ liệu | Phương pháp |
|---|---|---|
| [Thành viên 1] | Làm sạch HealthyAir | Gradient Boosting (LightGBM/XGBoost) + SHAP |
| [Thành viên 2] | Tải và ghép Open-Meteo | RNN (LSTM/GRU) |
| [Thành viên 3] | Dữ liệu Bắc Kinh, pipeline và hàm đánh giá chung | Transformer (PatchTST) |

## Quy trình làm việc

- **Không push thẳng vào `main`.** Mỗi task làm trên một nhánh riêng, sau đó tạo Pull Request và cần 1 người review.
- Đặt tên nhánh: `data/<tên-task>`, `model/<tên-mô-hình>`, `report/<nội-dung>`
- Trước khi tạo nhánh mới: chạy `git checkout main` rồi `git pull`
- Notebook để trong thư mục riêng của mỗi người. Code dùng chung đưa vào `src/`.
- Sửa `src/evaluate.py` hoặc phần xử lý dữ liệu thì phải báo cả nhóm.
- Kết quả ghi vào `results/<tên-phương-pháp>.csv`, dùng chung một mẫu cột.

## Quy ước so sánh công bằng

Cùng phiên bản dữ liệu, cùng cách chia train/val/test theo thời gian, cùng hàm đánh giá (MAE, RMSE, R²), cùng ngân sách tối ưu siêu tham số.