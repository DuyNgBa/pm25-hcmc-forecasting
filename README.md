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

## Môi trường và tái lập kết quả (G0-02)

- **Python 3.10** (máy local dùng 3.10.8). Trên Kaggle/Colab dùng bản Python có sẵn, chỉ cần `pip install -r requirements.txt`.
- **`seed = 42`** cho mọi thí nghiệm (khai báo trong `configs/data.yaml`, gọi qua `src.utils.seed_everything`). Mô hình học sâu chạy thêm seed 43 và 44 để báo cáo trung bình ± độ lệch chuẩn.
- Mọi mốc chia train/val/test và đường dẫn dữ liệu nằm trong một file duy nhất: `configs/data.yaml`.

## Quy ước dữ liệu (G0-03)

| Quy ước | Nội dung |
|---|---|
| Định dạng file | Parquet (`data/interim/`, `data/processed/`); dữ liệu gốc trong `data/raw/` giữ nguyên |
| Khoá | (`city`, `station_id`, `time`), không trùng khoá |
| `city` | `hcm` hoặc `beijing` |
| `station_id` | TP.HCM: `hcm_01` … `hcm_06`. Bắc Kinh: tên trạm gốc của bộ UCI (`Aotizhongxin`, `Changping`, …) |
| `time` | Giờ địa phương, không kèm múi giờ, tròn giờ. TP.HCM UTC+7, Bắc Kinh UTC+8 |
| Giá trị thiếu | `NaN`; giá trị lỗi cảm biến chuyển thành `NaN` |
| Nội suy | Chỉ nội suy khoảng trống ≤ 3 giờ, đánh dấu bằng cột `is_imputed` |
| Nhãn đánh giá | Chỉ dùng giá trị đo thật: `is_imputed == 0` và `pm25` không NaN |
| PM2.5 | µg/m³; không cắt outlier (cả hai thành phố) |
| Khí tượng | Open-Meteo, `models=era5_seamless`, cùng nguồn cho cả hai thành phố; gió m/s, mưa mm, nhiệt độ °C, áp suất hPa |
| Tên cột | snake_case, theo lược đồ chung trong `src/data/schema.py` |
| Môi trường | Python 3.10, `seed = 42`, thư viện trong `requirements.txt` |

Muốn thay đổi quy ước: tạo Pull Request riêng, cả 3 người đồng ý, ghi vào bảng lịch sử thay đổi bên dưới.

### Lịch sử thay đổi quy ước

| Ngày | Thay đổi | Người đề xuất |
|---|---|---|
| 23/11/2026 | Ghi lại quy ước đang dùng trong code (mục này) | Bo bo |
