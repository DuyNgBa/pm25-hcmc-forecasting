# Hướng dẫn chạy GĐ2 (LSTM/GRU) trên Kaggle – TV2

Notebook: `notebooks/tridao/rnn_colab_kaggle.ipynb`. Chỉ chạy **sau khi** nhóm đã chốt `data_version: "v1.0"` và nhánh của bạn đã có code TV1 + TV3 + GĐ2 mới nhất.

## 1. Chuẩn bị (một lần)

1. **Token GitHub chỉ đọc:** GitHub → *Settings → Developer settings → Fine-grained tokens → Generate*. Chọn *Only select repositories* → `DuyNgBa/pm25-hcmc-forecasting`; *Repository permissions → Contents: Read-only*; hạn 7–30 ngày. Nếu không chọn được repo (bạn không phải chủ repo) thì nhờ Duy duyệt token, hoặc dùng cách Dataset ở bước 3.
2. **Kaggle:** tạo notebook mới → *File → Import notebook* → chọn file `.ipynb` trên.
   - *Add-ons → Secrets* → thêm `GITHUB_TOKEN` (dán token) → bật cho notebook này.
   - *Settings → Accelerator: GPU T4 x2*; *Internet: On* (cần xác minh số điện thoại).
3. (Không dùng token) Nén thư mục repo đúng nhánh thành zip → *Datasets → New dataset* (Private) → trong notebook *Add Input* dataset đó.

## 2. Chạy

1. Ô 1, sửa nếu cần: `BRANCH` (mặc định `Tridao-gd2`), `REQUIRE_DATA_VERSION = 'v1.0'`.
2. *Save Version → Save & Run All (Commit)* để chạy nền (tối đa 12 giờ, ước tính 1–3 giờ trên T4). Notebook **dừng ngay** nếu:
   - không có GPU;
   - thiếu token hoặc sai nhánh (thiếu `src/models/provenance.py`, `rnn_check.py`, `clean_healthyair.py`…);
   - `configs/openmeteo.yaml` chưa có `model: era5_seamless`;
   - `data_version` khác `v1.0`;
   - bất kỳ lệnh dựng dữ liệu, test, HPO hay thí nghiệm nào lỗi;
   - `best` trong `configs/rnn.yaml` không được tối ưu trên đúng dữ liệu (dấu vân tay `data_fp`).
3. Bị ngắt giữa chừng: tải `rnn_results.zip` + `rnn_checkpoints.zip` của phiên trước → tạo Dataset → *Add Input* → chạy lại. Chỉ phần **cùng `data_fp`** được dùng lại (lần chạy, checkpoint pretrain Bắc Kinh, lần thử Optuna). Dữ liệu khác thì tự chạy lại.

## 3. Tải về và dùng kết quả (không train lại)

Tab *Output*:
- `rnn_results.zip`: `results/rnn.csv`, `results/rnn_runs.csv`, `results/baselines.csv`, `results/preds/`, `results/hpo/`, `configs/rnn.yaml`, `reports/rnn/` (gồm `summary.md`, `g2_49_check.md`), `logs/`.
- `rnn_checkpoints.zip`: trọng số (cần cho demo web G3-28).
- `rnn_preds_h24.zip` (nặng): dự đoán đủ 24 tầm, chỉ cần cho G3-10.

Trên máy (từ gốc repo, đã dựng dữ liệu `v1.0` local):

```powershell
# Cất kết quả thử nghiệm cũ trước, vì giải nén sẽ ghi đè results/rnn.csv, rnn_runs.csv
New-Item -ItemType Directory -Force results/archive/rnn_v0.9-dev_best_match | Out-Null
Copy-Item results/rnn.csv, results/rnn_runs.csv, reports/rnn/summary.md results/archive/rnn_v0.9-dev_best_match/
Expand-Archive rnn_results.zip -DestinationPath . -Force
$env:PYTHONUTF8 = "1"
python -m src.models.rnn_check --require-gpu
python -m src.models.rnn_report
```

- `rnn_check` phải in **ĐẠT**. Nếu dữ liệu local khác `data_fp` của Kaggle, nó bỏ qua bước đối chiếu với `hcm_schema` và báo cảnh báo. Thêm `--truth-anyway` để vẫn đối chiếu PM2.5.
- Đầu `reports/rnn/summary.md` phải ghi **"KẾT QUẢ CHÍNH THỨC"** thì mới chép số sang `g2_50_muc_bao_cao_lstm_gru.md` mục 4. Nhãn này đòi cả ba điều kiện: `data/DATA_LOCK.json` của nhóm có mặt và khớp từng dấu vân tay (`raw_fp`, `hcm_fp`, `beijing_fp`, `v1.0`); `rnn_check` đạt. Nếu bản khoá được commit sau khi chạy Kaggle, chỉ cần pull rồi chạy lại `rnn_check` + `rnn_report`, không phải train lại. Kiểm tra nhanh dữ liệu local: `python -m src.models.provenance`.
- Sau khi HPO chính thức ghi `best` mới, xoá 2 dòng chú thích "best hiện tại: … THỬ NGHIỆM" phía trên mục `best:` trong `configs/rnn.yaml`.
