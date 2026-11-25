# Hướng dẫn pipeline chung (TV3 — Bo bo)

Tài liệu này bổ sung cho README.md của repo: quy ước dữ liệu, chia dữ liệu, chuẩn hoá, cửa sổ, đánh giá và cách chạy lại toàn bộ.

> **Trạng thái dữ liệu: `v0.9-dev`.** Hiện pipeline đang chạy trên bản làm sạch **tạm**
> `healthyair_clean_DEV.parquet` (do `scripts/dev_hcm_minimal_clean.py` tạo) và **khí tượng gốc
> Bắc Kinh** (vì chưa có Open-Meteo). Mọi số liệu trong `results/` lúc này chỉ để kiểm tra code.
> Khi TV1 xuất `healthyair_clean.parquet` và TV2 xuất `meteo_*.parquet`, chạy lại theo mục
> "Chạy lại end-to-end", rồi tag `data-v1.0`.

---

## 1. Môi trường (G0-02)

- Nơi chạy: **Kaggle Notebook** (GPU T4/P100 cho PatchTST và LSTM; CPU đủ cho phần dữ liệu và cây).
- Python ≥ 3.10. Gói: xem `requirements.txt` (gộp với file đã chốt nếu khác phiên bản).
- **Seed chung: 42** (`configs/data.yaml → seed`). Mô hình DL chạy 3 seed: 42, 43, 44.
- Mọi lệnh chạy **từ gốc repo**: `python -m src.<module>`. Trên Kaggle: `%cd /kaggle/working/pm25-hcmc` trước.

```bash
pip install -r requirements.txt
pytest -q                     # 22 test (test cần dữ liệu tự skip khi chưa có dữ liệu)
```

## 2. Quy ước dữ liệu (G0-03, cả nhóm thống nhất)

| Quy ước | Nội dung |
|---|---|
| Tên cột | `snake_case` (vd. `pm25`, `wind_u`, `is_imputed`) |
| Thời gian | cột `time` = **giờ địa phương, không kèm múi giờ** (TP.HCM UTC+7, Bắc Kinh UTC+8). Chu kỳ sinh hoạt gắn với giờ địa phương |
| Khoá | `(city, station_id, time)`, `city ∈ {"hcm", "beijing"}`; `station_id` của TP.HCM = `"hcm_01"`..`"hcm_06"` (quy ước TV1), của Bắc Kinh = tên trạm gốc (`"Aotizhongxin"`...) |
| Lưới giờ | Mọi bảng đều có **lưới giờ đầy đủ**; giờ không có dữ liệu thì để NaN, không xoá dòng |
| Định dạng | `parquet`. `data/raw` không bao giờ bị sửa; `data/interim` là bản đã làm sạch theo từng nguồn; `data/processed` là bản theo lược đồ chung |
| Nhãn đánh giá | Chỉ dùng **giá trị đo thật**: `is_imputed = 0` và không NaN |
| Fit | Mọi bước có "fit" (scaler, ngưỡng outlier, trung bình theo giờ) **chỉ fit trên train** của split hoặc fold đang dùng |

**File của TV1** (`data/interim/healthyair_clean.parquet`, tạo bằng `python src/data/clean_healthyair.py`, mô tả ở `data/data_dictionary.md`): dùng các cột `time`, `station_id` (`hcm_01`..`hcm_06`), `pm25`, `is_imputed`, `temp_sensor`, `rh_sensor`, `is_lockdown`.

**Giao diện file TV2 → TV3** (`data/interim/meteo_{hcm,beijing}.parquet`): các cột `station_id`, `time`, cùng các biến **giữ đúng tên của Open-Meteo API**: `temperature_2m, relative_humidity_2m, dew_point_2m, precipitation, surface_pressure, wind_speed_10m (m/s), wind_direction_10m` (thêm cột khác cũng được).

### Lược đồ chung (G1-47) — `data/processed/{hcm,beijing}_schema.parquet`, hai file có cùng cột

`city, station_id, time, pm25, is_imputed, temp, rh, dew, wind_speed, wind_u, wind_v, precip, precip_6h, precip_24h, pressure, hour_sin, hour_cos, dow, dow_sin, dow_cos, temp_ha, rh_ha, is_lockdown, meteo_source`

- Khí tượng của **cả hai** thành phố lấy từ Open-Meteo (`meteo_source = "open_meteo"`).
- `wind_u = −v·sin θ`, `wind_v = −v·cos θ`, với θ là hướng gió **thổi từ** (quy ước khí tượng).
- `temp_ha, rh_ha` là khí tượng gốc của HealthyAir, chỉ dùng cho ablation (b); ở Bắc Kinh là NaN.
- Không có `month` và các chất ô nhiễm khác (đúng mục 4.2 của đề xuất).

## 3. Chia dữ liệu và chuẩn hoá (G1-48 → G1-53)

Nguồn sự thật duy nhất là `configs/data.yaml`. Không hard-code ngày trong notebook.

| | train | val | test |
|---|---|---|---|
| TP.HCM | 23/02/2021 – 31/12/2021 | 01/2022 – 02/2022 | 03/2022 – 21/06/2022 |
| Bắc Kinh (chỉ để pretrain) | 01/03/2013 – 29/02/2016 | 01/03/2016 – 28/02/2017 | — (không bao giờ dùng để đánh giá) |

**Rolling-origin** (expanding window, `reports/figures/rolling_origin_folds.png`): mỗi fold gồm val 1 tháng và test 2 tháng.

| Fold | train đến | val | test |
|---|---|---|---|
| 1 | 30/09/2021 | 10/2021 | 11–12/2021 |
| 2 | 30/11/2021 | 12/2021 | 01–02/2022 |
| 3 | 31/01/2022 | 02/2022 | 03–04/2022 |
| 4 | 31/03/2022 | 04/2022 | 05–21/06/2022 |

**Gán cửa sổ vào split:** một cửa sổ thuộc split X nếu **mọi** thời điểm mục tiêu t+1..t+H nằm trong X. Riêng train còn yêu cầu input cũng nằm trong train.

**Chuẩn hoá:**
- PM2.5: `z = (log(1+x) − μ_trạm) / σ_trạm`, với μ, σ tính trên train và chỉ trên giá trị đo thật.
- Khí tượng: tham số theo thành phố, fit trên train. Nhiệt độ, độ ẩm, điểm sương, áp suất, tốc độ gió dùng z-score. `wind_u/v` chỉ chia σ, không trừ mean (để giữ hướng gió). Mưa dùng `log1p` rồi chia σ (0 mm vẫn là 0).
- Tham số của split chính lưu ở `data/processed/scaler_params.json`. Hàm đảo ngược: `src.data.scaling.inverse_pm25`.

**Lấy mẫu con train** (`train_frac` = 1 / 0,5 / 0,25): lấy **đoạn liên tục cuối** của train (kịch bản "trạm mới lắp", sát với val).

## 4. Dùng module chung (G1-63) — xem thêm `notebooks/bobo/01_pipeline_usage.ipynb`

Hàm đánh giá có thể import gọn từ `src/evaluate.py` (đúng quy ước README nhóm).

```python
from src.data.pipeline import prepare
from src.data.windows import feature_columns
from src.eval.metrics import wide_to_long, evaluate_grouped, add_area_type
from src.eval.results import results_rows, append_results, save_predictions, make_run_id
from src.utils import results_file
from src.data.scaling import inverse_pm25

P = prepare("hcm", fold=None, train_frac=1.0, neighbors=False)   # fold=1..4 cho rolling-origin
feats = feature_columns("c")                    # a | b | c | d (ablation RQ2)
ws_tr = P.windows("train", feats, L=48)         # WindowSet: dùng như Dataset PyTorch
ws_te = P.windows("test",  feats, L=48)
X, Y, M = ws_tr.materialize()                   # (N, L, F), (N, 24) z, (N, 24) mask — cho sklearn/cây

# ... train mô hình -> z_pred (N, 24)
y_pred = inverse_pm25(z_pred, ws_te.meta()["station_id"], P.params, "hcm")   # µg/m³
preds  = wide_to_long(ws_te.meta(), ws_te.y_true_ug(), y_pred, horizons=(1, 6, 24), model="LightGBM", config=run_id)
evaluate_grouped(add_area_type(preds, P.cfg), ["horizon", "area_type"])
append_results(results_rows(preds, member="Duy Ngô", model="LightGBM", strategy="S0", ablation="c",
                            train_frac=1.0, L=48, seed=42, eval_set="test", run_id=run_id), results_file(P.cfg, "lightgbm"))   # -> results/lightgbm.csv
save_predictions(preds, "results/preds", run_id, "test")
```

**Mô hình cây (TV1)** cần đặc trưng bảng (lag, rolling): có thể lấy `X[:, -k, :]` từ `materialize()`, hoặc tự tạo bằng `src/features/tabular.py` trên `P.df`. Dù làm cách nào cũng **phải dùng chung** `ws_te.meta()` và `ws_te.y_true_ug()` để tập giờ đánh giá giống hệt nhau.

## 5. File kết quả `results/<tên-phương-pháp>.csv` (G0-04)

Theo quy ước README nhóm: **mỗi phương pháp một file** (`baselines.csv`, `lightgbm.csv`, `lstm.csv`, `patchtst.csv`...), **cùng mẫu cột** `results/results_template.csv`. `python -m src.aggregate_results` gộp tất cả lại. Mỗi dòng ứng với một bộ (mô hình, cấu hình, tầm, trạm hoặc ALL, tập đánh giá, seed).

| Cột | Giá trị |
|---|---|
| `member` | `Duy Ngô` / `Minh Trí` / `Bo bo` |
| `model` | `persistence`, `hourly_mean`, `Ridge`, `RandomForest`, `LightGBM`, `XGBoost`, `LSTM`, `GRU`, `Seq2Seq`, `PatchTST` |
| `strategy` | `S0`, `S1`, `S2_full`, `S2_head`, `S3` (với cây, S2 = `init_model`, ghi `S2`); `-` cho baseline Bắc Kinh |
| `ablation` | `a`, `b`, `c`, `d`, `c_oracle` (khí tượng thực tại t+h); `-` cho baseline |
| `train_frac` | 1.0 / 0.5 / 0.25 |
| `horizon` | 1 / 6 / 24 |
| `station` | `ALL` hoặc `S1`..`S6` |
| `L` | 24 / 48 / 72 (baseline ghi 0) |
| `seed` | 42 / 43 / 44 (baseline ghi 0) |
| `MAE, RMSE, R2, MAE_high, AQI_acc, AQI_macroF1` | tính bằng `src/eval/metrics.py`, đơn vị µg/m³ |
| `notes` | tự do (số tham số, thời gian train, phiên bản dữ liệu) |
| **`eval_set`** *(bổ sung)* | `test` / `fold1`..`fold4` — để tách kết quả rolling-origin khỏi test chính |
| **`n`** *(bổ sung)* | số giờ đo thật dùng để tính độ đo; giữa các mô hình cùng `eval_set`, tầm và L thì n phải bằng nhau |
| **`run_id`** *(bổ sung)* | khoá nối sang file dự đoán |

> Ba cột bổ sung là đề xuất của TV3 (không có `eval_set` thì không phân biệt được fold). Cần cả nhóm OK trong buổi họp G2-01.

## 6. Format file dự đoán (G0-05) — `results/preds/<run_id>__<eval_set>.parquet`

| Cột | Ý nghĩa |
|---|---|
| `city` | `"hcm"` |
| `station_id` | `"hcm_01"`..`"hcm_06"` |
| `time_origin` | thời điểm gốc t (giờ cuối của input), giờ địa phương |
| `horizon` | h ∈ {1, 6, 24}; thời điểm mục tiêu = `time_origin + h` giờ |
| `y_true` | PM2.5 thật (µg/m³); **NaN nếu không phải giá trị đo thật** |
| `y_pred` | dự đoán (µg/m³, đã đảo chuẩn hoá) |
| `model` | tên mô hình như trong file kết quả |
| `config` | = `run_id` (nối được với file kết quả) |

Dùng `src.eval.results.save_predictions` để ghi; hàm sẽ báo lỗi nếu thiếu cột.

## 7. Thang AQI (G1-58)

- **Thang chính: US EPA, bảng điểm gãy PM2.5 cập nhật 2024** (hiệu lực 06/05/2024), 6 mức:
  - Good: ≤ 9,0
  - Moderate: 9,1–35,4
  - Unhealthy for Sensitive Groups: 35,5–55,4
  - Unhealthy: 55,5–125,4
  - Very Unhealthy: 125,5–225,4
  - Hazardous: ≥ 225,5 µg/m³
- **Thang bổ sung: VN_AQI** theo QĐ 1459/QĐ-TCMT (2019). Theo nồng độ: ≤ 25 Tốt, ≤ 50 Trung bình, ≤ 80 Kém, ≤ 150 Xấu, ≤ 250 Rất xấu, > 250 Nguy hại.
- **Lý do chọn US EPA làm thang chính:**
  1. Mốc 35,4 µg/m³ dùng cho `MAE_high` chính là ranh giới Moderate/USG của EPA, nên hai độ đo dùng chung một mốc.
  2. VN_AQI giờ của PM2.5 chính thức tính trên giá trị NowCast 12 giờ, nên áp lên từng giờ là không đúng định nghĩa.
- **Ghi rõ trong báo cáo:** ta áp các *điểm gãy nồng độ* lên giá trị theo giờ ("mức nồng độ theo giờ"). Đây không phải AQI chính thức, vốn tính trên trung bình 24 giờ hoặc NowCast. macro-F1 lấy trung bình trên các mức có xuất hiện trong y_true.

## 8. Chạy lại end-to-end (G3-25)

```bash
# 1) Dữ liệu thô
python -m src.data.download_beijing          # UCI -> data/raw/beijing (+ kiểm tra md5)
#    HealthyAir: tải từ Mendeley (DOI 10.17632/pk6tzrjks8.1) -> data/raw/healthyair/AirQuality_hcmc.csv  (TV1)
#    Open-Meteo: python -m src.data.fetch_openmeteo ...                                                 (TV2)
# 2) Làm sạch
python -m src.data.beijing                    # beijing_raw.parquet, beijing_clean.parquet, bảng % thiếu trước/sau
#    TV1: -> data/interim/healthyair_clean.parquet   (tạm thời: python scripts/dev_hcm_minimal_clean.py)
#    TV2: -> data/interim/meteo_hcm.parquet, meteo_beijing.parquet
# 3) Lược đồ chung
python -m src.data.schema                     # data/processed/{hcm,beijing}_schema.parquet
pytest -q                                     # phải pass hết trước khi train
# 4) Baseline + EDA so sánh 2 thành phố
python -m src.baselines                       # results/baselines.csv + results/preds (test + 4 fold)
python -m src.baselines --city beijing        # mốc tham chiếu trên val Bắc Kinh
python -m src.eda.compare_cities              # bảng 4.1 cập nhật + hình
python -m src.data.splits                     # hình các fold
# 5) Mô hình
bash scripts/run_patchtst_all.sh              # TV3 — toàn bộ thí nghiệm PatchTST
#    TV1, TV2: script train của mình, ghi results/<tên-phương-pháp>.csv cùng mẫu cột
# 6) Tổng hợp
python -m src.aggregate_results               # gộp + kiểm tra mọi results/*.csv, bảng RQ3, hình đường cong
```

## 9. Cấu trúc repo

```
configs/          data.yaml (split, fold, scaler, cửa sổ), patchtst.yaml
src/data/         download_beijing, beijing, cleaning_common, schema, splits, scaling, windows, pipeline
src/eval/         metrics, aqi, results   (src/evaluate.py gom lại các hàm này)
src/models/       patchtst.py
src/baselines.py  src/train_patchtst.py  src/aggregate_results.py  src/eda/compare_cities.py
scripts/          dev_hcm_minimal_clean.py (tạm), run_patchtst_all.sh
tests/            test_pipeline.py
docs/notes/       ghi chú tài liệu và thiết kế;  docs/report/  bản nháp các mục báo cáo
results/          <phương-pháp>.csv, results_template.csv, preds/
reports/          bảng + hình cho báo cáo
```

## 10. Nguồn dữ liệu và giấy phép

- HealthyAir: Rakholia et al. (2023), Data in Brief 46:108774, CC BY 4.0. Dữ liệu ở Mendeley, DOI 10.17632/pk6tzrjks8.1.
- Beijing Multi-Site Air Quality: UCI, DOI 10.24432/C5RK5G, CC BY 4.0.
- Open-Meteo Historical Weather API (dữ liệu nền ERA5/ERA5-Land, ECMWF).
