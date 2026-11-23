# Mạng hồi quy LSTM / GRU (G2-50 – TV2 Minh Trí)

> **Trạng thái (23/11/2026):** mục 1–3 (phương pháp) **viết xong**, mô tả đúng code hiện tại trong `src/models/` và module chung `src/data/`. Mục 4 (kết quả) **chờ lần chạy chính thức** trên `data-v1.0` bằng GPU (`notebooks/tridao/rnn_colab_kaggle.ipynb`). Các số ở Phụ lục A là **kết quả thử nghiệm** (dữ liệu `v0.9-dev`, khí tượng `best_match`, CPU), chỉ để kiểm tra pipeline, **không dùng để kết luận**. Đặc biệt, **chưa kết luận** khí tượng hay dữ liệu Bắc Kinh có ích hay không trên dữ liệu mới.
> Đơn vị µg/m³. Đoạn có ✍️ là chỗ Trí cần viết bằng giọng của mình sau khi có kết quả chính thức.

## 1. Bài toán, dữ liệu và cách đánh giá

### 1.1. Đầu vào và đầu ra

Tại mỗi trạm s và mỗi **gốc dự báo t** (một giờ cụ thể):

- **Đầu vào** `x`: ma trận L × F gồm F đặc trưng tại các giờ t−L+1, …, t (L giờ gần nhất, tính cả t). L ∈ {24, 48, 72}, chọn bằng tối ưu siêu tham số (mục 2.4).
- **Đầu ra** `ŷ`: PM2.5 tại **cả 24 giờ tới** t+1, …, t+24, tính trong **một lần chạy** mô hình. Báo cáo chính ở t+1, t+6, t+24 (quy ước chung). Dự đoán đủ 24 tầm được lưu thêm để tính "TB tầm 1..24" khi so với bài [2] (G3-10).

Mỗi cửa sổ chỉ lấy trong **một trạm**, không vắt qua hai trạm hay hai thành phố (`src/data/windows.py`, có test).

### 1.2. Đặc trưng theo từng thiết lập ablation (RQ2)

| Thiết lập | Đặc trưng (F) | Số kênh |
|---|---|---|
| (a) | PM2.5 đã chuẩn hoá `pm25_z` + sin/cos giờ trong ngày + sin/cos thứ trong tuần | 5 |
| (b) | (a) + nhiệt độ, độ ẩm **cảm biến HealthyAir** + 2 kênh cờ "thiếu" cho hai cột này | 9 |
| (c) | (a) + 10 biến **Open-Meteo** (`era5_seamless`): nhiệt độ, độ ẩm, điểm sương, tốc độ gió, gió u, gió v, mưa, mưa 6 giờ, mưa 24 giờ, áp suất | 15 |
| (d) | (c) + PM2.5 (z) của 5 trạm gần nhất cùng giờ + 5 kênh cờ "thiếu" | 25 |
| (c-oracle), G2-48 | (c) + khí tượng **thật** tại t+1..t+24 (10 biến × 24 giờ), mô phỏng "dự báo thời tiết hoàn hảo" | 15 + khối tương lai |

Khi một đặc trưng phụ bị thiếu, giá trị được điền 0 (bằng mức trung bình sau chuẩn hoá) và kênh cờ tương ứng bằng 1. Chỉ **PM2.5 đầu vào** là bắt buộc: cửa sổ có PM2.5 thiếu ở bất kỳ giờ nào trong L giờ đầu vào thì bị bỏ (giờ nội suy ≤ 3 giờ của TV1 vẫn được dùng). Nhờ quy tắc này, **mọi thiết lập (a)–(d) có cùng tập cửa sổ**, nên so sánh giữa chúng là công bằng.

Thiết lập chính (RQ1, LSTM vs GRU, Seq2Seq, RQ3) dùng **(c)**.

### 1.3. Chia dữ liệu (theo thời gian, không xáo trộn)

| Tập | TP.HCM | Bắc Kinh (chỉ để pretrain, RQ3) |
|---|---|---|
| Train | 23/02/2021 – 31/12/2021 | 01/03/2013 – 29/02/2016 |
| Val | 01/2022 – 02/2022 | 01/03/2016 – 28/02/2017 |
| Test | 01/03/2022 – 21/06/2022 | không có (Bắc Kinh không bao giờ vào test) |

- Một cửa sổ thuộc tập X khi **mọi giờ mục tiêu** t+1..t+24 nằm trong X. Với **train**, cả L giờ đầu vào cũng phải nằm trong train. Với val/test, đầu vào được lấy lịch sử trước đó, giống như khi dự báo thật.
- **Rolling-origin** (G2-35): 4 fold mở rộng dần. Mỗi fold có val 1 tháng và test 2 tháng, phủ liên tục 11/2021 → 06/2022 (gồm cả đỉnh mùa khô mà test chính không có).
- **Đường cong dữ liệu** (G2-45): giữ 50% hoặc 25% train TP.HCM, lấy **đoạn liên tục cuối** của train (sát val), mô phỏng "trạm mới lắp".
- **Tập đánh giá chung:** val/test chỉ giữ các gốc t hợp lệ với L = 72. Vì vậy mọi mô hình của nhóm, mọi L và mọi ablation được chấm trên **đúng cùng tập giờ** với persistence.

Số cửa sổ trên dữ liệu hiện tại (ablation c; có thể đổi nhẹ ở `data-v1.0`):

| L | Train | Val | Test |
|---|---|---|---|
| 24 | 26.910 | 7.298 | 10.326 |
| 48 | 23.240 | 7.298 | 10.326 |
| 72 | 20.264 | 7.298 | 10.326 |

L nhỏ thì có nhiều cửa sổ train hơn (cần ít giờ lịch sử liên tục hơn), còn val/test luôn giống nhau. Bắc Kinh (L = 24): 305.261 cửa sổ train, 101.719 cửa sổ val.

### 1.4. Chuẩn hoá và chống rò rỉ

- **PM2.5:** z = (log(1 + PM2.5) − μ_s) / σ_s, trong đó μ_s và σ_s tính **riêng cho từng trạm** s, **chỉ trên giờ đo thật của tập train**. Log nén đuôi phải (PM2.5 lệch phải mạnh). Z-score theo trạm bỏ mức nền riêng của trạm, nên mô hình học **hình dạng biến động**. Đây cũng là phần có thể chuyển giao giữa Bắc Kinh (TB ~80) và TP.HCM (TB ~21).
- **Khí tượng:** chuẩn hoá theo từng thành phố, trên train. Nhiệt độ, độ ẩm, điểm sương, áp suất, tốc độ gió dùng z-score. Gió u/v chỉ chia độ lệch chuẩn (giữ dấu = hướng gió). Mưa dùng log(1 + x) rồi chia độ lệch chuẩn (0 mm vẫn là 0).
- **Fit lại cho từng lần chia:** mỗi fold rolling-origin và mỗi mức 50%/25% đều fit lại scaler trên train của chính nó. Toàn bộ nằm trong một hàm chung `prepare()` của nhóm, nên không thể vô tình fit trên toàn bộ dữ liệu.
- **Không nhìn trước:** đặc trưng cuộn (mưa 6/24 giờ…) chỉ dùng giờ ≤ t (test G1-39). Trạm lân cận chỉ lấy giá trị cùng giờ trong cửa sổ đầu vào. Khí tượng tương lai chỉ có ở thiết lập phụ G2-48.
- **Đổi ngược:** dự đoán z được đổi về µg/m³ bằng exp(z·σ_s + μ_s) − 1 rồi mới tính độ đo.

### 1.5. Độ đo

MAE, RMSE, R² và MAE trên giờ ô nhiễm cao (> 35,4 µg/m³), tính trên µg/m³ và **chỉ trên giờ đo thật** (bỏ giờ thiếu và giờ nội suy). Mỗi kết quả kèm số giờ `n`; `n` phải bằng `n` của persistence ở cùng tập và cùng tầm (`src/models/rnn_check.py` kiểm tra). Ngoài ra có độ chính xác và macro-F1 của mức AQI quy đổi (hàm chung `src/eval/`).

## 2. Mô hình và huấn luyện

### 2.1. Dự báo trực tiếp nhiều bước: LSTM (G2-29), GRU (G2-30)

```
x (L × F) → RNN nhiều lớp (LSTM hoặc GRU) → trạng thái ẩn ở giờ t (h_L) → Dropout → Linear(hidden → 24) → Δ(1..24)
ŷ(t+k) = y(t) + Δk          (y(t) = PM2.5 dạng z tại gốc t, k = 1..24)
```

- Encoder là LSTM hoặc GRU nhiều lớp, có dropout giữa các lớp (khi số lớp > 1). Chỉ dùng trạng thái ẩn ở bước cuối.
- Một lớp tuyến tính sinh **cả 24 tầm cùng lúc**, nên không có tích luỹ sai số giữa các tầm.
- GRU có 3 cổng và LSTM có 4 cổng, nên với cùng kích thước thì GRU ít tham số hơn khoảng 25%. Kích thước thực tế do HPO chọn riêng cho từng cell.

### 2.2. Encoder–decoder (Seq2Seq, G2-31)

```
Encoder RNN đọc x → trạng thái (h, c)
Decoder RNN (cùng loại cell, cùng số lớp, cùng hidden) chạy 24 bước:
   đầu vào bước k = dự báo bước k−1 (bước đầu = y(t)) [+ khí tượng thật tại t+k nếu là c-oracle]
   ŷ(t+k) = ŷ(t+k−1) + δk,   δk = Linear(hidden → 1)
```

- Khi train dùng **teacher forcing**: ở mỗi bước, với xác suất p, đầu vào bước sau là **giá trị thật** thay cho dự báo (nếu giá trị thật có; thiếu thì vẫn dùng dự báo). p giảm tuyến tính từ 0,5 về 0 trong nửa đầu số epoch tối đa. Khi đánh giá thì không dùng teacher forcing.
- Seq2Seq dùng lại **bộ siêu tham số của dự báo trực tiếp cùng cell** (không tối ưu riêng). Nhờ vậy G2-47 so được "cách sinh đầu ra" khi mọi yếu tố khác giữ nguyên.

### 2.3. Kết nối phần dư (residual)

Cả hai kiến trúc dự báo **độ thay đổi so với PM2.5 tại t** thay vì mức tuyệt đối. Vì y ở dạng z = (log(1+x) − μ)/σ, một Δ trong không gian z tương ứng với một **tỉ lệ** tăng/giảm của (1 + PM2.5). Nếu mô hình chưa học được gì (Δ = 0), dự báo đúng bằng persistence. Mô hình chỉ phải học **phần lệch khỏi persistence**, nên khó overfit hơn. Đầu vào không đổi, nên so sánh với mô hình khác của nhóm vẫn công bằng.

Lý do chọn: ở lượt chạy thử đầu tiên (v1, không residual), mô hình overfit sau 2–3 epoch. Trên **val**, MAE tốt nhất qua 30 lần thử là 8,04 (v1) so với 7,93 (residual, LSTM). Kết quả v1 được giữ ở `results/archive/rnn_v1_no_residual/`. ✍️ Khi viết báo cáo, nên dẫn chứng lựa chọn này bằng **val**, không bằng test.

### 2.4. Hàm mất mát, mask và vòng huấn luyện (G2-28)

- **Loss:** MSE trong không gian z, **chỉ tính trên các mục tiêu có giá trị** (mask = mục tiêu không thiếu). Cửa sổ được giữ khi trong 24 giờ tới có **ít nhất 1 giờ đo thật**; các giờ mục tiêu thiếu không đóng góp vào loss.
  - ⚠️ Mask khi **train** coi giờ **nội suy** (khoảng trống ≤ 3 giờ, nội suy tuyến tính hai phía) là có giá trị (0,45% mục tiêu train), còn khi **đánh giá** chỉ dùng giờ đo thật. Gốc t có PM2.5 nội suy (khoảng 0,5%) làm đầu vào chứa giá trị của 1–3 giờ sau. Đây là chính sách chung hiện tại của module cửa sổ, đang chờ nhóm chốt (mục 3, câu hỏi mở 2).
- **Tối ưu:** AdamW (weight decay 1e-5), learning rate và batch size do HPO chọn, cắt gradient ở chuẩn 1,0. Learning rate giảm một nửa khi loss val không giảm trong 3 epoch.
- **Early stopping:** sau mỗi epoch, tính cùng loss có mask trên **val** (tập gốc t chung). Dừng khi val không cải thiện quá 1e-4 trong **8 epoch** liên tiếp (tối đa 60 epoch), rồi **nạp lại trọng số của epoch tốt nhất**. Checkpoint tốt nhất được lưu ở `checkpoints/rnn/`.
- **Seed:** 42, 43, 44 cho các cấu hình chính. Seed cố định cả Python, NumPy, PyTorch và bật chế độ cuDNN tất định; trên GPU vẫn có thể lệch rất nhỏ giữa các lần chạy.

### 2.5. Tối ưu siêu tham số (G2-33)

Optuna, sampler TPE (seed 42), **30 lần thử trên val** cho mỗi cell, ablation (c), chiến lược S0, seed 42. Mục tiêu là **MAE trên val (µg/m³), gộp 24 tầm, chỉ giờ đo thật**. Tập test không được dùng ở bước này. Mỗi lần thử train tối đa 40 epoch, early stopping 6 epoch.

| Siêu tham số | Không gian tìm |
|---|---|
| L (giờ lịch sử) | {24, 48, 72} |
| hidden | {32, 64, 128} |
| Số lớp | {1, 2, 3} |
| dropout | [0; 0,4] |
| learning rate | [3e-4; 3e-3], thang log |
| batch size | {128, 256, 512} |

Bộ tốt nhất được ghi vào `configs/rnn.yaml` → `best`, **kèm dấu vân tay dữ liệu** (`data_fp`). Mọi thí nghiệm sau đó tự dùng bộ này. Ablation (a), (b), (d) dùng lại bộ của (c) (không tối ưu riêng, giữ ngân sách chung 30 lần thử).

### 2.6. Các chiến lược dùng dữ liệu Bắc Kinh (RQ3, G2-40 → G2-45)

Chỉ áp dụng cho dự báo trực tiếp, ablation (c). Cả hai thành phố dùng chung lược đồ và chung nguồn khí tượng Open-Meteo.

| Chiến lược | Cách làm trong code |
|---|---|
| S0 | Chỉ train trên TP.HCM (thiết lập chính) |
| S1 zero-shot | Train trên Bắc Kinh (tối đa 15 epoch, early stopping 3 epoch theo val Bắc Kinh), test thẳng trên TP.HCM |
| S2 fine-tune toàn bộ | Lấy mô hình S1, train tiếp mọi tham số trên train TP.HCM với learning rate × 0,3 |
| S2 chỉ lớp cuối | Như trên nhưng đóng băng encoder, chỉ train lớp Linear cuối |
| S3 train chung | Gộp cửa sổ hai thành phố, thêm kênh `is_hcm` (1 = TP.HCM). Mỗi epoch lấy mẫu có hoàn lại 50% TP.HCM, 50% Bắc Kinh, số mẫu = 2 × số cửa sổ train TP.HCM. Early stopping theo val TP.HCM |

Đường cong dữ liệu chạy S0, S2 toàn bộ và S3 ở 50% và 25% train TP.HCM. Checkpoint pretrain Bắc Kinh được dùng chung giữa S1, S2 và đường cong dữ liệu. Tên file checkpoint gắn với dấu vân tay dữ liệu và cấu hình huấn luyện, nên không dùng nhầm checkpoint của dữ liệu cũ.

### 2.7. Tái lập và chống dùng nhầm kết quả cũ

- Mỗi lần chạy có `run_id` tính từ cấu hình **và** `data_version` + **dấu vân tay nội dung dữ liệu** (`data_fp`, băm hai bảng lược đồ chung TP.HCM và Bắc Kinh) + dấu vân tay mục huấn luyện của `configs/rnn.yaml`. Dữ liệu đổi (kể cả khi quên đổi `data_version`) thì `run_id` đổi. Khi đó kết quả, checkpoint pretrain và lần thử Optuna cũ **không** bị dùng lại.
- `rnn_experiments` **dừng** nếu `best` trong `configs/rnn.yaml` được tối ưu trên dữ liệu khác.
- `rnn_report` chỉ tổng hợp **một** bộ dữ liệu. Nó chỉ ghi "chính thức" khi dữ liệu của mọi lần chạy **khớp bản khoá của nhóm** `data/DATA_LOCK.json` (so `raw_fp`, `hcm_fp`, `beijing_fp`, `data_version = v1.0`) **và** `rnn_check` đạt. Chỉ ghi `v1.0` trong config thì chưa đủ.
- `rnn_check` (G2-49) kiểm tra: đủ cấu hình/seed/tầm; file dự đoán đúng format; **cùng tập (trạm, gốc t, tầm) và cùng y_true từng dòng với persistence**; y_true = PM2.5 đo thật tại t+h trong dữ liệu; độ đo tính lại từ dự đoán khớp bảng kết quả.

## 3. Đề xuất, quy ước chung và lựa chọn riêng của TV2

| Nội dung | Đề xuất (PDF v1) | Quy ước chung trong code chung (TV3 + recheck của nhóm trưởng) | Lựa chọn của TV2 (chưa có biên bản chốt G2-01) |
|---|---|---|---|
| Chia dữ liệu | train 23/02–31/12/2021, val 01–02/2022, test 03–06/2022 + rolling-origin | Như đề xuất; test kết thúc 21/06/2022; 4 fold cụ thể (`configs/data.yaml`) | – |
| Cửa sổ | L = 24, thử thêm 48 và 72 | L ∈ {24, 48, 72}, H = 24; tập đánh giá chung theo L = 72 | Chọn L bằng HPO trong {24, 48, 72} |
| Đầu ra DL | Dự báo cả chuỗi 1–24 giờ cùng lúc | Báo cáo t+1, t+6, t+24 | Lưu thêm đủ 24 tầm (`*_h24.parquet`) để so với bài [2] |
| Chuẩn hoá | Fit trên train | log1p + z-score theo trạm; khí tượng theo thành phố | – |
| Khí tượng | Open-Meteo dùng chung 2 thành phố | `models=era5_seamless` (recheck 22/11/2026) | – |
| Ngân sách HPO | "Ví dụ 30 lần thử trên val" | – | 30 lần thử Optuna TPE seed 42, mục tiêu MAE val TB 24 tầm, tối ưu trên (c), dùng lại cho a/b/d và Seq2Seq |
| Seed | – | `seed = 42` | 3 seed (42/43/44) cho RQ1, RQ2, Seq2Seq; **1 seed** cho rolling-origin, RQ3, đường cong dữ liệu, G2-48 |
| Thiết lập chính | – | – | Ablation (c), S0 |
| Kiến trúc | LSTM, GRU; so seq2seq với dự báo trực tiếp | – | Kết nối phần dư cho mọi mô hình RNN |
| RQ3 | S0–S3; fine-tune toàn bộ hoặc chỉ lớp cuối; 100/50/25% | – | Pretrain 15 epoch; fine-tune lr × 0,3; S3 cân bằng 50/50 |
| Dữ liệu cuối | – | `data-v1.0` (G1-66) | Mọi số trong báo cáo lấy từ lần chạy GPU trên `data-v1.0` |

**Câu hỏi mở cần nhóm/Trí quyết trước lần chạy chính thức:**
1. Chốt các mục ở cột "Lựa chọn của TV2" (họp G2-01), đặc biệt là số seed cho rolling-origin và RQ3. Nếu chạy 1 seed thì không đủ để kết luận về các chênh lệch nhỏ.
2. Xử lý PM2.5 **nội suy** (đề xuất riêng: `đồ án/de_xuat_chinh_sach_noi_suy.md`). Hiện loss train và early stopping tính cả nhãn nội suy (0,45% mục tiêu train). Ngoài ra, khi PM2.5 tại gốc t là nội suy (0,5% gốc test), đầu vào chứa giá trị đo thật ở t+1..t+3, áp dụng cho mọi mô hình kể cả persistence. Đề xuất phương án B: nhãn train/early stopping/HPO chỉ dùng đo thật, và bỏ gốc t nội suy ở module chung. Chờ nhóm chốt trước `data-v1.0`.

## 4. Kết quả (chờ lần chạy chính thức `data-v1.0` trên GPU)

Sinh tự động bởi `python -m src.models.rnn_report` sau khi giải nén `rnn_results.zip` từ Kaggle và `python -m src.models.rnn_check` **đạt**. Đầu `reports/rnn/summary.md` phải ghi **"KẾT QUẢ CHÍNH THỨC"** thì mới chép số vào đây.

| Mục | Bảng nguồn (`reports/rnn/`) | Trạng thái |
|---|---|---|
| RQ1 – test chính, 3 seed (G2-32, G2-34) | `rq1_test.csv`, `rq1_vs_baseline.csv` | ⏳ chờ GPU |
| RQ1 – rolling-origin (G2-35) | `rq1_rolling.csv` | ⏳ chờ GPU |
| G2-46 – LSTM vs GRU | `g2_46_lstm_vs_gru.csv`, `g2_46_cost.csv` | ⏳ chờ GPU (code tổng hợp xong) |
| G2-47 – Seq2Seq vs trực tiếp | `g2_47_seq2seq_vs_direct.csv` | ⏳ chờ GPU (code tổng hợp xong) |
| RQ2 – ablation a/b/c/d (G2-36..39) | `rq2_ablation.csv` | ⏳ chờ GPU |
| RQ3 – S0–S3 + đường cong (G2-40..45) | `rq3_transfer.csv`, `figures/rq3_curve.png` | ⏳ chờ GPU |
| G2-48 – khí tượng tương lai | `oracle.csv` | ⏳ chờ GPU |
| G2-49 – kiểm tra dự đoán | `g2_49_check.md` | ⏳ chờ GPU (script xong, đã chạy đạt trên kết quả thử nghiệm) |
| Tham chiếu bài [2] (G3-10) | `vs_paper2.csv` | ⏳ chờ GPU |

✍️ Sau khi có số chính thức, viết nhận xét cho từng mục: (i) RNN hơn persistence bao nhiêu ở từng tầm, có giữ trên cả 4 fold không; (ii) GRU hay LSTM, chênh lệch có lớn hơn độ lệch giữa các seed không, chi phí ra sao; (iii) Seq2Seq có tích luỹ sai số ở t+24 không; (iv) khí tượng hiện tại / tương lai / trạm lân cận giúp bao nhiêu; (v) dữ liệu Bắc Kinh giúp ở mức dữ liệu nào.

## Phụ lục A. Kết quả THỬ NGHIỆM (không dùng để kết luận)

Chạy ngày 22/11/2026 trên CPU, `data v0.9-dev`, khí tượng Open-Meteo `best_match` (trước khi nhóm cố định `era5_seamless`). `rnn_check` đạt trên bộ này (đủ 60 lần chạy, y_true khớp persistence và khớp PM2.5 đo thật). Khi thử lại 1 lần GRU seed 42 với khí tượng mới, ablation (c) kém đi 1–2% ở t+6/t+24 còn (b) gần như không đổi. Vì vậy các số dưới đây **sẽ thay đổi**, và các nhận định về khí tượng/Bắc Kinh phải chờ dữ liệu mới. Bảng đầy đủ: `reports/rnn/summary.md` (đầu file có ghi "KẾT QUẢ THỬ NGHIỆM").

| Mô hình (ablation c, test, TB 3 seed) | MAE t+1 | MAE t+6 | MAE t+24 |
|---|---|---|---|
| Persistence | 3,13 | 7,93 | 8,06 |
| GRU trực tiếp | 3,05 | 6,25 | 6,90 |
| LSTM trực tiếp | 3,11 | 6,52 | 7,06 |
| Seq2Seq-GRU | 3,03 | 6,26 | 7,40 |
| Seq2Seq-LSTM | 3,06 | 6,51 | 7,24 |

Quan sát trên bộ thử nghiệm, **cần kiểm chứng lại**, chưa phải kết luận: RNN hơn persistence rõ ở t+6/t+24 nhưng rất ít ở t+1; Seq2Seq kém hơn ở t+24 với dao động giữa seed lớn hơn; các thiết lập (a)/(b)/(c) chênh nhau cỡ độ lệch giữa seed; (d) kém hơn; RQ3 chỉ có 1 seed.

## Tái lập

GPU (khuyến nghị): `notebooks/tridao/rnn_colab_kaggle.ipynb`. Hoặc chạy tay:
`python -m src.models.hpo_rnn --cell lstm` và `--cell gru` → `python -m src.models.rnn_experiments --group all --device cuda` → `python -m src.models.rnn_check --require-gpu` → `python -m src.models.rnn_report`.
