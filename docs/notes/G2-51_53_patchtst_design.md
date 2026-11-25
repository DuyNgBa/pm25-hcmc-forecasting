# Ghi chú thiết kế PatchTST (G2-51, G2-52, G2-53)

*Người phụ trách: Bo bo (TV3). Code: `src/models/patchtst.py`, `src/train_patchtst.py`, cấu hình `configs/patchtst.yaml`.*

## Kiến thức cần có trước khi đọc

1. **Self-attention / Transformer** (Vaswani et al., 2017): mỗi token "nhìn" mọi token khác và lấy trung bình có trọng số.
2. **Chuẩn hoá theo cửa sổ (RevIN)** (Kim et al., 2022): trừ trung bình, chia độ lệch chuẩn của chính cửa sổ input rồi cộng lại ở đầu ra. Cách này giúp mô hình bớt nhạy với dịch chuyển phân phối (mùa khô so với mùa mưa, Bắc Kinh so với TP.HCM).
3. **PatchTST** (Nie et al., ICLR 2023): xem mục G2-51.

## G2-51 — Chọn cài đặt

| Phương án | Ưu | Nhược |
|---|---|---|
| Repo gốc `yuqinie98/PatchTST` | Chuẩn nhất | Viết cho bộ benchmark (ETT, Weather...): đọc CSV một chuỗi, không có mask nhãn thiếu, không có cách đưa kênh ngoại sinh vào dự báo PM2.5, khó gắn với pipeline cửa sổ chung |
| Thư viện (HF `PatchTSTForPrediction`, neuralforecast, tsai) | Có sẵn, ít code | Thêm phụ thuộc lớn; rất khó làm 3 việc ta cần: đóng băng backbone cho S2_head, nhúng thành phố cho S3, khí tượng tương lai; mỗi thư viện lại có cách xử lý NaN riêng |
| **Tự cài đặt, bám sát backbone của repo gốc (chọn)** | Chạy thẳng trên `WindowSet` chung (cùng cửa sổ, cùng mask, cùng hàm đánh giá với TV1/TV2); kiểm soát hết | Phải tự kiểm tra đúng: đã có smoke test cho mọi chiến lược |

Những gì giữ đúng như repo gốc (`PatchTST_supervised/layers/PatchTST_backbone.py`):

- RevIN có tham số affine.
- `padding_patch="end"`: đệm thêm S bước bằng giá trị cuối, nên số patch là N = ⌊(L−P)/S⌋ + 2.
- Nhúng patch tuyến tính, vị trí học được (khởi tạo U(−0,02; 0,02)).
- Encoder post-norm dùng **BatchNorm**, kích hoạt GELU, **residual attention** (điểm attention của lớp trước được cộng vào lớp sau).
- Đầu ra Flatten + Linear.

Phần tự viết: lớp attention viết tay, tương đương `_MultiheadAttention` của repo nhưng không có tuỳ chọn LSA.

## G2-52 — Đưa khí tượng và đặc trưng thời gian vào thế nào?

**Vấn đề.** PatchTST gốc là *channel-independent*: mỗi kênh đi qua encoder riêng rồi **tự dự báo chính nó**. Nếu đưa nhiệt độ hay gió vào như các kênh thêm, dự báo PM2.5 vẫn **không hề phụ thuộc** vào chúng. Khi đó ablation (c) và (d) sẽ cho kết quả giống hệt (a), và RQ2 trở nên vô nghĩa.

**Các phương án đã cân nhắc.**

1. *Chỉ PM2.5* (đúng bản gốc): đơn giản nhưng không trả lời được RQ2. → Vẫn giữ làm biến thể `head: target_only` để so sánh.
2. *Channel-mixing ở đầu vào* (gộp mọi kênh vào một patch): phá bỏ ưu điểm chính của PatchTST. Chính bài gốc cho thấy CI tốt hơn channel-mixing.
3. **Encoder CI + trộn kênh ở đầu ra (chọn, `head: fusion`)**: mọi kênh đi qua CÙNG encoder (chia sẻ trọng số, giữ nguyên tinh thần CI). Đầu ra flatten biểu diễn của **tất cả** kênh, C × N × d, rồi một lớp Linear dự báo 24 giờ PM2.5. Khí tượng tác động lên dự báo qua lớp Linear này.

**Các chi tiết đi kèm.**

- **RevIN chỉ áp lên kênh PM2.5.** Các kênh khác đã được z-score toàn cục theo train. Ngoài ra, chuẩn hoá theo cửa sổ sẽ làm hỏng các kênh gần hằng số: cờ `_isna` hoặc mưa = 0 có độ lệch chuẩn ≈ 0, chia cho nó sẽ nổ giá trị.
- **Đặc trưng thời gian** (`hour_sin/cos`, `dow_sin/cos`) được đưa vào như các kênh bình thường. Cách này cho phép fine-tune học lại chu kỳ ngày riêng của TP.HCM, đúng lý do ghi ở mục 4.2. Không dùng `month`.
- **S3 (train chung)**: dùng nhúng thành phố học được, cộng vào mọi token.
- **Khí tượng tương lai (ablation `c_oracle`)**: khí tượng thật tại t+1..t+24 được flatten rồi nối vào đầu vào của lớp Linear cuối. Không đưa qua encoder vì đây không phải chuỗi quá khứ.
- **Hàm mất mát**: MSE trên z, có mask. Giờ thiếu nhãn không tính vào loss. Giờ nội suy ngắn vẫn được dùng để train, nhưng không bao giờ được dùng để đánh giá.

## G2-53 — Patch length / stride theo L

| L | P | S | Số patch N | Ý nghĩa |
|---|---|---|---|---|
| 24 | 6 | 3 | 8 | mỗi patch = ¼ ngày |
| 48 | 6 | 3 | 16 | |
| 72 | 6 | 3 | 24 | |

**Lý do giữ P cố định cho cả 3 L.** Thí nghiệm G2-69 cần cô lập ảnh hưởng của *độ dài lịch sử*. Nếu P thay đổi theo L thì sẽ lẫn thêm ảnh hưởng của *kích thước token*. P = 6 đủ ngắn để bắt đỉnh sáng 6–9h của TP.HCM (đỉnh rộng khoảng 4–5 giờ), và đủ dài để mỗi token mang một đoạn hình dạng có ý nghĩa. S = P/2 cho các patch chồng lên nhau một nửa, giống tỉ lệ 16/8 của bài gốc.

Bài gốc dùng P = 16 vì L ≥ 336. Với L = 24, P = 16 chỉ cho 2 patch, khi đó attention gần như vô dụng.

HPO (G2-56) vẫn được phép chọn P ∈ {4, 6, 8, 12} trên **val** cho cấu hình chính. Nếu kết quả HPO khác 6, báo cáo cả hai.

## Siêu tham số mặc định (trước HPO)

`d_model = 64, n_heads = 4, n_layers = 3, d_ff = 128, dropout = 0,2, head_dropout = 0,1, AdamW lr = 1e-3, batch = 256`, early stopping với patience = 8, giảm lr ×0,5 khi val không giảm.

Mô hình nhỏ có chủ đích: TP.HCM chỉ có khoảng 23 nghìn cửa sổ train ở L = 48. Với ablation (a), mô hình có khoảng 0,2 triệu tham số.

Fine-tune (S2): `lr_full = 3e-4` (nhỏ hơn để không xoá kiến thức pretrain), `lr_head = 1e-3`.

## Kế hoạch chạy (G2-55 → G2-71)

Thứ tự đề xuất trên Kaggle GPU (P100/T4):

1. Khi đã có `meteo_hcm.parquet` của TV2, chạy HPO: `--hpo --ablation c --L 48` (30 trial).
2. Chạy cấu hình chính với 3 seed: `--strategy S0 --ablation c --L 48 --seeds 42 43 44` (G2-55, G2-57).
3. Rolling-origin: `--fold all` (G2-58).
4. Ablation a, b, c, d (G2-59 → 62); L = 24/48/72 (G2-69); `c_oracle` (G2-70).
5. RQ3: S1, S2_full, S2_head, S3 × train_frac 1/0,5/0,25 (G2-63 → 68).

Script `scripts/run_patchtst_all.sh` chạy toàn bộ theo đúng thứ tự trên.
