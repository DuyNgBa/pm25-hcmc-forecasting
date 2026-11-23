# Kết quả LSTM / GRU / Seq2Seq (TV2 – Minh Trí)

> **KẾT QUẢ THỬ NGHIỆM (dữ liệu cũ).** Chạy trên CPU với `data v0.9-dev` và khí tượng Open-Meteo `best_match` (trước khi nhóm cố định `era5_seamless`). Chỉ dùng để thử pipeline, KHÔNG dùng để kết luận; mọi số sẽ được thay bằng lần chạy trên `data-v1.0` (GPU).

Bộ dữ liệu: `data_version=v0.9-dev`, `data_fp=legacy`, khí tượng `None`; thiết bị {'cpu': 60}; 60 lần chạy.

Sinh tự động bởi `python -m src.models.rnn_report` từ `results/rnn.csv`. Đơn vị µg/m³, chỉ tính trên giờ đo thật,
tập gốc t chung với mọi mô hình của nhóm (cột n bằng nhau). ± = độ lệch chuẩn qua các seed.

_Các nhận xét về khí tượng (RQ2) và dữ liệu Bắc Kinh (RQ3) trong báo cáo CHƯA được rút ra từ bảng này._

## Ghi chú thiết kế: vì sao dùng kết nối phần dư (residual)

Lượt chạy đầu (v1) dự báo thẳng mức PM2.5. Mô hình overfit sau 2–3 epoch (train loss giảm mạnh, val loss tăng),
early stopping dừng sớm và LSTM trực tiếp **thua persistence ở t+1**. Bản v2 dự báo độ thay đổi so với PM2.5 tại t
(ŷ(t+k) = y(t) + Δk). Đầu vào giữ nguyên; HPO 30 lần thử được chạy LẠI cho v2. Kết quả v1 lưu ở
`results/archive/rnn_v1_no_residual/` (MAE, TB 3 seed):

| version                            | model   |   MAE_h1 |   MAE_h6 |   MAE_h24 |   MAE_avg24 |   best_epoch |
|:-----------------------------------|:--------|---------:|---------:|----------:|------------:|-------------:|
| v1 không residual                  | LSTM    |    4.408 |    6.495 |     6.983 |       6.651 |        2     |
| v1 không residual                  | Seq2Seq |    3.188 |    6.34  |     7.221 |       6.585 |        3.667 |
| v2 residual (dùng cho mọi kết quả) | GRU     |    3.045 |    6.251 |     6.896 |       6.391 |        3     |
| v2 residual (dùng cho mọi kết quả) | LSTM    |    3.108 |    6.515 |     7.058 |       6.648 |        4.667 |
| v2 residual (dùng cho mọi kết quả) | Seq2Seq |    3.046 |    6.386 |     7.319 |       6.649 |        3.667 |

v1 chạy trên `v0.9-dev` + khí tượng `best_match`; chỉ để minh hoạ lý do chọn residual.

Persistence trên cùng test: MAE t+1 = 3.13; t+6 = 7.93; t+24 = 8.06.

## RQ1 – Test chính (G2-34, G2-32: TB ± std 3 seed)

| model   | cell   |   horizon | MAE           | RMSE           | R2            | MAE_high       |   n_seeds |
|:--------|:-------|----------:|:--------------|:---------------|:--------------|:---------------|----------:|
| GRU     | gru    |         1 | 3.045 ± 0.018 | 5.203 ± 0.026  | 0.796 ± 0.002 | 9.425 ± 0.261  |         3 |
| GRU     | gru    |         6 | 6.251 ± 0.041 | 9.408 ± 0.066  | 0.336 ± 0.009 | 20.918 ± 0.624 |         3 |
| GRU     | gru    |        24 | 6.896 ± 0.034 | 10.436 ± 0.031 | 0.209 ± 0.005 | 24.355 ± 0.364 |         3 |
| LSTM    | lstm   |         1 | 3.108 ± 0.045 | 5.305 ± 0.056  | 0.788 ± 0.004 | 9.975 ± 0.383  |         3 |
| LSTM    | lstm   |         6 | 6.515 ± 0.151 | 9.813 ± 0.191  | 0.277 ± 0.028 | 22.03 ± 0.643  |         3 |
| LSTM    | lstm   |        24 | 7.058 ± 0.153 | 10.678 ± 0.211 | 0.172 ± 0.033 | 24.381 ± 0.522 |         3 |
| Seq2Seq | gru    |         1 | 3.032 ± 0.016 | 5.176 ± 0.028  | 0.798 ± 0.002 | 9.249 ± 0.407  |         3 |
| Seq2Seq | gru    |         6 | 6.264 ± 0.033 | 9.418 ± 0.244  | 0.334 ± 0.035 | 20.862 ± 1.618 |         3 |
| Seq2Seq | gru    |        24 | 7.398 ± 0.223 | 10.66 ± 0.117  | 0.175 ± 0.018 | 22.987 ± 1.837 |         3 |
| Seq2Seq | lstm   |         1 | 3.059 ± 0.03  | 5.259 ± 0.073  | 0.792 ± 0.006 | 9.588 ± 0.578  |         3 |
| Seq2Seq | lstm   |         6 | 6.509 ± 0.032 | 9.855 ± 0.276  | 0.271 ± 0.04  | 22.308 ± 2.262 |         3 |
| Seq2Seq | lstm   |        24 | 7.239 ± 0.219 | 10.826 ± 0.227 | 0.149 ± 0.036 | 24.885 ± 1.402 |         3 |

## RQ1 – So với baseline

| model   | cell   |   horizon |   MAE |   MAE_persistence |   MAE_hourly_mean |   skill_vs_persistence_pct |
|:--------|:-------|----------:|------:|------------------:|------------------:|---------------------------:|
| GRU     | gru    |         1 | 3.045 |             3.134 |             8.285 |                        2.9 |
| GRU     | gru    |         6 | 6.251 |             7.928 |             8.287 |                       21.2 |
| GRU     | gru    |        24 | 6.896 |             8.058 |             8.327 |                       14.4 |
| LSTM    | lstm   |         1 | 3.108 |             3.134 |             8.285 |                        0.8 |
| LSTM    | lstm   |         6 | 6.515 |             7.928 |             8.287 |                       17.8 |
| LSTM    | lstm   |        24 | 7.058 |             8.058 |             8.327 |                       12.4 |
| Seq2Seq | gru    |         1 | 3.032 |             3.134 |             8.285 |                        3.3 |
| Seq2Seq | gru    |         6 | 6.264 |             7.928 |             8.287 |                       21   |
| Seq2Seq | gru    |        24 | 7.398 |             8.058 |             8.327 |                        8.2 |
| Seq2Seq | lstm   |         1 | 3.059 |             3.134 |             8.285 |                        2.4 |
| Seq2Seq | lstm   |         6 | 6.509 |             7.928 |             8.287 |                       17.9 |
| Seq2Seq | lstm   |        24 | 7.239 |             8.058 |             8.327 |                       10.2 |

## RQ1 – Rolling-origin 4 fold (G2-35), MAE

| model   |   horizon |   fold1 |   fold2 |   fold3 |   fold4 |   mean_4_fold |   persistence_mean_4_fold |
|:--------|----------:|--------:|--------:|--------:|--------:|--------------:|--------------------------:|
| GRU     |         1 |   4.788 |   3.913 |   2.864 |   3.217 |         3.696 |                     3.863 |
| GRU     |         6 |   9.167 |   8.201 |   5.902 |   6.31  |         7.395 |                     9.934 |
| GRU     |        24 |   9.286 |   8.2   |   6.64  |   6.81  |         7.734 |                     9.328 |
| LSTM    |         1 |   4.806 |   3.974 |   3.062 |   3.243 |         3.771 |                     3.863 |
| LSTM    |         6 |   9.542 |   8.488 |   7.089 |   6.651 |         7.943 |                     9.934 |
| LSTM    |        24 |   9.425 |   8.507 |   8.367 |   7.048 |         8.337 |                     9.328 |

## G2-46 – LSTM vs GRU: sai số (trực tiếp, ablation c, S0, test)

|   horizon | LSTM MAE      |   LSTM RMSE |   LSTM MAE_high |   LSTM số seed | GRU MAE       |   GRU RMSE |   GRU MAE_high |   GRU số seed |   GRU so với LSTM (MAE, %) |
|----------:|:--------------|------------:|----------------:|---------------:|:--------------|-----------:|---------------:|--------------:|---------------------------:|
|         1 | 3.108 ± 0.045 |       5.305 |           9.975 |              3 | 3.045 ± 0.018 |      5.203 |          9.425 |             3 |                       -2   |
|         6 | 6.515 ± 0.151 |       9.813 |          22.03  |              3 | 6.251 ± 0.041 |      9.408 |         20.918 |             3 |                       -4.1 |
|        24 | 7.058 ± 0.153 |      10.678 |          24.381 |              3 | 6.896 ± 0.034 |     10.436 |         24.355 |             3 |                       -2.3 |

## G2-46 – LSTM vs GRU: kích thước và chi phí

| model   |   L |   params |   n_seeds |   train_sec_TB |   epoch_tot_nhat_TB | thiet_bi   |   giay_moi_epoch | hidden/layers/dropout/lr/batch   |
|:--------|----:|---------:|----------:|---------------:|--------------------:|:-----------|-----------------:|:---------------------------------|
| GRU     |  24 |    67032 |         3 |         129.53 |                3    | cpu        |            11.78 | 64/3/0.32/0.00099/256            |
| LSTM    |  48 |   341528 |         3 |         276.6  |                4.67 | cpu        |            21.84 | 128/3/0.392/0.000801/512         |

## G2-47 – Seq2Seq vs dự báo trực tiếp (cùng cell, cùng siêu tham số)

| cell   |   horizon | direct MAE    |   direct số seed | seq2seq MAE   |   seq2seq số seed |   Seq2Seq so với trực tiếp (MAE, %) |   thời gian train Seq2Seq / trực tiếp |
|:-------|----------:|:--------------|-----------------:|:--------------|------------------:|------------------------------------:|--------------------------------------:|
| gru    |         1 | 3.045 ± 0.018 |                3 | 3.032 ± 0.016 |                 3 |                                -0.4 |                                  2.46 |
| gru    |         6 | 6.251 ± 0.041 |                3 | 6.264 ± 0.033 |                 3 |                                 0.2 |                                  2.46 |
| gru    |        24 | 6.896 ± 0.034 |                3 | 7.398 ± 0.223 |                 3 |                                 7.3 |                                  2.46 |
| lstm   |         1 | 3.108 ± 0.045 |                3 | 3.059 ± 0.03  |                 3 |                                -1.6 |                                  1.67 |
| lstm   |         6 | 6.515 ± 0.151 |                3 | 6.509 ± 0.032 |                 3 |                                -0.1 |                                  1.67 |
| lstm   |        24 | 7.058 ± 0.153 |                3 | 7.239 ± 0.219 |                 3 |                                 2.6 |                                  1.67 |

## RQ2 – Ablation (G2-36..39), MAE TB 3 seed

| model   | ablation   |   horizon |   mean |   std |   size |   improve_vs_a_pct |
|:--------|:-----------|----------:|-------:|------:|-------:|-------------------:|
| GRU     | a          |         1 |  3.034 | 0.012 |      3 |                0   |
| GRU     | a          |         6 |  6.163 | 0.043 |      3 |                0   |
| GRU     | a          |        24 |  6.849 | 0.047 |      3 |                0   |
| GRU     | b          |         1 |  3.042 | 0.027 |      3 |               -0.3 |
| GRU     | b          |         6 |  6.136 | 0.049 |      3 |                0.4 |
| GRU     | b          |        24 |  6.742 | 0.055 |      3 |                1.6 |
| GRU     | c          |         1 |  3.045 | 0.018 |      3 |               -0.3 |
| GRU     | c          |         6 |  6.251 | 0.041 |      3 |               -1.4 |
| GRU     | c          |        24 |  6.896 | 0.034 |      3 |               -0.7 |
| GRU     | d          |         1 |  3.081 | 0.017 |      3 |               -1.5 |
| GRU     | d          |         6 |  6.633 | 0.149 |      3 |               -7.6 |
| GRU     | d          |        24 |  7.241 | 0.142 |      3 |               -5.7 |
| LSTM    | a          |         1 |  3.049 | 0.02  |      3 |                0   |
| LSTM    | a          |         6 |  6.26  | 0.079 |      3 |                0   |
| LSTM    | a          |        24 |  6.843 | 0.05  |      3 |                0   |
| LSTM    | b          |         1 |  3.035 | 0.018 |      3 |                0.5 |
| LSTM    | b          |         6 |  6.237 | 0.109 |      3 |                0.4 |
| LSTM    | b          |        24 |  6.944 | 0.192 |      3 |               -1.5 |
| LSTM    | c          |         1 |  3.108 | 0.045 |      3 |               -1.9 |
| LSTM    | c          |         6 |  6.515 | 0.151 |      3 |               -4.1 |
| LSTM    | c          |        24 |  7.058 | 0.153 |      3 |               -3.1 |
| LSTM    | d          |         1 |  3.193 | 0.096 |      3 |               -4.7 |
| LSTM    | d          |         6 |  7.461 | 0.705 |      3 |              -19.2 |
| LSTM    | d          |        24 |  7.719 | 0.356 |      3 |              -12.8 |

## RQ3 – Chiến lược dùng dữ liệu Bắc Kinh (G2-40..45), MAE

| model   | strategy   |   train_frac |   horizon |   MAE |
|:--------|:-----------|-------------:|----------:|------:|
| GRU     | S0         |         0.25 |         1 | 3.148 |
| GRU     | S0         |         0.25 |         6 | 6.974 |
| GRU     | S0         |         0.25 |        24 | 7.353 |
| GRU     | S0         |         0.5  |         1 | 3.106 |
| GRU     | S0         |         0.5  |         6 | 6.475 |
| GRU     | S0         |         0.5  |        24 | 7.046 |
| GRU     | S0         |         1    |         1 | 3.045 |
| GRU     | S0         |         1    |         6 | 6.251 |
| GRU     | S0         |         1    |        24 | 6.896 |
| GRU     | S1         |         1    |         1 | 3.156 |
| GRU     | S1         |         1    |         6 | 7.084 |
| GRU     | S1         |         1    |        24 | 7.758 |
| GRU     | S2_full    |         0.25 |         1 | 3.182 |
| GRU     | S2_full    |         0.25 |         6 | 7.208 |
| GRU     | S2_full    |         0.25 |        24 | 7.598 |
| GRU     | S2_full    |         0.5  |         1 | 3.145 |
| GRU     | S2_full    |         0.5  |         6 | 6.778 |
| GRU     | S2_full    |         0.5  |        24 | 7.349 |
| GRU     | S2_full    |         1    |         1 | 3.042 |
| GRU     | S2_full    |         1    |         6 | 6.212 |
| GRU     | S2_full    |         1    |        24 | 6.867 |
| GRU     | S2_head    |         1    |         1 | 3.073 |
| GRU     | S2_head    |         1    |         6 | 6.899 |
| GRU     | S2_head    |         1    |        24 | 7.16  |
| GRU     | S3         |         0.25 |         1 | 3.164 |
| GRU     | S3         |         0.25 |         6 | 7.281 |
| GRU     | S3         |         0.25 |        24 | 8.243 |
| GRU     | S3         |         0.5  |         1 | 3.105 |
| GRU     | S3         |         0.5  |         6 | 6.918 |
| GRU     | S3         |         0.5  |        24 | 8.163 |
| GRU     | S3         |         1    |         1 | 3.053 |
| GRU     | S3         |         1    |         6 | 6.234 |
| GRU     | S3         |         1    |        24 | 7.291 |
| LSTM    | S0         |         0.25 |         1 | 3.152 |
| LSTM    | S0         |         0.25 |         6 | 6.847 |
| LSTM    | S0         |         0.25 |        24 | 7.24  |
| LSTM    | S0         |         0.5  |         1 | 3.072 |
| LSTM    | S0         |         0.5  |         6 | 6.642 |
| LSTM    | S0         |         0.5  |        24 | 7.122 |
| LSTM    | S0         |         1    |         1 | 3.108 |
| LSTM    | S0         |         1    |         6 | 6.515 |
| LSTM    | S0         |         1    |        24 | 7.058 |
| LSTM    | S1         |         1    |         1 | 3.173 |
| LSTM    | S1         |         1    |         6 | 7.332 |
| LSTM    | S1         |         1    |        24 | 8.124 |
| LSTM    | S2_full    |         0.25 |         1 | 3.206 |
| LSTM    | S2_full    |         0.25 |         6 | 7.344 |
| LSTM    | S2_full    |         0.25 |        24 | 7.876 |
| LSTM    | S2_full    |         0.5  |         1 | 3.403 |
| LSTM    | S2_full    |         0.5  |         6 | 7.708 |
| LSTM    | S2_full    |         0.5  |        24 | 8     |
| LSTM    | S2_full    |         1    |         1 | 3.1   |
| LSTM    | S2_full    |         1    |         6 | 6.671 |
| LSTM    | S2_full    |         1    |        24 | 7.298 |
| LSTM    | S2_head    |         1    |         1 | 3.069 |
| LSTM    | S2_head    |         1    |         6 | 6.762 |
| LSTM    | S2_head    |         1    |        24 | 7.203 |
| LSTM    | S3         |         0.25 |         1 | 3.182 |
| LSTM    | S3         |         0.25 |         6 | 7.434 |
| LSTM    | S3         |         0.25 |        24 | 7.833 |
| LSTM    | S3         |         0.5  |         1 | 3.136 |
| LSTM    | S3         |         0.5  |         6 | 6.877 |
| LSTM    | S3         |         0.5  |        24 | 7.85  |
| LSTM    | S3         |         1    |         1 | 3.087 |
| LSTM    | S3         |         1    |         6 | 6.367 |
| LSTM    | S3         |         1    |        24 | 7.061 |

## G2-48 – Khí tượng thật tại t+h (c_oracle) vs thiết lập chính (c)

| model   |   horizon |     c |   c_oracle |
|:--------|----------:|------:|-----------:|
| GRU     |         1 | 3.045 |      3.041 |
| GRU     |         6 | 6.251 |      5.998 |
| GRU     |        24 | 6.896 |      6.372 |
| LSTM    |         1 | 3.108 |      3.066 |
| LSTM    |         6 | 6.515 |      6.324 |
| LSTM    |        24 | 7.058 |      6.79  |

## Chuẩn bị G3-10 – So với bài [2] (TB tầm 1..24). Lưu ý: KHÁC tập test và cách chia nên chỉ tham khảo

| source                                              | model        | ablation   |   MAE_avg24 |   RMSE_avg24 |
|:----------------------------------------------------|:-------------|:-----------|------------:|-------------:|
| Nhóm (test 03–06/2022)                              | GRU          | a          |       6.234 |        9.55  |
| Nhóm (test 03–06/2022)                              | GRU          | c          |       6.391 |        9.73  |
| Nhóm (test 03–06/2022)                              | LSTM         | a          |       6.294 |        9.587 |
| Nhóm (test 03–06/2022)                              | LSTM         | c          |       6.648 |       10.087 |
| Nhóm (test 03–06/2022)                              | Seq2Seq      | c          |       6.649 |        9.976 |
| Nhóm (test 03–06/2022)                              | persistence  | -          |       8.123 |       12.521 |
| Bài [2] Table 5 (bộ con 3 tháng 2021, TB theo trạm) | CNN-LSTM     | chỉ PM2.5  |       6.65  |        8.86  |
| Bài [2] Table 5 (bộ con 3 tháng 2021, TB theo trạm) | Prophet      | chỉ PM2.5  |       5.89  |        8.47  |
| Bài [2] Table 5 (bộ con 3 tháng 2021, TB theo trạm) | SGDRegressor | chỉ PM2.5  |       5.75  |        7.74  |
| Bài [2] Table 5 (bộ con 3 tháng 2021, TB theo trạm) | XGBoost      | chỉ PM2.5  |       6.28  |        8.73  |
