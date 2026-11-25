#!/usr/bin/env bash
# Chạy TOÀN BỘ thí nghiệm PatchTST của TV3 (G2-55 -> G2-71) theo đúng thứ tự.
# Yêu cầu: đã có data/processed/{hcm,beijing}_schema.parquet với khí tượng Open-Meteo (data-v1.0).
# Kaggle: !bash scripts/run_patchtst_all.sh 2>&1 | tee results/run_patchtst.log
# Thời gian ước tính trên GPU T4: vài giờ; có thể chạy từng khối bằng cách comment phần còn lại.
set -euo pipefail
PY="python -m src.train_patchtst"
SEEDS="42 43 44"

# 0) G2-56 — tối ưu siêu tham số trên val (30 trial) cho cấu hình chính và cho ablation a
$PY --hpo --ablation c --L 48
$PY --hpo --ablation a --L 48

# 1) G2-55, G2-57 — cấu hình chính (S0, ablation c, L=48), 3 seed, test chính
$PY --strategy S0 --ablation c --L 48 --seeds $SEEDS
# 2) G2-58 — rolling-origin
$PY --strategy S0 --ablation c --L 48 --fold all --seeds $SEEDS

# 3) G2-59..62 — ablation RQ2 (cùng L=48)
for AB in a b d; do $PY --strategy S0 --ablation $AB --L 48 --seeds $SEEDS; done

# 4) G2-69 — độ dài cửa sổ đầu vào
for L in 24 72; do $PY --strategy S0 --ablation c --L $L --seeds $SEEDS; done

# 5) G2-63..68 — RQ3: chiến lược dùng dữ liệu Bắc Kinh × lượng dữ liệu TP.HCM
for FRAC in 1.0 0.5 0.25; do
  [ "$FRAC" != "1.0" ] && $PY --strategy S0 --ablation c --L 48 --train-frac $FRAC --seeds $SEEDS
  for ST in S2_full S2_head S3; do $PY --strategy $ST --ablation c --L 48 --train-frac $FRAC --seeds $SEEDS; done
done
$PY --strategy S1 --ablation c --L 48 --seeds $SEEDS

# 6) G2-70 (tuỳ chọn) — khí tượng thực tại t+h (giới hạn trên)
$PY --strategy S0 --ablation c_oracle --L 48 --seeds $SEEDS

# 7) Tổng hợp nhanh
python -m src.aggregate_results
