"""Sinh notebooks/bobo/02_beijing_prep.ipynb (G1-45, G1-46) và chạy thử."""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# 02 — Chuẩn bị dữ liệu Bắc Kinh (G1-45, G1-46)\n\n"
       "Notebook này **không viết lại logic**: mọi bước gọi hàm trong `src/data/beijing.py` "
       "(code dùng chung của nhóm). Mục đích là xem từng bước và kết quả.\n\n"
       "Yêu cầu: đã chạy `python -m src.data.download_beijing` (có 12 file trong `data/raw/beijing/`)."),
    code("import os, sys\n"
         "while not os.path.exists('configs/data.yaml') and os.path.dirname(os.getcwd()) != os.getcwd():\n"
         "    os.chdir('..')                      # đưa thư mục làm việc về gốc repo\n"
         "sys.path.insert(0, os.getcwd())\n"
         "import pandas as pd\n"
         "from src.utils import load_config, path_of, ROOT\n"
         "from src.data.beijing import load_raw, clean\n"
         "from src.data.cleaning_common import missing_table\n"
         "cfg = load_config()"),
    md("## Bước 1 — G1-45: gộp 12 file thành một bảng\n\n"
       "- 4 cột `year, month, day, hour` → 1 cột `time` (giờ địa phương Bắc Kinh)\n"
       "- Tên cột → snake_case (`PM2.5` → `pm25`); khí tượng gốc thêm đuôi `_native`\n"
       "- Khoá duy nhất: `(city, station_id, time)`; hàm tự kiểm tra không trùng, không thiếu giờ"),
    code("raw = load_raw(ROOT / cfg['paths']['raw_beijing'])\n"
         "print(raw.shape)\n"
         "raw.head()"),
    code("# Mỗi trạm phải có đúng 35.064 giờ, từ 01/03/2013 00:00 đến 28/02/2017 23:00\n"
         "raw.groupby('station_id')['time'].agg(['min', 'max', 'count'])"),
    md("## Bước 2 — Thiếu dữ liệu TRƯỚC khi làm sạch (% giờ thiếu theo trạm)"),
    code("cols = ['pm25', 'temp_native', 'pres_native', 'dewp_native', 'rain_native', 'wspm_native']\n"
         "missing_table(raw, cols)"),
    md("## Bước 3 — G1-46: làm sạch theo CÙNG quy tắc với HealthyAir\n\n"
       "1. PM2.5 ≤ 0 → thiếu\n"
       "2. **Không cắt** outlier PM2.5 (giống TV1): giá trị cao là sự kiện thật cần dự báo\n"
       "3. Nội suy tuyến tính khoảng trống **≤ 3 giờ** trong từng trạm, đánh dấu `is_imputed = 1`; "
       "khoảng trống dài hơn giữ nguyên là thiếu"),
    code("bj, log = clean(raw, cfg)\n"
         "log"),
    code("# Ví dụ một đoạn được nội suy: giá trị nối thẳng giữa 2 đầu, cờ is_imputed = 1\n"
         "i = bj.index[bj['is_imputed'] == 1][0]\n"
         "bj.loc[i - 2:i + 3, ['station_id', 'time', 'pm25', 'is_imputed']]"),
    md("## Bước 4 — So sánh % thiếu PM2.5 trước / sau"),
    code("cmp = pd.DataFrame({'truoc_%': missing_table(raw, ['pm25'])['pm25'],\n"
         "                    'sau_%': missing_table(bj, ['pm25'])['pm25']})\n"
         "cmp"),
    md("## Bước 5 — Lưu kết quả\n\n"
       "Hai file `.parquet` nằm trong `data/interim/` (không commit, ai cần thì chạy lại). "
       "Cách nhanh hơn để làm cả 2 bước: `python -m src.data.beijing`."),
    code("raw.to_parquet(path_of(cfg, 'beijing_raw_parquet'), index=False)\n"
         "bj.to_parquet(path_of(cfg, 'beijing_clean_parquet'), index=False)\n"
         "print('Đã lưu beijing_raw.parquet và beijing_clean.parquet')"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = ROOT / "notebooks" / "bobo" / "02_beijing_prep.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(out.parent)}}).execute()
nbf.write(nb, out)
print("Đã ghi", out)
