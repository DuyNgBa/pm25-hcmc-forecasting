import sys
from pathlib import Path

# Cho phép `import src...` khi chạy pytest từ gốc repo
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
