"""Điểm vào DUY NHẤT để lấy dữ liệu đã chuẩn hoá cho một thí nghiệm (dùng chung TV1, TV2, TV3).

    from src.data.pipeline import prepare
    P = prepare("hcm")                         # split chính, scaler fit trên train
    P = prepare("hcm", fold=3)                 # fold rolling-origin 3 (scaler FIT LẠI trên train fold 3)
    P = prepare("hcm", train_frac=0.25)        # train 25% (đoạn liên tục cuối)
    P = prepare("hcm", neighbors=True)         # thêm pm25_nb1..5 (ablation d)
    P.df, P.params, P.rng                      # bảng đã chuẩn hoá, tham số scaler, mốc chia

    ws_tr = P.windows("train", feature_columns("c"), L=48)
    ws_te = P.windows("test",  feature_columns("c"), L=48)

Thứ tự bên trong luôn là: đọc lược đồ -> xác định mốc chia -> fit scaler TRÊN TRAIN -> transform
-> (tuỳ chọn) thêm trạm lân cận. Nhờ gom vào một chỗ, không ai vô tình fit trên toàn bộ dữ liệu.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data.scaling import fit_scalers, load_params, save_params, transform
from src.data.splits import get_split_ranges, in_range
from src.data.windows import WindowSet, add_neighbor_pm25, origin_set, windows_for_split
from src.utils import load_config, path_of


@dataclass
class Prepared:
    city: str
    df: pd.DataFrame
    params: dict
    rng: dict
    cfg: dict
    _eval_cache: dict = field(default_factory=dict)

    def windows(self, split: str, feature_cols, L: int, H: int | None = None, common_eval: bool = True,
                **kw) -> WindowSet:
        """Cửa sổ cho một split.

        common_eval=True (mặc định) với val/test: chỉ giữ các gốc t thuộc TẬP ĐÁNH GIÁ CHUNG = các gốc hợp lệ
        với L_ref = 72 (PM2.5 đủ 72 giờ trước t, >= 1 nhãn đo thật). Nhờ vậy mọi mô hình, mọi L (24/48/72) và
        mọi ablation được chấm trên CÙNG một tập giờ (cột n trong results.csv phải bằng nhau).
        Train không bị giới hạn (L nhỏ được dùng thêm dữ liệu là hợp lệ).
        """
        H = H or self.cfg["windows"]["horizon_max"]
        kw.setdefault("nan_policy", self.cfg["windows"]["input_nan_policy"])
        kw.setdefault("min_valid_targets", self.cfg["windows"]["min_valid_targets"])
        if common_eval and split in ("val", "test"):
            kw["origin_whitelist"] = self.eval_origins(split, H)
        return windows_for_split(self.df, feature_cols, L, H, self.rng, split, **kw)

    def eval_origins(self, split: str, H: int) -> dict:
        key = (split, H)
        if key not in self._eval_cache:
            L_ref = self.cfg["windows"]["eval_reference_L"]
            ref = windows_for_split(self.df, ["pm25_z"], L_ref, H, self.rng, split, nan_policy="drop_target",
                                    min_valid_targets=self.cfg["windows"]["min_valid_targets"])
            self._eval_cache[key] = origin_set(ref)
        return self._eval_cache[key]


def load_schema(cfg: dict, city: str) -> pd.DataFrame:
    return pd.read_parquet(path_of(cfg, f"{city}_schema"))


def prepare(city: str, fold: int | None = None, train_frac: float = 1.0, neighbors: bool = False,
            cfg: dict | None = None, df: pd.DataFrame | None = None, save_scaler: bool = False) -> Prepared:
    cfg = cfg or load_config()
    df = load_schema(cfg, city) if df is None else df
    rng = get_split_ranges(cfg, city, fold=fold, train_frac=train_frac)
    train_mask = in_range(df["time"], rng["train"])
    params = fit_scalers(df, train_mask, cfg)
    params["fit_info"] = {"city": city, "fold": fold, "train_frac": train_frac,
                          "train_range": [str(x) for x in rng["train"]]}
    df_s = transform(df, params)
    if neighbors:
        df_s = add_neighbor_pm25(df_s, cfg["stations"][city], city=city)
    if save_scaler:  # scaler_params.json = {city: params của split chính} (G1-51)
        p = path_of(cfg, "scaler_params")
        allp = load_params(p) if p.exists() else {}
        allp[city] = params
        save_params(allp, p)
    return Prepared(city=city, df=df_s, params=params, rng=rng, cfg=cfg)


# ---------------------------------------------------------------------------
# Chạy trực tiếp:  python -m src.data.pipeline
# Minh hoạ G1-54 (cửa sổ trượt), G1-55 (trạm lân cận), G1-56 (lấy 50%/25% train) trên TP.HCM.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    from src.data.windows import feature_columns

    P = prepare("hcm")
    feats = feature_columns("a")
    print("=== G1-54: cửa sổ trượt (ablation a = PM2.5 quá khứ + giờ/thứ) ===")
    print("Đặc trưng:", feats)
    for L in (24, 48, 72):
        n = {s: len(P.windows(s, feats, L=L)) for s in ("train", "val", "test")}
        print(f"L={L:2d}: train={n['train']:6d}  val={n['val']:5d}  test={n['test']:5d}")
    print("-> test có CÙNG số cửa sổ với mọi L (tập đánh giá chung), train thì L nhỏ được nhiều hơn.")

    ws = P.windows("train", feats, L=48)
    x, y, m = ws[0]
    print(f"\nMột mẫu: x {x.shape} = 48 giờ × {x.shape[1]} đặc trưng, y {y.shape} = PM2.5 (z) của 24 giờ tới")
    meta = ws.meta().iloc[0]
    print(f"  trạm {meta.station_id}, gốc t = {meta.time_origin}  -> dự báo {meta.time_origin + pd.Timedelta(hours=1)} .. "
          f"{meta.time_origin + pd.Timedelta(hours=24)}")
    ok = ws.input_times().max() < P.rng["val"][0] and ws.target_times().max() <= P.rng["train"][1]
    print(f"  Kiểm tra rò rỉ: mọi thời điểm của cửa sổ train đều trước {P.rng['val'][0].date()} ->", "OK" if ok else "LỖI")

    print("\n=== G1-55: PM2.5 của 5 trạm gần nhất (ablation d) ===")
    Pn = prepare("hcm", neighbors=True, cfg=P.cfg)
    from src.data.windows import neighbor_order
    for s, nbs in neighbor_order(P.cfg["stations"]["hcm"]).items():
        print(f"  {s}: gần -> xa = {nbs}")
    cols = [f"pm25_nb{i}" for i in range(1, 6)]
    row = Pn.df[(Pn.df.station_id == "hcm_05") & Pn.df.pm25_z.notna()].iloc[1000]
    print(f"  Ví dụ hcm_05 lúc {row.time}: pm25_z = {row.pm25_z:.2f}, 5 trạm lân cận =",
          np.round(row[cols].to_numpy(float), 2))

    print("\n=== G1-56: lấy đoạn liên tục CUỐI của train ===")
    for frac in (1.0, 0.5, 0.25):
        Pf = prepare("hcm", train_frac=frac, cfg=P.cfg)
        s, e = Pf.rng["train"]
        print(f"  {int(frac * 100):3d}%: train {s.date()} -> {e.date()}, "
              f"{len(Pf.windows('train', feats, L=48)):6d} cửa sổ train (L=48)")
