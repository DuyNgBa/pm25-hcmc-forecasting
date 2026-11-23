"""G2-29 → G2-49 — Chạy các thí nghiệm LSTM / GRU / Seq2Seq và ghi kết quả theo format chung của nhóm.

    python -m src.models.rnn_experiments --group smoke          # chạy thử nhanh (2 epoch) để kiểm tra
    python -m src.models.rnn_experiments --group rq1            # G2-29/30/32/34: LSTM, GRU × 3 seed, test
    python -m src.models.rnn_experiments --group rolling        # G2-35: 4 fold rolling-origin
    python -m src.models.rnn_experiments --group rq2            # G2-36..39: ablation a, b, d (c = rq1)
    python -m src.models.rnn_experiments --group seq2seq        # G2-31: Seq2Seq (LSTM, GRU) × 3 seed
    python -m src.models.rnn_experiments --group rq3            # G2-40..44: S1, S2_full, S2_head, S3
    python -m src.models.rnn_experiments --group curve          # G2-45: S0/S2_full/S3 × train 50%, 25%
    python -m src.models.rnn_experiments --group oracle         # G2-48: khí tượng thật tại t+h (c_oracle)
    python -m src.models.rnn_experiments --group all            # tất cả theo thứ tự trên
    ... --cells lstm            # chỉ một loại cell;  --force  chạy lại dù đã có kết quả;  --device cuda

Kết quả:
  results/rnn.csv                    một dòng = mô hình × cấu hình × tầm (1/6/24) × trạm/ALL × eval_set × seed
                                     (src/eval/results.py: results_rows + append_results — hàm CHUNG của nhóm)
  results/preds/<run_id>__<eval_set>.parquet   dự đoán theo format chung (G2-49)
  results/rnn_runs.csv               nhật ký từng lần chạy: số tham số, thời gian train, epoch tốt nhất, thiết bị
  checkpoints/                       trọng số tốt nhất (không commit)
Siêu tham số: configs/rnn.yaml (mục best.<cell> do src/models/hpo_rnn.py ghi, G2-33).
Đã chạy rồi (cùng run_id + eval_set có trong rnn_runs.csv) thì bỏ qua -> chạy lại lệnh sẽ tiếp tục chỗ dở.
run_id gồm cấu hình + data_version + DẤU VÂN TAY nội dung dữ liệu + cấu hình huấn luyện (src/models/provenance.py),
nên dữ liệu đổi (kể cả khi quên đổi data_version) thì kết quả, checkpoint pretrain cũ KHÔNG bị dùng lại.
Mục best trong configs/rnn.yaml phải được tối ưu trên đúng dữ liệu hiện tại, nếu không lệnh dừng
(bỏ qua kiểm tra bằng --allow-stale-hp, chỉ dùng để chạy thử).
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import os
import time
import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.pipeline import prepare
from src.data.scaling import inverse_pm25
from src.data.windows import METEO_S, feature_columns
from src.eval.metrics import wide_to_long
from src.eval.results import append_results, make_run_id, results_rows, save_predictions
from src.models.rnn import build_model, count_params
from src.models.provenance import check_best, code_revision, data_provenance, hpo_fingerprint, train_fingerprint
from src.models.rnn_data import FastWindows, balanced_weights
from src.models.train_rnn import TrainConfig, fit, pick_device, predict_z
from src.utils import ROOT, load_config, path_of, results_file

MEMBER = "Minh Trí"
HORIZONS = (1, 6, 24)
RUNS_LOG = ROOT / "results" / "rnn_runs.csv"
CKPT_DIR = ROOT / "checkpoints" / "rnn"


@contextlib.contextmanager
def file_lock(target: Path, timeout: float = 600):
    """Khoá đơn giản bằng file .lock để nhiều tiến trình (vd. LSTM và GRU chạy song song) không ghi đè nhau."""
    lock = Path(str(target) + ".lock")
    t0 = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.time() - t0 > timeout:
                raise TimeoutError(f"Không lấy được khoá {lock}")
            time.sleep(0.2)
    try:
        yield
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)


def load_rnn_config(path=ROOT / "configs" / "rnn.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def hp_for(rcfg: dict, cell: str) -> dict:
    hp = dict(rcfg["defaults"])
    best = (rcfg.get("best") or {}).get(cell) or {}
    hp.update({k: v for k, v in best.items() if k in hp})   # bỏ val_MAE, n_trials
    return hp


@dataclass
class Spec:
    arch: str = "direct"             # direct | seq2seq
    cell: str = "lstm"               # lstm | gru
    ablation: str = "c"              # a | b | c | d | c_oracle
    strategy: str = "S0"             # S0 | S1 | S2_full | S2_head | S3
    train_frac: float = 1.0
    fold: int | None = None          # None = test chính; 1..4 = rolling-origin
    seed: int = 42
    hp: dict = field(default_factory=dict)
    max_epochs: int | None = None    # ghi đè (chạy thử)

    @property
    def model_name(self) -> str:
        return {"direct": self.cell.upper(), "seq2seq": "Seq2Seq"}[self.arch]

    @property
    def eval_set(self) -> str:
        return "test" if self.fold is None else f"fold{self.fold}"

    @property
    def L(self) -> int:
        return int(self.hp["L"])

    @property
    def oracle(self) -> bool:
        return self.ablation == "c_oracle"

    def feats(self) -> list[str]:
        return feature_columns("c" if self.oracle else self.ablation)

    def run_id(self, data_key: str, residual: bool = True) -> str:
        """data_key = "<data_version>|<data_fp>|<train_fp>" (Runner.data_key)."""
        return make_run_id(model=self.model_name, residual=residual, arch=self.arch, cell=self.cell, ablation=self.ablation,
                           strategy=self.strategy, train_frac=self.train_frac, fold=self.fold, seed=self.seed,
                           hp=self.hp, max_epochs=self.max_epochs, data=data_key)


class Runner:
    def __init__(self, device="auto", verbose=True):
        self.cfg = load_config()
        self.rcfg = load_rnn_config()
        self.device = pick_device(device)
        self.verbose = verbose
        self._prep = {}
        self.prov = data_provenance(self.cfg)
        self.train_fp = train_fingerprint(self.rcfg)
        self.data_key = f"{self.cfg['data_version']}|{self.prov['data_fp']}|{self.train_fp}"
        self.code_rev = code_revision()

    # ---------------- dữ liệu ----------------
    def prep(self, city="hcm", fold=None, frac=1.0, neighbors=False):
        key = (city, fold, frac, neighbors)
        if key not in self._prep:
            self._prep[key] = prepare(city, fold=fold, train_frac=frac, neighbors=neighbors, cfg=self.cfg)
        return self._prep[key]

    def windows(self, spec: Spec, city: str, split: str):
        P = self.prep(city, spec.fold if city == "hcm" else None, spec.train_frac if city == "hcm" else 1.0,
                      neighbors=(spec.ablation == "d"))
        kw = {"future_cols": METEO_S} if spec.oracle else {}
        return P, P.windows(split, spec.feats(), L=spec.L, H=self.cfg["windows"]["horizon_max"], **kw)

    def train_config(self, spec: Spec, phase: str = "train", lr_mult: float = 1.0) -> TrainConfig:
        t = dict(self.rcfg[phase])
        if spec.max_epochs:
            t["max_epochs"] = spec.max_epochs
        return TrainConfig(lr=spec.hp["lr"] * lr_mult, batch_size=int(spec.hp["batch_size"]),
                           verbose=self.verbose, **t)

    def new_model(self, spec: Spec, n_features: int, fut_dim: int = 0):
        hp = spec.hp
        return build_model(spec.arch, n_features, self.cfg["windows"]["horizon_max"], spec.cell,
                           int(hp["hidden"]), int(hp["layers"]), float(hp["dropout"]), fut_dim,
                           residual=bool(self.rcfg.get("residual", True)))

    # ---------------- tiền huấn luyện Bắc Kinh (dùng lại giữa S1, S2, đường cong dữ liệu) ----------------
    def pretrained(self, spec: Spec):
        base = Spec(arch=spec.arch, cell=spec.cell, ablation="c", strategy="pretrain", seed=spec.seed,
                    hp=spec.hp, max_epochs=spec.max_epochs)
        path = CKPT_DIR / f"pretrain_beijing_{base.run_id(self.data_key)}.pt"
        _, ws_tr = self.windows(base, "beijing", "train")
        _, ws_va = self.windows(base, "beijing", "val")
        fw_tr, fw_va = FastWindows([ws_tr]), FastWindows([ws_va])
        model = self.new_model(base, fw_tr.n_features).to(self.device)
        if path.exists():
            ck = torch.load(path, map_location=self.device)
            model.load_state_dict(ck["state_dict"])
            info = ck["result"]
            print(f"  [pretrain] dùng lại {path.name}")
        else:
            print(f"  [pretrain] Bắc Kinh: {len(fw_tr)} cửa sổ train, {len(fw_va)} val")
            info = fit(model, fw_tr, fw_va, self.train_config(base, "pretrain"), self.device, path, seed=spec.seed)
        return model, info

    # ---------------- một lần chạy ----------------
    def run(self, spec: Spec, force: bool = False) -> pd.DataFrame | None:
        dv = self.cfg["data_version"]
        run_id = spec.run_id(self.data_key)
        if not force and RUNS_LOG.exists():
            log = pd.read_csv(RUNS_LOG)
            if ((log["run_id"] == run_id) & (log["eval_set"] == spec.eval_set)).any():
                print(f"[bỏ qua] {run_id} {spec.eval_set} đã có")
                return None
        print(f"\n=== {spec.model_name} cell={spec.cell} ab={spec.ablation} {spec.strategy} frac={spec.train_frac} "
              f"{spec.eval_set} seed={spec.seed} L={spec.L} -> {run_id}", flush=True)

        P, ws_tr = self.windows(spec, "hcm", "train")
        _, ws_va = self.windows(spec, "hcm", "val")
        _, ws_te = self.windows(spec, "hcm", "test")
        const = [1.0] if spec.strategy == "S3" else None
        fw_va, fw_te = FastWindows([ws_va], const), FastWindows([ws_te], const)
        ckpt = CKPT_DIR / f"{run_id}__{spec.eval_set}.pt"
        info, trainable = {}, None
        ft_mult = self.rcfg["finetune"]["lr_mult"]

        if spec.strategy == "S0":
            fw_tr = FastWindows([ws_tr])
            model = self.new_model(spec, fw_tr.n_features, fw_tr.fut_dim)
            info = fit(model, fw_tr, fw_va, self.train_config(spec), self.device, ckpt, seed=spec.seed)
        elif spec.strategy in ("S1", "S2_full", "S2_head"):
            model, pre = self.pretrained(spec)
            info = {"pretrain_epochs": pre.get("epochs_run"), "pretrain_sec": pre.get("train_sec")}
            if spec.strategy != "S1":
                fw_tr = FastWindows([ws_tr])
                model = copy.deepcopy(model)
                if spec.strategy == "S2_head":          # đóng băng encoder, chỉ train lớp cuối
                    for p in model.encoder.parameters():
                        p.requires_grad_(False)
                    trainable = model.head_parameters()
                ft = fit(model, fw_tr, fw_va, self.train_config(spec, "train", ft_mult), self.device, ckpt,
                         seed=spec.seed, params=trainable)
                info.update(ft)
        elif spec.strategy == "S3":
            base = Spec(**{**asdict(spec), "strategy": "pretrain", "ablation": "c", "train_frac": 1.0, "fold": None})
            _, bj_tr = self.windows(base, "beijing", "train")
            fw_tr = FastWindows([bj_tr, ws_tr], extra_const=[0.0, 1.0])
            share = self.rcfg["s3"]["hcm_share"]
            w = balanced_weights(fw_tr, [1 - share, share])
            model = self.new_model(spec, fw_tr.n_features)
            tc = self.train_config(spec)
            tc.epoch_samples = int(self.rcfg["s3"]["epoch_factor"] * len(ws_tr))
            info = fit(model, fw_tr, fw_va, tc, self.device, ckpt, seed=spec.seed, train_weights=w)
        else:
            raise ValueError(spec.strategy)

        # ---- đánh giá trên test (hoặc test của fold) ----
        z = predict_z(model, fw_te, self.device)
        meta = ws_te.meta()
        y_pred = inverse_pm25(z, meta["station_id"].to_numpy(), P.params, "hcm")
        y_true = ws_te.y_true_ug()
        preds = wide_to_long(meta, y_true, y_pred, HORIZONS, model=spec.model_name, config=run_id)
        n_par = count_params(model)
        notes = (f"arch={spec.arch}; cell={spec.cell}; hidden={spec.hp['hidden']}; layers={spec.hp['layers']}; "
                 f"dropout={spec.hp['dropout']}; lr={spec.hp['lr']}; params={n_par}; "
                 f"best_epoch={info.get('best_epoch', '-')}; train_sec={info.get('train_sec', 0)}; "
                 f"device={self.device.type}; data={dv}; data_fp={self.prov['data_fp']}")
        rows = results_rows(preds, member=MEMBER, model=spec.model_name, strategy=spec.strategy,
                            ablation=spec.ablation, train_frac=spec.train_frac, L=spec.L, seed=spec.seed,
                            eval_set=spec.eval_set, run_id=run_id, notes=notes)
        if spec.max_epochs is None:          # chạy thử (smoke) không ghi vào bảng kết quả chung
            with file_lock(results_file(self.cfg, "rnn")):
                append_results(rows, results_file(self.cfg, "rnn"))
            save_predictions(preds, path_of(self.cfg, "preds_dir"), run_id, spec.eval_set)
            # Thêm bản ĐỦ 24 tầm (đề xuất G0-07/G2_plan) để tính "TB tầm 1..24" so với bài [2]
            full = wide_to_long(meta, y_true, y_pred, range(1, ws_te.H + 1), model=spec.model_name, config=run_id)
            save_predictions(full, path_of(self.cfg, "preds_dir"), run_id, f"{spec.eval_set}_h24")
        allrow = rows[rows["station"] == "ALL"].set_index("horizon")
        log = {"run_id": run_id, "eval_set": spec.eval_set, "model": spec.model_name, **{k: v for k, v in asdict(spec).items() if k != "hp"},
               "L": spec.L, "hp": json.dumps(spec.hp), "params": n_par,
               "trainable_params": count_params(model, True) if trainable is None else sum(p.numel() for p in trainable),
               "best_epoch": info.get("best_epoch"), "epochs_run": info.get("epochs_run"),
               "train_sec": info.get("train_sec"), "pretrain_sec": info.get("pretrain_sec"),
               "best_val_loss": info.get("best_val_loss"), "n_train": len(ws_tr), "n_test": len(ws_te),
               "device": self.device.type, "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
               "data_version": dv, "data_fp": self.prov["data_fp"], "meteo_model": self.prov["meteo_model"],
               "raw_fp": self.prov["raw_fp"], "hcm_fp": self.prov["hcm_fp"], "beijing_fp": self.prov["beijing_fp"],
               "train_fp": self.train_fp, "code_rev": self.code_rev, "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "",
               **{f"MAE_h{h}": allrow.loc[h, "MAE"] for h in HORIZONS},
               **{f"RMSE_h{h}": allrow.loc[h, "RMSE"] for h in HORIZONS},
               **avg24(y_true, y_pred)}
        if spec.max_epochs is None:
            with file_lock(RUNS_LOG):
                old = pd.read_csv(RUNS_LOG) if RUNS_LOG.exists() else pd.DataFrame()
                if len(old):
                    old = old[~((old["run_id"] == run_id) & (old["eval_set"] == spec.eval_set))]
                pd.concat([old, pd.DataFrame([log])], ignore_index=True).to_csv(RUNS_LOG, index=False,
                                                                                float_format="%.4f")
        print(f"  -> MAE h1/h6/h24 = {log['MAE_h1']:.3f} / {log['MAE_h6']:.3f} / {log['MAE_h24']:.3f}  "
              f"(params {n_par}, {info.get('train_sec', 0)}s)", flush=True)
        return rows


def avg24(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE / RMSE gộp mọi tầm 1..24 và mọi trạm (cách tính "TB tầm 1..24" của bài [2], để tham chiếu G3-10)."""
    ok = ~np.isnan(y_true)
    e = (y_pred - y_true)[ok]
    return {"MAE_avg24": float(np.abs(e).mean()), "RMSE_avg24": float(np.sqrt((e ** 2).mean()))}


# ---------------- danh sách thí nghiệm ----------------
def specs_for(group: str, rcfg: dict, cells=("lstm", "gru")) -> list[Spec]:
    seeds = rcfg["seeds"]
    S = []
    for cell in cells:
        hp = hp_for(rcfg, cell)
        if group == "smoke":
            S += [Spec(cell=cell, hp=hp, max_epochs=2), Spec(arch="seq2seq", cell=cell, hp=hp, max_epochs=2)]
        elif group == "rq1":                                       # G2-29, G2-30, G2-32, G2-34
            S += [Spec(cell=cell, hp=hp, seed=s) for s in seeds]
        elif group == "rolling":                                   # G2-35
            S += [Spec(cell=cell, hp=hp, fold=k) for k in (1, 2, 3, 4)]
        elif group == "rq2":                                       # G2-36..39 (c = kết quả rq1)
            S += [Spec(cell=cell, hp=hp, ablation=ab, seed=s) for ab in ("a", "b", "d") for s in seeds]
        elif group == "seq2seq":                                   # G2-31, G2-47
            S += [Spec(arch="seq2seq", cell=cell, hp=hp, seed=s) for s in seeds]
        elif group == "rq3":                                       # G2-41..44 (S0 = rq1)
            S += [Spec(cell=cell, hp=hp, strategy=st) for st in ("S1", "S2_full", "S2_head", "S3")]
        elif group == "curve":                                     # G2-45 (100% = rq1 / rq3)
            S += [Spec(cell=cell, hp=hp, strategy=st, train_frac=fr) for fr in (0.5, 0.25)
                  for st in ("S0", "S2_full", "S3")]
        elif group == "oracle":                                    # G2-48
            S += [Spec(cell=cell, hp=hp, ablation="c_oracle")]
        else:
            raise ValueError(group)
    return S


GROUPS = ["rq1", "rolling", "rq2", "seq2seq", "rq3", "curve", "oracle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=GROUPS + ["all", "smoke"])
    ap.add_argument("--cells", nargs="+", default=["lstm", "gru"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--allow-stale-hp", action="store_true",
                    help="cho chạy dù best trong configs/rnn.yaml được tối ưu trên dữ liệu khác (chỉ để chạy thử)")
    a = ap.parse_args()
    r = Runner(a.device, verbose=not a.quiet)
    print(f"Thiết bị: {r.device}; data_version={r.cfg['data_version']}; data_fp={r.prov['data_fp']} "
          f"(khí tượng {r.prov['meteo_model']}); train_fp={r.train_fp}; code={r.code_rev}")
    if a.group != "smoke":
        errs = [e for c in a.cells if (e := check_best(r.rcfg, c, r.prov["data_fp"], hpo_fingerprint(r.rcfg)))]
        if errs and not a.allow_stale_hp:
            raise SystemExit("Dừng vì siêu tham số không khớp dữ liệu:\n  - " + "\n  - ".join(errs))
        for e in errs:
            print("[CẢNH BÁO] " + e)
    for g in (GROUPS if a.group == "all" else [a.group]):
        for spec in specs_for(g, r.rcfg, a.cells):
            r.run(spec, force=a.force)


if __name__ == "__main__":
    main()
