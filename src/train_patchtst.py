"""G2-54..G2-71 — Huấn luyện & đánh giá PatchTST cho mọi thí nghiệm của TV3.

Ví dụ (chạy từ gốc repo, trên Kaggle thêm --device cuda):
  # RQ1 / RQ2 — chỉ TP.HCM (S0)
  python -m src.train_patchtst --strategy S0 --ablation c --L 48                 # test chính
  python -m src.train_patchtst --strategy S0 --ablation c --L 48 --fold all      # 4 fold rolling-origin
  python -m src.train_patchtst --strategy S0 --ablation a --L 48 --seeds 42 43 44
  # Tối ưu siêu tham số (30 trial trên val) -> configs/patchtst_best_L48_c.yaml
  python -m src.train_patchtst --hpo --ablation c --L 48
  # RQ3 — dữ liệu Bắc Kinh
  python -m src.train_patchtst --strategy S1 --ablation c --L 48                 # zero-shot
  python -m src.train_patchtst --strategy S2_full --ablation c --L 48 --train-frac 0.25
  python -m src.train_patchtst --strategy S2_head --ablation c --L 48
  python -m src.train_patchtst --strategy S3 --ablation c --L 48
  # Thiết lập phụ: khí tượng thực tại t+h
  python -m src.train_patchtst --strategy S0 --ablation c_oracle --L 48

Mỗi lần chạy: ghi results/patchtst.csv (format chung) + results/preds/<run_id>__<eval_set>.parquet,
checkpoint tốt nhất ở checkpoints/. Dữ liệu Bắc Kinh KHÔNG bao giờ xuất hiện trong tập đánh giá.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from src.data.pipeline import prepare
from src.data.scaling import inverse_pm25
from src.data.windows import METEO_S, feature_columns
from src.eval.metrics import wide_to_long
from src.eval.results import append_results, make_run_id, results_rows, save_predictions
from src.models.patchtst import PatchTSTX, count_params
from src.utils import ROOT, load_config, path_of, results_file, seed_everything

CITY_IDX = {"hcm": 0, "beijing": 1}
CKPT_DIR = ROOT / "checkpoints"


# ============================================================ Dataset PyTorch
class TorchWindows(Dataset):
    """Bọc WindowSet -> tensor. Trả về (x, y, mask, x_future, city_idx)."""

    def __init__(self, ws, city: str, max_items: int | None = None, seed: int = 0):
        self.ws, self.city = ws, CITY_IDX[city]
        self.idx = np.arange(len(ws))
        if max_items and len(ws) > max_items:            # chỉ dùng cho --smoke
            self.idx = np.sort(np.random.default_rng(seed).choice(len(ws), max_items, replace=False))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        item = self.ws[self.idx[i]]
        x, y, m = item[:3]
        xf = item[3] if len(item) == 4 else np.zeros((self.ws.H, 0), np.float32)
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y.astype(np.float32)),
                torch.from_numpy(m), torch.from_numpy(np.ascontiguousarray(xf)), self.city)


def masked_mse(pred, y, m):
    m = m.float()
    return ((pred - y) ** 2 * m).sum() / m.sum().clamp(min=1.0)


# ============================================================ huấn luyện
def run_epoch(model, loader, device, opt=None, grad_clip=1.0):
    train = opt is not None
    model.train(train)
    tot, cnt = 0.0, 0.0
    with torch.set_grad_enabled(train):
        for x, y, m, xf, c in loader:
            x, y, m, xf, c = x.to(device), y.to(device), m.to(device), xf.to(device), c.to(device)
            pred = model(x, xf if model.n_future else None, c)
            loss = masked_mse(pred, y, m)
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
            k = m.float().sum().item()
            tot += loss.item() * k
            cnt += k
    return tot / max(cnt, 1)


def fit(model, train_loader, val_loader, device, lr, max_epochs, patience, weight_decay=1e-4,
        grad_clip=1.0, log=print):
    """Early stopping theo val loss; trả về model ở epoch tốt nhất + lịch sử."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(1, patience // 3))
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device, opt, grad_clip)
        va = run_epoch(model, val_loader, device)
        sched.step(va)
        hist.append({"epoch": ep, "train_loss": tr, "val_loss": va, "sec": round(time.time() - t0, 1)})
        log(f"  epoch {ep:3d} | train {tr:.4f} | val {va:.4f} | {time.time() - t0:.1f}s")
        if va < best - 1e-5:
            best, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                log(f"  early stop ở epoch {ep} (tốt nhất: {best:.4f})")
                break
    model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def predict_z(model, ds, device, batch_size=1024):
    model.eval()
    out = []
    for x, _, _, xf, c in DataLoader(ds, batch_size=batch_size, shuffle=False):
        out.append(model(x.to(device), xf.to(device) if model.n_future else None, c.to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, ds.ws.H))


# ============================================================ cấu hình
def load_hparams(L: int, ablation: str, override: dict | None = None) -> dict:
    pc = yaml.safe_load(open(ROOT / "configs" / "patchtst.yaml", encoding="utf-8"))
    hp = {**pc["model"], **pc["patch"][L], **pc["train"]}
    best = ROOT / "configs" / f"patchtst_best_L{L}_{ablation.replace('_oracle', '')}.yaml"
    if best.exists():
        hp.update(yaml.safe_load(open(best, encoding="utf-8")))
    hp.update(override or {})
    hp["_finetune"], hp["_joint"], hp["_hpo"], hp["_seeds"] = pc["finetune"], pc["joint"], pc["hpo"], pc["seeds"]
    return hp


def build_model(n_channels, L, hp, n_cities=0, n_future=0, H=24):
    keys = ["patch_len", "stride", "d_model", "n_heads", "n_layers", "d_ff", "dropout", "attn_dropout",
            "head_dropout", "head", "revin", "norm", "res_attention"]
    return PatchTSTX(n_channels, L, H, n_cities=n_cities, n_future=n_future, **{k: hp[k] for k in keys})


def loaders(train_ds, val_ds, hp, sampler=None, num_workers=0):
    tl = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=sampler is None, sampler=sampler,
                    num_workers=num_workers, drop_last=len(train_ds) > hp["batch_size"])
    vl = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=num_workers)
    return tl, vl


# ============================================================ một thí nghiệm
def experiment(strategy: str, ablation: str, L: int, seed: int, fold: int | None, train_frac: float,
               device: str, hp_override: dict | None = None, smoke: bool = False, write: bool = True,
               log=print) -> dict:
    cfg = load_config()
    H = cfg["windows"]["horizon_max"]
    hp = load_hparams(L, ablation, hp_override)
    if smoke:
        hp.update(max_epochs=2, patience=1, d_model=16, n_layers=1, d_ff=32)
        hp["_finetune"] = {**hp["_finetune"], "max_epochs": 2, "patience": 1}
        hp["_joint"] = {**hp["_joint"], "epoch_size": 2000}
    if strategy != "S0" and ablation not in ("a", "c", "c_oracle"):
        raise ValueError("S1/S2/S3 chỉ dùng lược đồ chung: ablation a, c hoặc c_oracle "
                         "(b: Bắc Kinh không có khí tượng gốc HealthyAir; d: trạm lân cận không cùng nghĩa)")
    seed_everything(seed)
    feats = feature_columns("c" if ablation == "c_oracle" else ablation)
    future = METEO_S if ablation == "c_oracle" else None
    cap = 3000 if smoke else None

    # ---- dữ liệu TP.HCM (luôn là nơi đánh giá) ----
    P = prepare("hcm", fold=fold, train_frac=train_frac, neighbors=(ablation == "d"), cfg=cfg)
    kw = dict(future_cols=future)
    ws = {s: P.windows(s, feats, L, H, **kw) for s in ("train", "val", "test")}
    ds = {s: TorchWindows(w, "hcm", cap, seed) for s, w in ws.items()}
    log(f"[{strategy}/{ablation}/L{L}/fold={fold}/frac={train_frac}/seed={seed}] "
        f"HCM windows: train={len(ds['train'])} val={len(ds['val'])} test={len(ds['test'])}, F={len(ws['train'].feature_cols)}")
    if len(ds["train"]) == 0 or len(ds["val"]) == 0:
        raise RuntimeError("Không có cửa sổ train/val — kiểm tra dữ liệu khí tượng (ablation c cần Open-Meteo).")

    n_future = len(future) if future else 0
    C = len(ws["train"].feature_cols)
    t0 = time.time()
    if strategy == "S0":
        model = build_model(C, L, hp, n_future=n_future, H=H).to(device)
        tl, vl = loaders(ds["train"], ds["val"], hp)
        model, hist = fit(model, tl, vl, device, hp["lr"], hp["max_epochs"], hp["patience"], hp["weight_decay"], log=log)
    else:
        Bj = prepare("beijing", cfg=cfg)
        bws = {s: Bj.windows(s, feats, L, H, **kw) for s in ("train", "val")}
        bds = {s: TorchWindows(w, "beijing", cap, seed) for s, w in bws.items()}
        log(f"  Beijing windows: train={len(bds['train'])} val={len(bds['val'])}")
        if strategy in ("S1", "S2_full", "S2_head"):
            model = pretrained_beijing(bds, C, L, hp, n_future, H, seed, ablation, device, smoke, log)
            hist = []
            if strategy != "S1":
                ft = hp["_finetune"]
                head_only = strategy == "S2_head"
                model.freeze_backbone(head_only)
                tl, vl = loaders(ds["train"], ds["val"], hp)
                lr = ft["lr_head"] if head_only else ft["lr_full"]
                model, hist = fit(model, tl, vl, device, lr, ft["max_epochs"], ft["patience"], hp["weight_decay"], log=log)
        elif strategy == "S3":
            jt = hp["_joint"]
            model = build_model(C, L, hp, n_cities=2, n_future=n_future, H=H).to(device)
            joint = ConcatDataset([bds["train"], ds["train"]])
            nb, nh = len(bds["train"]), len(ds["train"])
            w = np.r_[np.full(nb, (1 - jt["hcm_share"]) / nb), np.full(nh, jt["hcm_share"] / nh)]
            sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), jt["epoch_size"], replacement=True,
                                            generator=torch.Generator().manual_seed(seed))
            tl, vl = loaders(joint, ds["val"], hp, sampler=sampler)        # val = CHỈ TP.HCM
            model, hist = fit(model, tl, vl, device, hp["lr"], hp["max_epochs"], hp["patience"], hp["weight_decay"], log=log)
        else:
            raise ValueError(strategy)
    train_sec = time.time() - t0

    # ---- đánh giá trên TP.HCM ----
    eval_set = "test" if fold is None else f"fold{fold}"
    z = predict_z(model, ds["test"], device)
    wt = ds["test"].ws
    meta = wt.meta().iloc[ds["test"].idx].reset_index(drop=True)
    y_pred = inverse_pm25(z, meta["station_id"], P.params, "hcm")
    y_true = wt.y_true_ug()[ds["test"].idx]
    hp_pub = {k: v for k, v in hp.items() if not k.startswith("_")}
    run_cfg = dict(model="PatchTST", strategy=strategy, ablation=ablation, L=L, train_frac=train_frac, seed=seed,
                   data=cfg["data_version"], hp=hp_pub, smoke=smoke)
    run_id = make_run_id(**run_cfg)
    preds = wide_to_long(meta, y_true, y_pred, cfg["windows"]["eval_horizons"], model="PatchTST", config=run_id)
    rows = results_rows(preds, member="Bo bo", model="PatchTST", strategy=strategy, ablation=ablation,
                        train_frac=train_frac, L=L, seed=seed, eval_set=eval_set, run_id=run_id,
                        notes=f"params={count_params(model)}; train_sec={train_sec:.0f}; epochs={len(hist)}; "
                              f"data={cfg['data_version']}" + ("; SMOKE" if smoke else ""))
    if write:
        append_results(rows, results_file(cfg, "patchtst"))
        save_predictions(preds, path_of(cfg, "preds_dir"), run_id, eval_set)
        CKPT_DIR.mkdir(exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "run_cfg": run_cfg, "features": ws["train"].feature_cols},
                   CKPT_DIR / f"{run_id}__{eval_set}.pt")
        (CKPT_DIR / f"{run_id}__{eval_set}.history.json").write_text(json.dumps(hist, indent=1))
    allrow = rows[rows["station"] == "ALL"]
    log(allrow[["horizon", "n", "MAE", "RMSE", "R2", "MAE_high", "AQI_acc"]].to_string(index=False))
    return {"run_id": run_id, "rows": rows, "train_sec": train_sec, "model": model}


def pretrained_beijing(bds, C, L, hp, n_future, H, seed, ablation, device, smoke, log):
    """Pretrain trên Bắc Kinh (có cache theo cấu hình để S1, S2_full, S2_head dùng CHUNG một mô hình)."""
    key = json.dumps({k: v for k, v in hp.items() if not k.startswith("_")}, sort_keys=True) + \
        f"|C{C}|L{L}|F{n_future}|s{seed}|{ablation}|smoke{smoke}|{load_config()['data_version']}"
    path = CKPT_DIR / f"pretrain_beijing_{hashlib.md5(key.encode()).hexdigest()[:10]}.pt"
    model = build_model(C, L, hp, n_future=n_future, H=H).to(device)
    if path.exists():
        log(f"  dùng lại mô hình pretrain {path.name}")
        model.load_state_dict(torch.load(path, map_location=device))
        return model
    log("  pretrain trên Bắc Kinh ...")
    tl, vl = loaders(bds["train"], bds["val"], hp)
    model, _ = fit(model, tl, vl, device, hp["lr"], hp["max_epochs"], hp["patience"], hp["weight_decay"], log=log)
    CKPT_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path)
    return model


# ============================================================ G2-56: Optuna 30 trial trên val
def hpo(ablation: str, L: int, device: str, n_trials: int | None = None, smoke: bool = False):
    import optuna

    cfg = load_config()
    H = cfg["windows"]["horizon_max"]
    base = load_hparams(L, ablation)
    sp, hcfg = base["_hpo"]["space"], base["_hpo"]
    feats = feature_columns(ablation)
    P = prepare("hcm", neighbors=(ablation == "d"), cfg=cfg)
    cap = 3000 if smoke else None
    ds = {s: TorchWindows(P.windows(s, feats, L, H), "hcm", cap) for s in ("train", "val")}
    C = len(ds["train"].ws.feature_cols)
    y_val = ds["val"].ws.y_true_ug()[ds["val"].idx]
    meta_val = ds["val"].ws.meta().iloc[ds["val"].idx]

    def objective(trial):
        hp = dict(base)
        hp["d_model"] = trial.suggest_categorical("d_model", sp["d_model"])
        hp["n_heads"] = trial.suggest_categorical("n_heads", sp["n_heads"])
        hp["n_layers"] = trial.suggest_categorical("n_layers", sp["n_layers"])
        hp["d_ff"] = trial.suggest_categorical("d_ff", sp["d_ff"])
        hp["dropout"] = trial.suggest_float("dropout", *sp["dropout"])
        hp["head_dropout"] = trial.suggest_float("head_dropout", *sp["head_dropout"])
        hp["lr"] = trial.suggest_float("lr", *sp["lr"], log=True)
        hp["patch_len"] = trial.suggest_categorical("patch_len", [p for p in sp["patch_len"] if p <= L])
        hp["stride"] = max(1, hp["patch_len"] // 2)
        hp["batch_size"] = trial.suggest_categorical("batch_size", sp["batch_size"])
        if hp["d_model"] % hp["n_heads"]:
            raise optuna.TrialPruned()
        seed_everything(cfg["seed"])
        model = build_model(C, L, hp, H=H).to(device)
        tl, vl = loaders(ds["train"], ds["val"], hp)
        ep = 2 if smoke else hcfg["max_epochs"]
        model, _ = fit(model, tl, vl, device, hp["lr"], ep, hcfg["patience"], hp["weight_decay"], log=lambda *_: None)
        y_pred = inverse_pm25(predict_z(model, ds["val"], device), meta_val["station_id"], P.params, "hcm")
        ok = ~np.isnan(y_val)
        return float(np.abs(y_pred[ok] - y_val[ok]).mean())        # MAE µg/m³ TB trên 24 tầm (val)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
    study.optimize(objective, n_trials=n_trials or hcfg["n_trials"])
    best = dict(study.best_params)
    best["stride"] = max(1, best["patch_len"] // 2)
    out = ROOT / "configs" / f"patchtst_best_L{L}_{ablation}.yaml"
    if not smoke:
        yaml.safe_dump(best, open(out, "w", encoding="utf-8"), allow_unicode=True)
    study.trials_dataframe().to_csv(ROOT / "results" / f"hpo_patchtst_L{L}_{ablation}.csv", index=False)
    print(f"Tốt nhất: val MAE = {study.best_value:.3f} µg/m³ với {best}" + ("" if smoke else f" -> {out}"))
    return study


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", default="S0", choices=["S0", "S1", "S2_full", "S2_head", "S3"])
    ap.add_argument("--ablation", default="c", choices=["a", "b", "c", "d", "c_oracle"])
    ap.add_argument("--L", type=int, default=48, choices=[24, 48, 72])
    ap.add_argument("--fold", default="none", help="none | 1..4 | all")
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hpo", action="store_true")
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="chạy nhanh để kiểm tra code (KHÔNG dùng số liệu)")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    if a.hpo:
        hpo(a.ablation, a.L, a.device, a.n_trials, a.smoke)
        return
    folds = [None] if a.fold == "none" else ([1, 2, 3, 4] if a.fold == "all" else [int(a.fold)])
    for fold in folds:
        for seed in a.seeds:
            experiment(a.strategy, a.ablation, a.L, seed, fold, a.train_frac, a.device,
                       smoke=a.smoke, write=not a.no_write)


if __name__ == "__main__":
    main()
