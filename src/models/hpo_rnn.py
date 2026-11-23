"""G2-33 — Tối ưu siêu tham số LSTM / GRU bằng Optuna: 30 lần thử trên tập VAL (ngân sách chung của nhóm, G2-01).

    python -m src.models.hpo_rnn --cell lstm          # 30 lần thử, ghi best.lstm vào configs/rnn.yaml
    python -m src.models.hpo_rnn --cell gru
    python -m src.models.hpo_rnn --cell lstm --trials 5   # thử ít hơn

Thiết lập tối ưu: ablation (c) (PM2.5 + Open-Meteo + thời gian), chiến lược S0, seed 42, chỉ train TP.HCM.
Mục tiêu: MAE (µg/m³) trên VAL, trung bình 24 tầm, chỉ trên giờ đo thật, tập gốc t chung (L_ref = 72) nên
mọi L được chấm trên cùng tập giờ. Tập TEST KHÔNG được dùng ở bước này.
Không gian tìm: configs/rnn.yaml -> hpo.space (L, hidden, layers, dropout, lr, batch_size).
Lịch sử từng lần thử: results/hpo/rnn_<cell>__<data_fp>_trials.csv (bản mới nhất chép ra rnn_<cell>_trials.csv).
Optuna lưu vào results/hpo/rnn_<cell>__<data_fp>__<hpo_fp>.db: chạy lại sẽ tiếp tục, nhưng CHỈ với cùng dữ liệu
(data_fp, xem src/models/provenance.py) và cùng mục hpo. Dữ liệu đổi -> nghiên cứu mới, không nối lần thử cũ.
Mục best.<cell> ghi kèm data_version, data_fp, hpo_fp để rnn_experiments từ chối dùng best của dữ liệu khác.
"""
from __future__ import annotations

import argparse

import numpy as np
import optuna
import yaml

from src.data.scaling import inverse_pm25
from src.models.rnn_data import FastWindows
from src.models.provenance import hpo_fingerprint
from src.models.rnn_experiments import Runner, Spec, file_lock, load_rnn_config
from src.models.train_rnn import TrainConfig, fit, predict_z
from src.utils import ROOT

OUT = ROOT / "results" / "hpo"


def val_mae(runner: Runner, spec: Spec, max_epochs: int, patience: int) -> tuple[float, dict]:
    P, ws_tr = runner.windows(spec, "hcm", "train")
    _, ws_va = runner.windows(spec, "hcm", "val")
    fw_tr, fw_va = FastWindows([ws_tr]), FastWindows([ws_va])
    model = runner.new_model(spec, fw_tr.n_features)
    tc = TrainConfig(lr=spec.hp["lr"], batch_size=int(spec.hp["batch_size"]), max_epochs=max_epochs,
                     patience=patience, verbose=False)
    info = fit(model, fw_tr, fw_va, tc, runner.device, None, seed=spec.seed)
    z = predict_z(model, fw_va, runner.device)
    y_pred = inverse_pm25(z, ws_va.meta()["station_id"].to_numpy(), P.params, "hcm")
    y_true = ws_va.y_true_ug()
    ok = ~np.isnan(y_true)
    return float(np.abs(y_pred - y_true)[ok].mean()), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=["lstm", "gru"])
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    rcfg = load_rnn_config()
    H = rcfg["hpo"]
    sp = H["space"]
    n_trials = a.trials or H["n_trials"]
    runner = Runner(a.device, verbose=False)
    OUT.mkdir(parents=True, exist_ok=True)
    fp, hfp = runner.prov["data_fp"], hpo_fingerprint(rcfg)
    tag = f"rnn_{a.cell}__{fp}"
    print(f"HPO {a.cell}: thiết bị {runner.device}; data_version={runner.cfg['data_version']}; data_fp={fp}; hpo_fp={hfp}")
    study = optuna.create_study(study_name=f"{tag}__{hfp}", direction="minimize",
                                storage=f"sqlite:///{(OUT / f'{tag}__{hfp}.db').as_posix()}", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=42))

    def objective(trial: optuna.Trial) -> float:
        hp = {"L": trial.suggest_categorical("L", sp["L"]),
              "hidden": trial.suggest_categorical("hidden", sp["hidden"]),
              "layers": trial.suggest_categorical("layers", sp["layers"]),
              "dropout": trial.suggest_float("dropout", *sp["dropout"]),
              "lr": trial.suggest_float("lr", *sp["lr"], log=True),
              "batch_size": trial.suggest_categorical("batch_size", sp["batch_size"])}
        mae, info = val_mae(runner, Spec(cell=a.cell, hp=hp), H["max_epochs"], H["patience"])
        trial.set_user_attr("best_epoch", info["best_epoch"])
        trial.set_user_attr("train_sec", info["train_sec"])
        print(f"[{a.cell} trial {trial.number:2d}] val MAE={mae:.3f}  {hp}  ({info['train_sec']}s)", flush=True)
        return mae

    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if done < n_trials:
        study.optimize(objective, n_trials=n_trials - done)
    trials = study.trials_dataframe().assign(data_version=runner.cfg["data_version"], data_fp=fp, hpo_fp=hfp,
                                             device=runner.device.type)
    trials.to_csv(OUT / f"{tag}_trials.csv", index=False)
    trials.to_csv(OUT / f"rnn_{a.cell}_trials.csv", index=False)

    best = dict(study.best_params)
    best["dropout"] = round(best["dropout"], 3)
    best["lr"] = float(f"{best['lr']:.2e}")
    path = ROOT / "configs" / "rnn.yaml"
    with file_lock(path):
        _write_best(path, a.cell, best, study, {"data_version": runner.cfg["data_version"], "data_fp": fp,
                                                 "meteo_model": runner.prov["meteo_model"], "hpo_fp": hfp,
                                                 "device": runner.device.type})
    print(f"Tốt nhất ({a.cell}): val MAE={study.best_value:.3f}  {best}  -> đã ghi configs/rnn.yaml")


def _write_best(path, cell, best, study, prov: dict | None = None):
    text = path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    cfg.setdefault("best", {}) or {}
    cfg["best"] = cfg["best"] or {}
    cfg["best"][cell] = {**best, "val_MAE": round(study.best_value, 4), "n_trials": len(study.trials), **(prov or {})}
    # chỉ thay khối "best:" để giữ nguyên chú thích của file
    block = yaml.safe_dump({"best": cfg["best"]}, allow_unicode=True, sort_keys=False).rstrip()
    start = text.index("\nbest:") + 1
    end = text.index("\nseeds:")
    path.write_text(text[:start] + block + "\n" + text[end:], encoding="utf-8")


if __name__ == "__main__":
    main()
