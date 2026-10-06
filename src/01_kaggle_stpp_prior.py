"""
01_kaggle_stpp_prior.py -- Kaggle Quantum Circuit Stability: classical baseline + noise prior.

Columns (train.csv): id, num_qubits, circuit_depth, gate_error_rate,
                     decoherence_time, chip_temperature_mK, run_day, target
test.csv has the same columns without `target` (unlabeled, so it is only
used for a distribution-shift check, never for scoring).

What this script does (and does NOT do):
  1. Classical baseline: predict `target` from hardware features.
     Reports ROC-AUC and F1 on a stratified 80/20 held-out split, plus a
     time-based split (train on early run_day, test on late) because
     run_day is temporal and a random split can look better than reality.
  2. Noise prior: coarse numbers (typical gate error, drift trend with
     run_day, temperature correlation) written to shared/stpp_prior.json.
  3. It does NOT derive eta = p_Z/p_X: the dataset has one coherence time
     column, not separate T1/T2 or X/Z rates. eta is reported as unavailable.

The prior is context/sanity-check information. Triage thresholds in the
estimator come from the measured clean baseline run, not from this file.

Usage:  python 01_kaggle_stpp_prior.py [--train PATH] [--test PATH]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stpp_config import SHARED, FIGURES, N_ANCILLAS, P_PHYS_DEFAULT, STPP_PRIOR_PATH

FEATURES = ["num_qubits", "circuit_depth", "gate_error_rate",
            "decoherence_time", "chip_temperature_mK", "run_day"]
LABEL = "target"


def make_model():
    import xgboost as xgb
    # CPU, 100 trees, depth 5 (as in the spec sheet)
    return xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                             eval_metric="logloss", n_jobs=-1, random_state=42)


def evaluate(model, Xtr, ytr, Xte, yte):
    from sklearn.metrics import roc_auc_score, f1_score
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return float(roc_auc_score(yte, proba)), float(f1_score(yte, pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(SHARED / "train.csv"))
    ap.add_argument("--test", default=str(SHARED / "test.csv"))
    args = ap.parse_args()

    train = pd.read_csv(args.train)
    print(f"[Kaggle] train: {train.shape}, columns: {list(train.columns)}")
    missing = [c for c in FEATURES + [LABEL] if c not in train.columns]
    if missing:
        sys.exit(f"[Kaggle] Missing columns {missing}. Edit FEATURES/LABEL at top of file.")

    y = train[LABEL]
    if y.nunique() > 2:
        sys.exit(f"[Kaggle] '{LABEL}' has {y.nunique()} distinct values; this script "
                 f"expects a binary label. Tell me what the target means.")
    y = (y == sorted(y.unique())[-1]).astype(int).values  # larger value = positive class
    X = train[FEATURES].astype(float).fillna(train[FEATURES].median())
    print(f"[Kaggle] positive-class rate: {y.mean():.3f}")

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    # --- 1a. stratified 80/20 split (spec) -----------------------------
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = make_model()
    auc, f1 = evaluate(model, Xtr, ytr, Xte, yte)
    print(f"[Kaggle] XGBoost  80/20 split: ROC-AUC={auc:.4f}  F1={f1:.4f}")

    # simple reference model so the number has context
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    auc_lr, f1_lr = evaluate(lr, Xtr, ytr, Xte, yte)
    print(f"[Kaggle] LogReg   80/20 split: ROC-AUC={auc_lr:.4f}  F1={f1_lr:.4f}")

    # --- 1b. time-based split: train early days, test late days --------
    cut = train["run_day"].quantile(0.8)
    early, late = (train["run_day"] <= cut).values, (train["run_day"] > cut).values
    auc_t = f1_t = None
    if late.sum() > 20 and len(np.unique(y[late])) == 2 and len(np.unique(y[early])) == 2:
        auc_t, f1_t = evaluate(make_model(), X[early], y[early], X[late], y[late])
        print(f"[Kaggle] XGBoost time split (run_day > {cut:.0f}): ROC-AUC={auc_t:.4f}  F1={f1_t:.4f}")

    # --- 2. feature importances (model trained on the 80% split) -------
    importances = dict(zip(FEATURES, map(float, model.feature_importances_)))
    print("[Kaggle] importances:", {k: round(v, 3) for k, v in importances.items()})

    # --- 3. coarse noise prior -----------------------------------------
    g = train["gate_error_rate"].astype(float)
    p_gate = float(g.mean())
    by_day = train.groupby("run_day")["gate_error_rate"].mean()
    if len(by_day) >= 3:
        slope = float(np.polyfit(by_day.index.values.astype(float), by_day.values, 1)[0])
    else:
        slope = 0.0
    drift_rel_per_day = slope / p_gate if p_gate > 0 else 0.0
    temp_corr = float(train["chip_temperature_mK"].corr(g))
    print(f"[Kaggle] mean gate_error_rate={p_gate:.5f} | relative drift/day={drift_rel_per_day:+.4f} "
          f"| corr(temp, gate_error)={temp_corr:+.3f}")

    # Only use the Kaggle gate error as a per-link prior if it looks like a probability
    plausible = 1e-4 < p_gate < 0.1
    p_link_prior = p_gate if plausible else P_PHYS_DEFAULT
    if not plausible:
        print(f"[Kaggle] NOTE: mean gate_error_rate={p_gate} is not a plausible probability "
              f"(units?). Using config default {P_PHYS_DEFAULT} for the link prior.")

    # --- 4. optional: test.csv distribution shift ----------------------
    shift = None
    if Path(args.test).exists():
        test = pd.read_csv(args.test)
        shift = {c: float(test[c].mean() - train[c].mean()) for c in FEATURES if c in test.columns}
        print(f"[Kaggle] test.csv: {test.shape} (unlabeled; mean shift vs train computed)")

    payload = {
        "schema_version": 2,
        "source": "kaggle_circuit_stability",
        "classifier": {
            "model": "XGBClassifier(n_estimators=100, max_depth=5)",
            "split": "stratified 80/20, random_state=42",
            "roc_auc": auc, "f1": f1,
            "logreg_roc_auc": auc_lr, "logreg_f1": f1_lr,
            "time_split_roc_auc": auc_t, "time_split_f1": f1_t,
            "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
            "positive_rate": float(y.mean()),
        },
        "feature_importances": importances,
        "stpp_prior": {
            "j_spatial": {f"link_{i}": float(p_link_prior) for i in range(N_ANCILLAS)},
            "mean_gate_error_rate": p_gate,
            "gate_error_plausible_probability": bool(plausible),
            "drift_rel_per_day": drift_rel_per_day,
            "temp_gate_error_corr": temp_corr,
            "noise_bias_eta": None,
            "noise_bias_note": "Not derivable: dataset has one coherence-time column and no X/Z rates.",
        },
        "test_mean_shift": shift,
    }
    STPP_PRIOR_PATH.parent.mkdir(exist_ok=True)
    with open(STPP_PRIOR_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Kaggle] wrote {STPP_PRIOR_PATH}")

    # --- 5. plot ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = sorted(importances.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.barh([k for k, _ in items], [v for _, v in items], color="#4ECDC4")
    ax.set_title(f"Circuit stability: XGBoost importances (ROC-AUC {auc:.3f})")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "kaggle_feature_importances.png")
    print(f"[Kaggle] wrote {FIGURES / 'kaggle_feature_importances.png'}")


if __name__ == "__main__":
    main()