"""
01_kaggle_stpp_prior.py -- Kaggle Circuit Stability Ingestion & STPP Prior Generation.

Pipeline:
  1. Load the Kaggle circuit_stability.csv dataset.
  2. Train a classical XGBoost classifier to predict circuit instability (ROC-AUC).
  3. Extract temporal variance of physical features over time windows.
  4. Compute noise bias η = p_Z / p_X from calibration features.
  5. Export a STPP prior JSON with:
     - Baseline spatial coupling estimates J_spatial
     - Temporal memory decay kernel K_temporal
     - Noise bias η
     - Feature importances from XGBoost

▸▸▸ REQUIRES: Place `circuit_stability.csv` in shared/ ◂◂◂
▸▸▸ REQUIRES: pip install xgboost scikit-learn pandas        ◂◂◂
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stpp_config import (
    KAGGLE_CSV_PATH, STPP_PRIOR_PATH, SHARED, FIGURES,
    DISTANCE, N_ANCILLAS, P_PHYS_DEFAULT
)

warnings.filterwarnings("ignore", category=FutureWarning)


def load_dataset(csv_path):
    """
    Load the Kaggle Circuit Stability dataset.
    
    ▸▸▸ FILL: Adjust column names to match the actual CSV schema.    ◂◂◂
    ▸▸▸ Expected columns (adapt as needed):                          ◂◂◂
    ▸▸▸   - qubit_id, gate_type, error_rate, t1, t2, readout_error,  ◂◂◂
    ▸▸▸     cx_error, timestamp, stability_label (0/1)               ◂◂◂
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    print(f"[Kaggle] Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


def train_stability_classifier(df):
    """
    Train XGBoost binary classifier predicting circuit instability.
    
    Returns: (model, roc_auc, feature_importances_dict)
    
    ▸▸▸ FILL: Adjust feature_cols and label_col to match your CSV.   ◂◂◂
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb
    
    # ─── Placeholder column names: REPLACE with actual column names ───
    label_col = "stability_label"       # ◂◂◂ FILL: your binary label column
    feature_cols = [                     # ◂◂◂ FILL: your feature columns
        c for c in df.columns if c != label_col
    ]
    
    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    y = df[label_col].values
    
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    
    # 5-fold stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    print(f"[Kaggle] XGBoost 5-fold ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Fit on full data for feature importances
    clf.fit(X, y)
    importances = dict(zip(X.columns, clf.feature_importances_.tolist()))
    
    return clf, float(scores.mean()), importances


def extract_temporal_variance(df):
    """
    Compute the temporal variance of physical features across time windows.
    This captures how much the noise drifts over time → temporal memory kernel.
    
    Returns: dict of {feature_name: temporal_variance}
    
    ▸▸▸ FILL: Adjust timestamp_col and the features to track.        ◂◂◂
    """
    timestamp_col = "timestamp"   # ◂◂◂ FILL: your timestamp column
    
    # If no timestamp column, use row index as proxy
    if timestamp_col not in df.columns:
        print(f"[Kaggle] WARNING: No '{timestamp_col}' column found, using row index.")
        df = df.copy()
        df[timestamp_col] = np.arange(len(df))
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != timestamp_col]
    
    # Compute rolling variance in windows of ~100 rows
    window = min(100, len(df) // 5)
    temporal_var = {}
    for col in numeric_cols:
        rolling_std = df[col].rolling(window, min_periods=1).std()
        temporal_var[col] = float(rolling_std.mean())
    
    return temporal_var


def compute_noise_bias(df):
    """
    Estimate noise bias η = p_Z / p_X from calibration data.
    
    ▸▸▸ FILL: Adjust based on available error rate columns.          ◂◂◂
    ▸▸▸ If CSV has separate X/Z error rates, compute ratio directly. ◂◂◂
    ▸▸▸ Otherwise, estimate from T1/T2 relaxation times.             ◂◂◂
    """
    # Strategy 1: Direct ratio if columns exist
    if "p_z" in df.columns and "p_x" in df.columns:
        eta = (df["p_z"].mean()) / max(1e-6, df["p_x"].mean())
        return float(eta)
    
    # Strategy 2: Estimate from T1/T2
    # For depolarizing noise: η ≈ T1 / T2 (roughly)
    if "t1" in df.columns and "t2" in df.columns:
        t1_mean = df["t1"].mean()
        t2_mean = df["t2"].mean()
        eta = t1_mean / max(1e-6, t2_mean)
        return float(eta)
    
    # Fallback: assume balanced noise
    print("[Kaggle] WARNING: Cannot compute noise bias, defaulting to η=1.0 (balanced)")
    return 1.0


def build_stpp_prior(importances, temporal_var, eta):
    """
    Assemble the STPP prior from Kaggle-derived quantities.
    
    Maps feature importances → spatial coupling graph J_ij
    Maps temporal variance → memory decay kernel K(Δt)
    """
    # ── Spatial coupling prior: J_ij ─────────────────────────────────
    # Map CX-related feature importances to link coupling strengths
    # For d=5 repetition code: 4 ancillas → 8 CX links (each ancilla has 2 CNOTs)
    # Links: (d0,a0), (d1,a0), (d1,a1), (d2,a1), (d2,a2), (d3,a2), (d3,a3), (d4,a3)
    j_spatial = {}
    for i in range(N_ANCILLAS):
        # Use CX error importance as spatial coupling proxy
        cx_importance = max(
            importances.get("cx_error", 0.01),
            importances.get(f"cx_error_{i}", 0.01),
            P_PHYS_DEFAULT
        )
        j_spatial[f"link_{i}"] = float(cx_importance)
    
    # ── Temporal memory kernel: K(Δt) ────────────────────────────────
    # Aggregate temporal variance into a single decay constant
    # Higher variance → stronger non-Markovian memory
    if temporal_var:
        mean_var = np.mean(list(temporal_var.values()))
        # Normalize to [0, 1] range as a memory strength coefficient
        k_temporal = float(np.clip(mean_var / max(1e-6, max(temporal_var.values())), 0, 1))
    else:
        k_temporal = 0.0
    
    return {
        "j_spatial": j_spatial,
        "k_temporal": k_temporal,
        "noise_bias_eta": eta,
    }


def export_prior(prior, importances, roc_auc, output_path):
    """Write the STPP prior JSON for downstream consumption."""
    payload = {
        "schema_version": 1,
        "source": "kaggle_circuit_stability",
        "classifier_roc_auc": roc_auc,
        "feature_importances": importances,
        "stpp_prior": prior,
    }
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Kaggle] Exported STPP prior → {output_path}")


def plot_feature_importances(importances, output_dir):
    """Bar chart of top-20 feature importances."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    sorted_feats = sorted(importances.items(), key=lambda x: -x[1])[:20]
    names, vals = zip(*sorted_feats) if sorted_feats else ([], [])
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.barh(range(len(names)), vals, color="#4ECDC4")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Kaggle Circuit Stability — XGBoost Feature Importances")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_dir / "kaggle_feature_importances.png")
    plt.close()
    print(f"[Kaggle] Saved feature importances plot → {output_dir / 'kaggle_feature_importances.png'}")


# ═════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════
def main():
    csv_path = KAGGLE_CSV_PATH
    
    if not csv_path.exists():
        print(f"[Kaggle] ERROR: Dataset not found at {csv_path}")
        print(f"[Kaggle] Please place circuit_stability.csv in {SHARED}/")
        print(f"[Kaggle] Generating synthetic STPP prior with defaults...")
        
        # ── Fallback: generate a reasonable prior without Kaggle data ──
        prior = {
            "j_spatial": {f"link_{i}": P_PHYS_DEFAULT for i in range(N_ANCILLAS)},
            "k_temporal": 0.05,
            "noise_bias_eta": 1.0,
        }
        export_prior(prior, {}, 0.0, STPP_PRIOR_PATH)
        return
    
    # 1. Load
    df = load_dataset(csv_path)
    
    # 2. Train classifier
    clf, roc_auc, importances = train_stability_classifier(df)
    
    # 3. Extract temporal variance
    temporal_var = extract_temporal_variance(df)
    
    # 4. Compute noise bias
    eta = compute_noise_bias(df)
    print(f"[Kaggle] Noise bias η = {eta:.3f}")
    
    # 5. Build STPP prior
    prior = build_stpp_prior(importances, temporal_var, eta)
    
    # 6. Export
    export_prior(prior, importances, roc_auc, STPP_PRIOR_PATH)
    
    # 7. Plot
    plot_feature_importances(importances, FIGURES)


if __name__ == "__main__":
    main()
