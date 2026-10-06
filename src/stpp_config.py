"""
stpp_config.py -- Central configuration for the STPP-Triage project.

All tunable parameters, file paths, and physical constants live here.
Every other module imports from this file instead of hardcoding values.
"""
import os
from pathlib import Path

# ── Project Root ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SHARED = ROOT / "shared"
FIGURES = ROOT / "figures"
RAW_DIR = SHARED / "telemetry_raw"

# Ensure output dirs exist
SHARED.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Repetition Code Parameters ───────────────────────────────────────
DISTANCE        = 5          # code distance (data qubits)
N_ANCILLAS      = DISTANCE - 1   # 4 ancillas for d=5
ROUNDS_DEFAULT  = 4          # syndrome extraction rounds
SHOTS_DEFAULT   = 100_000    # default shot budget

# ── Physical Noise Baseline ─────────────────────────────────────────
P_PHYS_DEFAULT  = 0.02       # baseline physical error rate
P_IDLE_DEFAULT  = 0.003      # data qubit idle bit-flip rate
P_CX_DEFAULT    = 0.005      # CX depolarizing rate
P_RO_DEFAULT    = 0.010      # readout error rate

# ── STPP Estimator Defaults ─────────────────────────────────────────
STPP_TEMPORAL_DEPTH  = 1     # m=1: memory depends only on (t, t-1)
STPP_SPATIAL_RADIUS  = 1     # r=1: nearest-neighbor coupling
ALPHA_TRIAGE         = 0.01  # Bonferroni significance level
N_BOOTSTRAP          = 40    # bootstrap resamples for SE

# ── Planted Defect Scenarios ─────────────────────────────────────────
# Keys are (data_qubit, ancilla_qubit) in Stim qubit IDs.
# Values are multiplier k on baseline CX error.
PLANTED_DEFECTS = {
    "clean":  {},
    "single": {(2, 1): 10.0},
    "double": {(2, 1): 10.0, (6, 7): 5.0},
    "triple": {(2, 1): 10.0, (6, 7): 5.0, (4, 3): 8.0},
}

# ── Non-Markovian Temporal Injection ─────────────────────────────────
# For sim-level temporal memory injection: probability that a measurement
# flip in round r-1 causes a correlated flip in round r.
TEMPORAL_MEMORY_INJECTION_P = 0.05   # κ injection strength

# ── Decoder Settings ────────────────────────────────────────────────
DECODER_WEIGHT_CLAMP_MIN = 1e-4
DECODER_WEIGHT_CLAMP_MAX = 0.45

# ── IBM Quantum Hardware ─────────────────────────────────────────────
# ▸▸▸ FILL YOUR IBM QUANTUM TOKEN HERE (or set env var) ◂◂◂
IBM_TOKEN = os.environ.get("IBM_QUANTUM_TOKEN", None)
IBM_BACKEND_NAME = "ibm_brisbane"     # or "ibm_sherbrooke", etc.
IBM_SHOTS_CALIBRATION = 1_000
IBM_SHOTS_DEFECT      = 3_000

# ── Kaggle Dataset ──────────────────────────────────────────────────
KAGGLE_CSV_PATH = SHARED / "circuit_stability.csv"

# ── Shared Data Contract Paths ───────────────────────────────────────
SYNDROME_TELEMETRY_PATH = SHARED / "syndrome_telemetry.json"
PLANTED_TRUTH_PATH      = SHARED / "planted_truth.json"
HARDWARE_PROFILE_PATH   = SHARED / "hardware_profile.json"
STPP_PRIOR_PATH         = SHARED / "stpp_prior.json"
STPP_RESULTS_PATH       = SHARED / "stpp_results.json"
DECODER_RESULTS_PATH    = SHARED / "decoder_results.json"
