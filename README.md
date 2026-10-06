# STPP-Triage

## In-Situ Characterization and Non-Markovian Decoding of Spatiotemporal Pauli Processes in Quantum Error Correction

**Qiskit Fall Fest 2026 — Track 3: Quantum Error Correction**

---

## The Problem

Standard QEC treats noise as **independent, identically distributed, and Markovian** — assuming faults occur independently across qubits and memorylessly across syndrome rounds. Real quantum hardware violates both assumptions: spectator crosstalk creates spatial correlations, while TLS fluctuations and quasiparticle poisoning generate non-Markovian temporal memory.

## Our Solution

**STPP-Triage** implements the first end-to-end framework that characterizes hardware as a **Spatiotemporal Pauli Process (STPP)** directly from error-correction telemetry:

1. **Physical Twirling** — Pauli twirling projects noise into a classical stochastic process
2. **Kaggle Stability Prior** — Baseline spatial/temporal noise estimates from calibration data
3. **Multi-Time Correlator Inversion** — Extract spatial crosstalk, temporal memory, and measurement noise from detection events
4. **Closed-Loop STPP Decoding** — Feed learned STPP into a non-Markovian PyMatching decoder
5. **Planted-Defect Validation** — End-to-end validation on controlled noise injection

---

## Repository Structure

```
qec-track3/
├── src/
│   ├── stpp_config.py           # Central configuration (all tunable params)
│   ├── 01_kaggle_stpp_prior.py  # Kaggle ingestion → STPP prior
│   ├── 02_stpp_estimator.py     # Core STPP learner & triage engine
│   ├── 03_stpp_digital_twin.py  # Chip visualization & diagnostics
│   ├── 04_task8_tradeoffs.py    # Pareto trade-off measurements
│   ├── noise_models_stpp.py     # Non-Markovian noise injection (Aer)
│   ├── decoder_stpp.py          # STPP-aware PyMatching decoder
│   ├── run_hardware_ibm.py      # IBM Quantum hardware execution
│   ├── run_full_pipeline.py     # Master orchestrator
│   │
│   ├── qec_circuits.py          # Repetition code circuits (Stim + Qiskit)
│   ├── noise_models.py          # Stim-based noise model
│   ├── decoder_compare.py       # Legacy Stim-DEM decoder comparison
│   ├── estimator.py             # Legacy fault-dictionary estimator
│   ├── sim.py                   # Aer-based simulation engine
│   ├── experiment.py            # Caught/missed experiment sweeps
│   ├── export_telemetry.py      # Telemetry data generation
│   ├── qiskit_parse.py          # Qiskit bitstring parsing
│   └── test_*.py                # Unit tests
│
├── shared/                      # Inter-module data exchange
│   ├── syndrome_telemetry.json  # Detection events (Person B → Person A)
│   ├── planted_truth.json       # Ground truth (scoring only)
│   ├── stpp_prior.json          # Kaggle-derived prior
│   ├── stpp_results.json        # STPP estimator output
│   ├── decoder_results.json     # Decoder benchmark results
│   └── telemetry_raw/           # Compressed NPZ per run
│
├── figures/                     # Generated plots
├── requirements.txt
├── schema.md                    # Data contract
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# For Kaggle prior (optional):
pip install xgboost scikit-learn pandas
```

### 2. Generate Simulation Data

```bash
cd src
python export_telemetry.py --shots 100000
```

### 3. Run the Full Pipeline

```bash
python run_full_pipeline.py --skip-kaggle
```

Or run stages individually:

```bash
python 02_stpp_estimator.py     # STPP estimation + triage
python decoder_stpp.py          # Decoder benchmark
python 03_stpp_digital_twin.py  # Visualization
python 04_task8_tradeoffs.py    # Trade-off curves
```

### 4. IBM Quantum Hardware (requires API token)

```bash
# Set your token:
export IBM_QUANTUM_TOKEN="your-token-here"    # Linux/Mac
$env:IBM_QUANTUM_TOKEN = "your-token-here"    # PowerShell

# Run on hardware:
python run_hardware_ibm.py

# Or test with Aer fallback:
python run_hardware_ibm.py --aer-fallback
```

---

## Where to Fill In (For Your Setup)

### 🔑 IBM Quantum Token
- **File:** `src/stpp_config.py` line ~55 → set `IBM_TOKEN`
- **Or:** environment variable `IBM_QUANTUM_TOKEN`
- **Or:** `python src/run_hardware_ibm.py --token "your-token"`

### 📊 Kaggle Dataset
- **File:** Place `circuit_stability.csv` in `shared/`
- **Then:** Edit column names in `src/01_kaggle_stpp_prior.py`:
  - `label_col` (line ~73) → your binary stability label column
  - `feature_cols` (line ~74) → auto-detected, but review
  - `timestamp_col` (line ~102) → your timestamp column
  - `p_z`, `p_x`, `t1`, `t2` column names (line ~117-127) → for noise bias η

### 🔬 Pauli Twirling (Optional Enhancement)
- **File:** `src/run_hardware_ibm.py` function `add_pauli_twirling()`
- Stub is provided with the full CX twirling group table
- Implement to get rigorous STPP projection on real hardware

### 🎯 Planted Defects
- **File:** `src/stpp_config.py` → `PLANTED_DEFECTS` dict
- Current defaults match the existing `shared/planted_truth.json`
- Modify to test different defect scenarios

### ⚙️ Backend Selection
- **File:** `src/stpp_config.py` line ~57 → `IBM_BACKEND_NAME`
- Default: `"ibm_brisbane"` — change to your available backend

---

## Mathematical Foundation

Under Pauli twirling, the noise channel is projected to a joint distribution over Pauli sequences:

```
P(P₁, P₂, ..., P_T) ∝ exp(-Σ h_i(t)σ_i(t) - Σ J_ij(t)σ_i(t)σ_j(t) - Σ K_ij(Δt)σ_i(t)σ_j(t'))
```

We implement a **Second-Order Truncated STPP** (r=1 spatial, m=1 temporal) and invert detection event correlators:

| Mechanism | Correlator | Physical Origin |
|-----------|-----------|-----------------|
| Spatial link error `p_link(i,i+1)` | `⟨d_{r,i} · d_{r,i+1}⟩` (intra-round) | CX gate / crosstalk |
| Temporal memory `κ_i` | `⟨d_{r,i} · d_{r-1,i}⟩` (inter-round) | TLS / readout latching |
| Measurement noise `p_meas(i)` | `⟨d_{r,i}⟩ - Σ p_link` (residual) | Readout error |

---

## Honest Limitations

1. **Truncation Boundary:** Second-order, nearest-neighbor STPP (r=1, m=1). Long-range and deep-memory effects are approximated.
2. **Pauli Projection Prerequisite:** Rigorous STPP requires multi-time Pauli twirling. Without it, coherent phase drifts introduce estimation bias.
3. **Unattended Error Classes:**
   - Computational subspace leakage (|2⟩) breaks stabilizer projection
   - Catastrophic cosmic-ray events exceed the code threshold

---

## License

MIT
