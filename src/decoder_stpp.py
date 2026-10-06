"""
decoder_stpp.py -- Non-Markovian STPP-Aware PyMatching Decoder.

Builds a spacetime detector graph parameterized by the in-situ learned
STPP parameters (spatial link errors + temporal memory kernel).

Three decoder configurations are benchmarked:
  1. Uniform:     w = 1.0 for all edges (baseline)
  2. Static:      Spatial-only reweighting from p_link
  3. Full STPP:   Joint spatial + temporal memory reweighting
  4. Oracle:      True injected fault model (upper bound)

The existing decoder_compare.py (Stim DEM-based) is preserved for
comparison. This module builds the matching graph from scratch using
the STPP estimator output.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pymatching

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stpp_config import (
    SHARED, FIGURES, DISTANCE, N_ANCILLAS, ROUNDS_DEFAULT,
    P_PHYS_DEFAULT, DECODER_WEIGHT_CLAMP_MIN, DECODER_WEIGHT_CLAMP_MAX,
    STPP_RESULTS_PATH, DECODER_RESULTS_PATH, RAW_DIR, PLANTED_TRUTH_PATH,
)


def _clamp(p, lo=DECODER_WEIGHT_CLAMP_MIN, hi=DECODER_WEIGHT_CLAMP_MAX):
    """Clamp probability to avoid log(0) or negative weights."""
    return max(lo, min(hi, p))


def _prob_to_weight(p):
    """Convert error probability to MWPM matching weight: w = ln((1-p)/p)."""
    p = _clamp(p)
    return np.log((1.0 - p) / p)


# ═════════════════════════════════════════════════════════════════════
# GRAPH BUILDERS
# ═════════════════════════════════════════════════════════════════════

def build_uniform_matching(rounds, n_anc, p_uniform=0.01):
    """
    Uniform-weight matching graph. All edges have the same probability.
    Serves as the naive baseline decoder.
    """
    w = _prob_to_weight(p_uniform)
    n_nodes = rounds * n_anc
    matching = pymatching.Matching()

    for r in range(rounds):
        for i in range(n_anc):
            node = r * n_anc + i

            # Spatial edges (within same round)
            if i < n_anc - 1:
                matching.add_edge(node, node + 1, weight=w)

            # Temporal edges (between consecutive rounds)
            if r < rounds - 1:
                matching.add_edge(node, node + n_anc, weight=w)

            # Boundary edges (connect boundary nodes to virtual boundary)
            if i == 0 or i == n_anc - 1:
                matching.add_boundary_edge(node, weight=w)

    return matching


def build_static_matching(stpp_params, rounds, n_anc):
    """
    Static in-situ matching: spatial edges weighted by estimated p_link,
    temporal edges by p_meas only (no non-Markovian memory).
    """
    p_links = stpp_params["p_links"]
    p_meas = stpp_params["p_meas"]
    matching = pymatching.Matching()

    for r in range(rounds):
        for i in range(n_anc):
            node = r * n_anc + i

            # Spatial edges weighted by estimated p_link
            if i < n_anc - 1:
                w = _prob_to_weight(p_links[i])
                matching.add_edge(node, node + 1, weight=w)

            # Temporal edges weighted by p_meas (no kappa)
            if r < rounds - 1:
                w = _prob_to_weight(p_meas[i])
                matching.add_edge(node, node + n_anc, weight=w)

            # Boundary
            if i == 0:
                matching.add_boundary_edge(node, weight=_prob_to_weight(p_links[0]))
            elif i == n_anc - 1:
                matching.add_boundary_edge(node, weight=_prob_to_weight(p_links[-1]))

    return matching


def build_stpp_matching(stpp_params, rounds, n_anc):
    """
    Full STPP-aware matching: spatial edges weighted by p_link,
    temporal edges weighted by p_meas + κ (non-Markovian memory kernel).

    This is the key innovation: the temporal memory kernel modifies the
    effective error probability of measurement edges, allowing the decoder
    to account for correlated noise across rounds.
    """
    p_links = stpp_params["p_links"]
    p_meas = stpp_params["p_meas"]
    kappa = stpp_params["kappa_temporal"]
    matching = pymatching.Matching()

    for r in range(rounds):
        for i in range(n_anc):
            node = r * n_anc + i

            # 1. Spatial edges (data qubit / CNOT errors)
            if i < n_anc - 1:
                w = _prob_to_weight(p_links[i])
                matching.add_edge(node, node + 1, weight=w)

            # 2. Temporal edges (measurement + non-Markovian memory)
            if r < rounds - 1:
                # p_effective = p_meas + κ (memory kernel contribution)
                p_eff = p_meas[i] + kappa[i]
                w = _prob_to_weight(p_eff)
                matching.add_edge(node, node + n_anc, weight=w)

            # Boundary edges
            if i == 0:
                matching.add_boundary_edge(node, weight=_prob_to_weight(p_links[0]))
            elif i == n_anc - 1:
                matching.add_boundary_edge(node, weight=_prob_to_weight(p_links[-1]))

    return matching


def build_oracle_matching(distance, rounds, p_phys, defects):
    """
    Oracle decoder: builds matching from the TRUE detector error model.
    This is the ceiling — the best any decoder could do with perfect knowledge.
    """
    from noise_models import make_circuit

    defect_arg = {}
    for k, v in defects.items():
        if isinstance(k, str):
            parts = k.split("-")
            defect_arg[(int(parts[0]), int(parts[1]))] = v
        else:
            defect_arg[k] = v

    circ = make_circuit(distance, rounds, p_phys, defects=defect_arg or None)
    dem = circ.detector_error_model(
        decompose_errors=True, ignore_decomposition_failures=True
    )
    return pymatching.Matching.from_detector_error_model(dem)


# ═════════════════════════════════════════════════════════════════════
# DECODING
# ═════════════════════════════════════════════════════════════════════

def decode_batch(matching, detection_events):
    """
    Decode a batch of detection events.

    Parameters
    ----------
    matching : pymatching.Matching
    detection_events : ndarray, shape (shots, rounds, n_anc) or (shots, n_detectors)

    Returns
    -------
    predictions : ndarray
    """
    if detection_events.ndim == 3:
        shots, rounds, n_anc = detection_events.shape
        flat = detection_events.reshape(shots, -1)
    else:
        flat = detection_events

    return matching.decode_batch(flat.astype(np.uint8))


def benchmark_decoders(detection_events, observable_flips, stpp_params,
                       distance=DISTANCE, rounds=ROUNDS_DEFAULT,
                       p_phys=P_PHYS_DEFAULT, defects=None):
    """
    Run all four decoders on the same detection events and compare.

    Returns dict of {decoder_name: {ler, time_s, n_errors}}.
    """
    if detection_events.ndim == 3:
        shots, r, n = detection_events.shape
        flat = detection_events.reshape(shots, -1).astype(np.uint8)
        n_anc = n
        eff_rounds = r
    else:
        flat = detection_events.astype(np.uint8)
        shots = flat.shape[0]
        n_anc = N_ANCILLAS
        eff_rounds = flat.shape[1] // n_anc

    results = {}
    decoders = {
        "uniform": build_uniform_matching(eff_rounds, n_anc, p_phys),
        "static_insitu": build_static_matching(stpp_params, eff_rounds, n_anc),
        "stpp_full": build_stpp_matching(stpp_params, eff_rounds, n_anc),
    }

    # Oracle (if defects are known)
    if defects:
        try:
            decoders["oracle"] = build_oracle_matching(
                distance, rounds, p_phys, defects
            )
        except Exception as e:
            print(f"  [WARN] Oracle decoder failed: {e}")

    for name, matching in decoders.items():
        t0 = time.perf_counter()
        try:
            pred = matching.decode_batch(flat)
            elapsed = time.perf_counter() - t0

            if observable_flips is not None:
                obs = observable_flips
                if obs.ndim == 1:
                    obs = obs.reshape(-1, 1)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                # Trim to matching dimensions
                min_cols = min(pred.shape[1], obs.shape[1])
                n_errors = int(np.sum(np.any(pred[:, :min_cols] != obs[:, :min_cols], axis=1)))
                ler = n_errors / shots
            else:
                n_errors = -1
                ler = -1.0

            results[name] = {
                "logical_error_rate": float(ler),
                "n_errors": n_errors,
                "time_s": float(elapsed),
                "shots": shots,
            }
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"  [ERROR] Decoder '{name}': {e}")

    return results


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 72)
    print(" STPP-Aware Decoder Benchmark")
    print("=" * 72)

    # ── Load STPP parameters ─────────────────────────────────────────
    if not STPP_RESULTS_PATH.exists():
        print(f"[Decoder] ERROR: {STPP_RESULTS_PATH} not found.")
        print("[Decoder] Run 02_stpp_estimator.py first.")
        return

    with open(STPP_RESULTS_PATH) as f:
        stpp_results = json.load(f)

    # ── Load planted truth ───────────────────────────────────────────
    planted_runs = {}
    if PLANTED_TRUTH_PATH.exists():
        with open(PLANTED_TRUTH_PATH) as f:
            truth = json.load(f)
        planted_runs = truth.get("runs", {})

    # ── Process each run ─────────────────────────────────────────────
    all_decoder_results = {}

    for run_id in stpp_results:
        print(f"\n{'─' * 50}")
        print(f" Decoding run: {run_id}")
        print(f"{'─' * 50}")

        # Load detection events from raw NPZ
        npz_path = RAW_DIR / f"{run_id}.npz"
        if not npz_path.exists():
            print(f"  [SKIP] Raw data not found: {npz_path}")
            continue

        z = np.load(npz_path)
        packed_dets = z["detection_events"]
        n_dets = int(z["num_detectors"])
        dets = np.unpackbits(packed_dets, axis=1)[:, :n_dets]
        obs = z["observable_flips"]

        # Get STPP params for this run (use baseline params for informed decoding)
        # The decoder should use the LEARNED parameters, not the oracle truth
        stpp_params = {
            "p_links": np.array(stpp_results[run_id]["p_links"]),
            "kappa_temporal": np.array(stpp_results[run_id]["kappa_temporal"]),
            "p_meas": np.array(stpp_results[run_id]["p_meas"]),
        }

        # Get planted defects for oracle
        defects = planted_runs.get(run_id, {}).get("defects", {})

        n_anc = N_ANCILLAS
        eff_rounds = n_dets // n_anc
        det_3d = dets.reshape(-1, eff_rounds, n_anc)

        results = benchmark_decoders(
            det_3d, obs, stpp_params,
            distance=DISTANCE, rounds=eff_rounds - 1,
            p_phys=P_PHYS_DEFAULT, defects=defects or None,
        )

        for name, res in results.items():
            if "error" not in res:
                print(f"  {name:15s}: LER={res['logical_error_rate']:.5f}  "
                      f"({res['n_errors']}/{res['shots']})  "
                      f"time={res['time_s']:.3f}s")
            else:
                print(f"  {name:15s}: ERROR — {res['error']}")

        all_decoder_results[run_id] = results

    # ── Save results ─────────────────────────────────────────────────
    with open(DECODER_RESULTS_PATH, "w") as f:
        json.dump(all_decoder_results, f, indent=2)
    print(f"\n[Decoder] Results → {DECODER_RESULTS_PATH}")

    # ── Plot: grouped bar chart of LER across decoders and runs ──────
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    run_ids = list(all_decoder_results.keys())
    decoder_names = ["uniform", "static_insitu", "stpp_full", "oracle"]
    colors = {"uniform": "#E74C3C", "static_insitu": "#F39C12",
              "stpp_full": "#2ECC71", "oracle": "#3498DB"}

    x = np.arange(len(run_ids))
    width = 0.8 / len(decoder_names)

    for i, dec in enumerate(decoder_names):
        lers = []
        for run_id in run_ids:
            r = all_decoder_results.get(run_id, {}).get(dec, {})
            lers.append(r.get("logical_error_rate", 0))
        ax.bar(x + i * width, lers, width, label=dec, color=colors.get(dec, "gray"))

    ax.set_xlabel("Run")
    ax.set_ylabel("Logical Error Rate")
    ax.set_title("Decoder Comparison: Uniform vs Static vs STPP vs Oracle",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(run_ids, rotation=30, ha="right")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(FIGURES / "decoder_comparison_stpp.png")
    plt.close()
    print(f"[Decoder] Plot → {FIGURES / 'decoder_comparison_stpp.png'}")


if __name__ == "__main__":
    main()
