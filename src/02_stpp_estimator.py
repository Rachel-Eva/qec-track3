"""
02_stpp_estimator.py -- In-Situ Spatiotemporal Pauli Process (STPP) Learner & Triage Engine.

Theory:
  Under Pauli twirling, arbitrary quantum noise projects onto a classical
  stochastic process over Pauli fault trajectories. We characterize this
  process from detection events d_{r,i} = s_{r,i} ⊕ s_{r-1,i} by computing
  the spatiotemporal covariance matrix C_{(r,i),(r',j)} and inverting it
  into three physical mechanisms:

    1. Spatial link errors  p_link(i, i+1)  — intra-round crosstalk
    2. Temporal memory      κ_i             — inter-round non-Markovian drift
    3. Measurement noise    p_meas(i)       — isolated readout errors

Pipeline:
  1. Load detection events from shared/syndrome_telemetry.json (or direct numpy)
  2. Compute spatiotemporal covariance matrix
  3. Invert to extract STPP parameters
  4. Bonferroni-corrected statistical triage
  5. Ground truth validation (Caught / Missed / Wrongly Blamed)
  6. Export results to shared/stpp_results.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stpp_config import (
    SHARED, FIGURES, DISTANCE, N_ANCILLAS, ROUNDS_DEFAULT,
    ALPHA_TRIAGE, N_BOOTSTRAP, P_PHYS_DEFAULT, STPP_PRIOR_PATH,
    SYNDROME_TELEMETRY_PATH, PLANTED_TRUTH_PATH, STPP_RESULTS_PATH,
    RAW_DIR,
)


class STPPEstimator:
    """
    In-Situ Spatiotemporal Pauli Process (STPP) Learner.

    Inverts detection-event correlations to separate:
      - Spatial link error rates  p_link[i]
      - Temporal memory kernels   κ[i]
      - Measurement readout noise p_meas[i]
    """

    def __init__(self, n_ancillas=N_ANCILLAS, rounds=ROUNDS_DEFAULT):
        self.n_anc = n_ancillas
        self.rounds = rounds

    def compute_detection_events(self, raw_syndromes):
        """
        Convert raw syndrome bits to detection events.

        Parameters
        ----------
        raw_syndromes : ndarray, shape (shots, rounds, n_anc), dtype uint8
            Binary syndrome measurements.

        Returns
        -------
        events : ndarray, shape (shots, rounds, n_anc), dtype uint8
            Detection events: d_{r,i} = s_{r,i} ⊕ s_{r-1,i}
        """
        prev = np.concatenate(
            [np.zeros_like(raw_syndromes[:, :1, :]), raw_syndromes[:, :-1, :]],
            axis=1,
        )
        return (raw_syndromes ^ prev).astype(np.uint8)

    def compute_covariance_matrix(self, events):
        """
        Compute the full spatiotemporal covariance matrix C.

        C_{(r,i),(r',j)} = <d_{r,i} · d_{r',j}> - <d_{r,i}> · <d_{r',j}>

        Parameters
        ----------
        events : ndarray, shape (shots, rounds, n_anc)

        Returns
        -------
        C : ndarray, shape (rounds*n_anc, rounds*n_anc)
        """
        shots, rounds, n = events.shape
        flat = events.reshape(shots, rounds * n).astype(np.float32)
        means = flat.mean(axis=0)
        cov = (flat.T @ flat) / shots - np.outer(means, means)
        return cov

    def fit_stpp_parameters(self, events):
        """
        Extract STPP parameters from detection events via correlator inversion.

        Parameters
        ----------
        events : ndarray, shape (shots, rounds, n_anc)
            Detection events (NOT raw syndromes).

        Returns
        -------
        params : dict with keys:
            p_links        : ndarray (n_anc - 1,) — spatial link error rates
            kappa_temporal : ndarray (n_anc,) — temporal memory coefficients
            p_meas         : ndarray (n_anc,) — measurement noise rates
            mean_d         : ndarray (n_anc,) — marginal detection rates
            covariance     : ndarray — full spatiotemporal covariance matrix
        """
        shots, rounds, n = events.shape

        # Skip round 0 (initialization transients)
        ev = events[:, 1:, :].astype(np.float64)
        eff_rounds = rounds - 1

        # 1. Marginal detection rates <d_{r,i}>
        mean_d = ev.mean(axis=(0, 1))  # shape: (n_anc,)

        # 2. Intra-round spatial pairwise: p_link via correlator inversion
        #    p_link(i,i+1) = 0.5 - 0.5 * sqrt( (1 - 4*C_{ii+1}) / (1 - 2<di> - 2<di+1> + 4<di·di+1>) )
        p_links = np.zeros(n - 1)
        for i in range(n - 1):
            prod = (ev[:, :, i] * ev[:, :, i + 1]).mean()
            cov = prod - mean_d[i] * mean_d[i + 1]
            denom = 1.0 - 2.0 * mean_d[i] - 2.0 * mean_d[i + 1] + 4.0 * prod
            denom = max(1e-6, denom)
            arg = max(0.0, 1.0 - (4.0 * cov) / denom)
            p_links[i] = 0.5 - 0.5 * np.sqrt(arg)

        # 3. Inter-round temporal memory: κ_i = C_{(r,i),(r-1,i)}
        kappa_temporal = np.zeros(n)
        for i in range(n):
            if eff_rounds > 1:
                t_prod = (ev[:, 1:, i] * ev[:, :-1, i]).mean()
                kappa_temporal[i] = t_prod - (mean_d[i] ** 2)
            else:
                kappa_temporal[i] = 0.0

        # 4. Isolated measurement noise: p_meas(i) = <d_i> - Σ p_link(i,j)
        p_meas = np.zeros(n)
        for i in range(n):
            adjacent_link_sum = 0.0
            if i > 0:
                adjacent_link_sum += p_links[i - 1]
            if i < n - 1:
                adjacent_link_sum += p_links[i]
            p_meas[i] = max(1e-4, mean_d[i] - adjacent_link_sum)

        # 5. Full covariance matrix for diagnostics
        cov_matrix = self.compute_covariance_matrix(events)

        return {
            "p_links": p_links,
            "kappa_temporal": kappa_temporal,
            "p_meas": p_meas,
            "mean_d": mean_d,
            "covariance": cov_matrix,
        }

    def triage_defects(self, stpp_params, p0_baseline, n_shots=None,
                       alpha=ALPHA_TRIAGE):
        """
        Statistical fault isolation with Bonferroni correction.

        For each spatial link, test H₀: p_link = p₀ against H₁: p_link > p₀.

        Parameters
        ----------
        stpp_params : dict from fit_stpp_parameters()
        p0_baseline : dict {f"link_{idx}": p0, ...} — baseline link error rates
        n_shots     : int — effective sample size for SE computation
        alpha       : float — family-wise significance level

        Returns
        -------
        result : dict with z_scores, critical_z, flagged_defects, severity
        """
        p_links = stpp_params["p_links"]
        n_tests = len(p_links)
        z_scores = np.zeros(n_tests)

        # Bonferroni-corrected threshold
        z_crit = norm.isf(alpha / n_tests)
        flagged = []
        severity = {}

        if n_shots is None:
            n_shots = 5000  # fallback

        for idx, p_obs in enumerate(p_links):
            p0 = p0_baseline.get(f"link_{idx}", P_PHYS_DEFAULT)
            se = np.sqrt(p0 * (1.0 - p0) / n_shots)
            se = max(se, 1e-6)
            z = (p_obs - p0) / se
            z_scores[idx] = z
            if z > z_crit:
                flagged.append(idx)
                severity[f"link_{idx}"] = {
                    "p_obs": float(p_obs),
                    "p0": float(p0),
                    "z_score": float(z),
                    "excess_ratio": float(p_obs / max(1e-6, p0)),
                }

        return {
            "z_scores": z_scores,
            "critical_z": float(z_crit),
            "flagged_defects": flagged,
            "severity": severity,
        }

    def bootstrap_uncertainty(self, events, n_boot=N_BOOTSTRAP, seed=42):
        """
        Bootstrap the STPP parameter estimates to get standard errors.

        Returns
        -------
        dict with keys like 'p_links_se', 'kappa_temporal_se', 'p_meas_se'
        """
        rng = np.random.default_rng(seed)
        shots = events.shape[0]
        all_params = []

        for _ in range(n_boot):
            idx = rng.integers(0, shots, shots)
            params = self.fit_stpp_parameters(events[idx])
            all_params.append(params)

        p_links_stack = np.array([p["p_links"] for p in all_params])
        kappa_stack = np.array([p["kappa_temporal"] for p in all_params])
        p_meas_stack = np.array([p["p_meas"] for p in all_params])

        return {
            "p_links_se": p_links_stack.std(axis=0),
            "kappa_temporal_se": kappa_stack.std(axis=0),
            "p_meas_se": p_meas_stack.std(axis=0),
        }


def validate_against_ground_truth(flagged_links, planted_truth, distance=DISTANCE):
    """
    Score the triage against the known planted defects.

    Parameters
    ----------
    flagged_links : list[int] — link indices flagged by triage
    planted_truth : dict — e.g. {"2-1": 10.0, "6-7": 5.0}

    Returns
    -------
    dict with caught, missed, wrongly_blamed, precision, recall
    """
    from qec_circuits import layout
    _, anc, cx_pairs = layout(distance)

    # Map planted defect (stim qubit IDs) to link indices
    # Link i corresponds to data qubit i and i+1, checked by ancilla i
    # In Stim: ancilla i = qubit 2i+1, data i = qubit 2i
    # CX pairs for ancilla i: (2i, 2i+1) and (2i+2, 2i+1)
    true_defect_links = set()
    for key_str in planted_truth:
        parts = key_str.split("-")
        q_data, q_anc = int(parts[0]), int(parts[1])
        # ancilla index = (q_anc - 1) / 2
        anc_idx = (q_anc - 1) // 2 if q_anc % 2 == 1 else q_anc // 2
        if 0 <= anc_idx < distance - 1:
            true_defect_links.add(anc_idx)

    flagged_set = set(flagged_links)
    caught = true_defect_links & flagged_set
    missed = true_defect_links - flagged_set
    wrongly_blamed = flagged_set - true_defect_links

    precision = len(caught) / max(1, len(flagged_set))
    recall = len(caught) / max(1, len(true_defect_links))

    return {
        "caught": sorted(caught),
        "missed": sorted(missed),
        "wrongly_blamed": sorted(wrongly_blamed),
        "precision": precision,
        "recall": recall,
        "n_planted": len(true_defect_links),
        "n_flagged": len(flagged_set),
    }


def load_detection_events_from_telemetry(run_id="baseline"):
    """
    Load detection events from the shared telemetry JSON + raw NPZ files.

    Returns
    -------
    events : ndarray (shots, rounds+1, n_anc)
    """
    # Load from compressed NPZ (preferred, more data)
    npz_path = RAW_DIR / f"{run_id}.npz"
    if npz_path.exists():
        z = np.load(npz_path)
        packed_dets = z["detection_events"]
        n_dets = int(z["num_detectors"])
        dets = np.unpackbits(packed_dets, axis=1)[:, :n_dets]
        n_anc = N_ANCILLAS
        rounds_plus_one = n_dets // n_anc
        return dets.reshape(-1, rounds_plus_one, n_anc)

    # Fallback: parse inline shots from JSON
    with open(SYNDROME_TELEMETRY_PATH) as f:
        telemetry = json.load(f)

    for run in telemetry["runs"]:
        if run["run_id"] == run_id:
            inline = run["inline_shots"]
            n_anc = N_ANCILLAS
            events_list = []
            for shot in inline:
                det_strs = shot["detection_events"]
                det_arr = np.array([[int(c) for c in s] for s in det_strs], dtype=np.uint8)
                events_list.append(det_arr)
            return np.array(events_list)

    raise ValueError(f"Run '{run_id}' not found in telemetry")


# ═════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 72)
    print(" STPP Estimator & Triage Engine")
    print("=" * 72)

    # ── Load STPP prior (from Kaggle step, or defaults) ──────────────
    if STPP_PRIOR_PATH.exists():
        with open(STPP_PRIOR_PATH) as f:
            prior = json.load(f)
        print(f"[STPP] Loaded prior from {STPP_PRIOR_PATH}")
        p0_baseline = prior.get("stpp_prior", {}).get("j_spatial", {})
    else:
        print("[STPP] No prior found, using uniform baseline.")
        p0_baseline = {f"link_{i}": P_PHYS_DEFAULT for i in range(N_ANCILLAS)}

    # ── Load planted truth for validation ────────────────────────────
    planted_truth = {}
    if PLANTED_TRUTH_PATH.exists():
        with open(PLANTED_TRUTH_PATH) as f:
            truth_data = json.load(f)

    estimator = STPPEstimator()
    all_results = {}

    # ── Process each run ─────────────────────────────────────────────
    run_ids = ["baseline"]
    if PLANTED_TRUTH_PATH.exists():
        run_ids += list(truth_data.get("runs", {}).keys())

    for run_id in run_ids:
        print(f"\n{'─' * 60}")
        print(f" Processing run: {run_id}")
        print(f"{'─' * 60}")

        try:
            events = load_detection_events_from_telemetry(run_id)
        except (ValueError, FileNotFoundError) as e:
            print(f"  [SKIP] {e}")
            continue

        print(f"  Detection events shape: {events.shape}")
        shots = events.shape[0]

        # Fit STPP parameters
        params = estimator.fit_stpp_parameters(events)
        print(f"  p_links:        {np.round(params['p_links'], 5)}")
        print(f"  kappa_temporal: {np.round(params['kappa_temporal'], 6)}")
        print(f"  p_meas:         {np.round(params['p_meas'], 5)}")

        # Bootstrap uncertainties
        uncertainties = estimator.bootstrap_uncertainty(events, n_boot=20)
        print(f"  p_links SE:     {np.round(uncertainties['p_links_se'], 5)}")

        # Triage
        triage = estimator.triage_defects(params, p0_baseline, n_shots=shots)
        print(f"  Z-scores:       {np.round(triage['z_scores'], 2)}")
        print(f"  Critical Z:     {triage['critical_z']:.2f}")
        print(f"  Flagged links:  {triage['flagged_defects']}")

        # Ground truth validation (for observed runs only)
        validation = None
        if run_id != "baseline" and PLANTED_TRUTH_PATH.exists():
            run_truth = truth_data.get("runs", {}).get(run_id, {}).get("defects", {})
            if run_truth:
                validation = validate_against_ground_truth(
                    triage["flagged_defects"], run_truth
                )
                print(f"  ✓ Caught:        {validation['caught']}")
                print(f"  ✗ Missed:        {validation['missed']}")
                print(f"  ✗ Wrongly blamed:{validation['wrongly_blamed']}")
                print(f"  Precision={validation['precision']:.2f}  Recall={validation['recall']:.2f}")

        all_results[run_id] = {
            "p_links": params["p_links"].tolist(),
            "kappa_temporal": params["kappa_temporal"].tolist(),
            "p_meas": params["p_meas"].tolist(),
            "mean_d": params["mean_d"].tolist(),
            "p_links_se": uncertainties["p_links_se"].tolist(),
            "z_scores": triage["z_scores"].tolist(),
            "critical_z": triage["critical_z"],
            "flagged_defects": triage["flagged_defects"],
            "severity": triage["severity"],
            "validation": validation,
            "shots": shots,
        }

    # ── Export results ───────────────────────────────────────────────
    with open(STPP_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[STPP] Results exported → {STPP_RESULTS_PATH}")

    # ── Quick diagnostic plot ────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    for run_id, res in all_results.items():
        axes[0].plot(res["p_links"], "o-", label=run_id)
        axes[1].plot(res["kappa_temporal"], "s-", label=run_id)
        axes[2].bar(
            np.arange(len(res["z_scores"])) + list(all_results.keys()).index(run_id) * 0.2,
            res["z_scores"],
            width=0.2,
            label=run_id,
        )

    axes[0].set_title("Spatial Link Errors p_link")
    axes[0].set_xlabel("Link index")
    axes[0].set_ylabel("Error rate")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Temporal Memory κ")
    axes[1].set_xlabel("Ancilla index")
    axes[1].set_ylabel("κ coefficient")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].axhline(
        all_results.get("baseline", {}).get("critical_z", 2.8),
        color="red", ls="--", label="Bonferroni threshold",
    )
    axes[2].set_title("Triage Z-Scores")
    axes[2].set_xlabel("Link index")
    axes[2].set_ylabel("Z-score")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.suptitle("STPP In-Situ Characterization", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "stpp_characterization.png")
    plt.close()
    print(f"[STPP] Diagnostic plot → {FIGURES / 'stpp_characterization.png'}")


if __name__ == "__main__":
    main()
