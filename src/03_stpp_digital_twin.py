"""
03_stpp_digital_twin.py -- STPP Digital Twin Visualization.

Renders a multi-dimensional visualization of the quantum chip:
  1. Chip graph: spatial links colored by estimated p_link, nodes by κ
  2. Spatiotemporal covariance heatmap
  3. Convergence plot: parameter certainty vs shot budget
  4. Triage summary: flagged defects with severity gauges
"""
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stpp_config import (
    SHARED, FIGURES, DISTANCE, N_ANCILLAS,
    STPP_RESULTS_PATH, PLANTED_TRUTH_PATH, RAW_DIR,
)


def plot_chip_graph(stpp_results, run_id, ax, distance=DISTANCE):
    """
    Draw the 1D repetition code as a chip graph:
      - Data qubits as large circles
      - Ancilla qubits as diamonds
      - Links colored by p_link (green=low → red=high)
      - Node border colored by κ (blue=no memory → purple=high memory)
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    res = stpp_results[run_id]
    p_links = np.array(res["p_links"])
    kappa = np.array(res["kappa_temporal"])
    n_anc = len(kappa)

    # Positions: data qubits at even x, ancillas at odd x
    data_x = [2 * i for i in range(distance)]
    anc_x = [2 * i + 1 for i in range(n_anc)]

    # Color normalization for links
    link_norm = mcolors.Normalize(vmin=0, vmax=max(0.05, p_links.max()))
    link_cmap = plt.cm.RdYlGn_r  # Green=good, Red=bad

    # Color normalization for kappa
    kappa_abs = np.abs(kappa)
    kappa_norm = mcolors.Normalize(vmin=0, vmax=max(0.01, kappa_abs.max()))
    kappa_cmap = plt.cm.cool

    # Draw links (edges)
    for i in range(n_anc):
        ax.plot([data_x[i], anc_x[i]], [0, 0], color="#cccccc", lw=2, zorder=1)
        ax.plot([anc_x[i], data_x[i + 1]], [0, 0], color="#cccccc", lw=2, zorder=1)
    for i in range(len(p_links)):
        color = link_cmap(link_norm(p_links[i]))
        lw = 2 + 8 * link_norm(p_links[i])
        ax.plot([anc_x[i], data_x[i + 1]], [0, 0], color=color, lw=lw, zorder=2)

    # Draw data qubits (circles)
    for i, x in enumerate(data_x):
        ax.scatter(x, 0, s=600, c="white", edgecolors="black", linewidths=2,
                   zorder=3, marker="o")
        ax.text(x, 0, f"D{i}", ha="center", va="center", fontsize=8,
                fontweight="bold", zorder=4)

    # Draw ancilla qubits (diamonds), colored by kappa
    for i, x in enumerate(anc_x):
        color = kappa_cmap(kappa_norm(kappa_abs[i]))
        ax.scatter(x, 0, s=500, c=[color], edgecolors="black", linewidths=2,
                   zorder=3, marker="D")
        ax.text(x, 0, f"A{i}", ha="center", va="center", fontsize=7,
                fontweight="bold", zorder=4)

    # Annotations: p_link values
    for i in range(len(p_links)):
        mid_x = (data_x[i + 1])
        ax.text(mid_x, -0.15, f"{p_links[i]:.4f}", ha="center", va="top",
                fontsize=7, color="darkred")

    # Annotations: kappa values below ancillas
    for i, x in enumerate(anc_x):
        ax.text(x, 0.15, f"κ={kappa[i]:.4f}", ha="center", va="bottom",
                fontsize=6, color="purple")

    # Flag defects
    flagged = res.get("flagged_defects", [])
    for idx in flagged:
        mid_x = (data_x[idx] + data_x[idx + 1]) / 2
        ax.annotate("⚠ DEFECT", (mid_x, -0.25), ha="center", fontsize=9,
                    color="red", fontweight="bold")

    ax.set_xlim(-1, 2 * distance)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect("equal")
    ax.set_title(f"Chip Graph — {run_id}", fontsize=11, fontweight="bold")
    ax.axis("off")


def plot_covariance_heatmap(stpp_results, run_id, ax):
    """
    Heatmap of the full spatiotemporal covariance matrix.
    Requires 'covariance' key from the estimator (run main with --save-cov).
    """
    import matplotlib.pyplot as plt

    # If we have raw covariance stored, use it; otherwise skip
    res = stpp_results[run_id]
    if "covariance" not in res:
        ax.text(0.5, 0.5, "Covariance matrix\nnot stored in results.\nRe-run estimator to include.",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title(f"Covariance — {run_id}")
        return

    cov = np.array(res["covariance"])
    im = ax.imshow(cov, cmap="RdBu_r", aspect="auto",
                   vmin=-np.abs(cov).max(), vmax=np.abs(cov).max())
    ax.set_title(f"Spatiotemporal Covariance — {run_id}", fontsize=10)
    ax.set_xlabel("Detector index (round × ancilla)")
    ax.set_ylabel("Detector index")
    plt.colorbar(im, ax=ax, shrink=0.8)


def plot_convergence(run_id, distance=DISTANCE, ax=None):
    """
    Show how STPP parameter estimates converge with increasing shot budget.
    Re-runs the estimator on subsets of the data.

    ▸▸▸ NOTE: This reads the raw NPZ file to sub-sample shots.        ◂◂◂
    """
    from _02_stpp_estimator_import import STPPEstimator
    import matplotlib.pyplot as plt

    npz_path = RAW_DIR / f"{run_id}.npz"
    if not npz_path.exists():
        if ax:
            ax.text(0.5, 0.5, f"Raw data not found:\n{npz_path.name}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Convergence (no data)")
        return

    z = np.load(npz_path)
    packed = z["detection_events"]
    n_dets = int(z["num_detectors"])
    dets = np.unpackbits(packed, axis=1)[:, :n_dets]
    n_anc = N_ANCILLAS
    rp1 = n_dets // n_anc
    events = dets.reshape(-1, rp1, n_anc)
    total_shots = events.shape[0]

    shot_budgets = sorted(set([
        200, 500, 1000, 2000, 5000, 10000, 20000, 50000, total_shots
    ]))
    shot_budgets = [s for s in shot_budgets if s <= total_shots]

    estimator = STPPEstimator(n_ancillas=n_anc, rounds=rp1)
    results_by_budget = []

    for n_shots in shot_budgets:
        sub = events[:n_shots]
        params = estimator.fit_stpp_parameters(sub)
        results_by_budget.append({
            "shots": n_shots,
            "p_links": params["p_links"].copy(),
            "kappa_temporal": params["kappa_temporal"].copy(),
        })

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    for link_idx in range(n_anc - 1):
        vals = [r["p_links"][link_idx] for r in results_by_budget]
        ax.plot(shot_budgets, vals, "o-", label=f"link_{link_idx}", markersize=4)

    ax.set_xscale("log")
    ax.set_xlabel("Shot Budget")
    ax.set_ylabel("Estimated p_link")
    ax.set_title(f"Parameter Convergence — {run_id}", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_triage_summary(stpp_results, ax):
    """Table-style summary of all runs' triage results."""
    import matplotlib.pyplot as plt

    rows = []
    for run_id, res in stpp_results.items():
        val = res.get("validation")
        row = [
            run_id,
            str(res.get("flagged_defects", [])),
            f"{len(res.get('flagged_defects', []))}",
        ]
        if val:
            row += [
                str(val["caught"]),
                str(val["missed"]),
                f"{val['precision']:.2f}",
                f"{val['recall']:.2f}",
            ]
        else:
            row += ["—", "—", "—", "—"]
        rows.append(row)

    col_labels = ["Run", "Flagged", "#Flag", "Caught", "Missed", "Prec", "Rec"]
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    ax.set_title("Triage Summary", fontsize=12, fontweight="bold", pad=20)


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not STPP_RESULTS_PATH.exists():
        print(f"[Digital Twin] ERROR: {STPP_RESULTS_PATH} not found.")
        print(f"[Digital Twin] Run 02_stpp_estimator.py first.")
        return

    with open(STPP_RESULTS_PATH) as f:
        stpp_results = json.load(f)

    run_ids = list(stpp_results.keys())
    n_runs = len(run_ids)

    # ── Figure 1: Chip graphs for each run ───────────────────────────
    fig1, axes1 = plt.subplots(n_runs, 1, figsize=(14, 3 * n_runs), dpi=150)
    if n_runs == 1:
        axes1 = [axes1]
    for i, run_id in enumerate(run_ids):
        plot_chip_graph(stpp_results, run_id, axes1[i])
    fig1.suptitle("STPP Digital Twin — Chip Error Maps", fontsize=14,
                  fontweight="bold", y=1.02)
    fig1.tight_layout()
    fig1.savefig(FIGURES / "digital_twin_chips.png", bbox_inches="tight")
    plt.close(fig1)
    print(f"[Digital Twin] Chip graphs → {FIGURES / 'digital_twin_chips.png'}")

    # ── Figure 2: Triage summary table ───────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(12, 2 + 0.5 * n_runs), dpi=150)
    plot_triage_summary(stpp_results, ax2)
    fig2.tight_layout()
    fig2.savefig(FIGURES / "digital_twin_triage.png", bbox_inches="tight")
    plt.close(fig2)
    print(f"[Digital Twin] Triage table → {FIGURES / 'digital_twin_triage.png'}")

    # ── Figure 3: Comparative bar charts ─────────────────────────────
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    x = np.arange(N_ANCILLAS - 1)
    width = 0.8 / max(1, n_runs)
    for i, run_id in enumerate(run_ids):
        p = stpp_results[run_id]["p_links"]
        axes3[0].bar(x + i * width, p, width, label=run_id)
    axes3[0].set_xlabel("Link index")
    axes3[0].set_ylabel("p_link")
    axes3[0].set_title("Spatial Link Errors Across Runs")
    axes3[0].legend()
    axes3[0].grid(alpha=0.3, axis="y")

    x2 = np.arange(N_ANCILLAS)
    for i, run_id in enumerate(run_ids):
        k = stpp_results[run_id]["kappa_temporal"]
        axes3[1].bar(x2 + i * width, k, width, label=run_id)
    axes3[1].set_xlabel("Ancilla index")
    axes3[1].set_ylabel("κ")
    axes3[1].set_title("Temporal Memory Across Runs")
    axes3[1].legend()
    axes3[1].grid(alpha=0.3, axis="y")

    fig3.suptitle("STPP Parameter Comparison", fontsize=14, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig(FIGURES / "digital_twin_comparison.png")
    plt.close(fig3)
    print(f"[Digital Twin] Comparison → {FIGURES / 'digital_twin_comparison.png'}")


if __name__ == "__main__":
    main()
