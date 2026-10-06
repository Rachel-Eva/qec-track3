"""Noise model: baseline error rate from the hardware profile + planted CX defects.

Link ids are Stim qubit ids written (data, ancilla), e.g. (2, 1) = d1 -> a0.
"""
import json
import os

from qec_circuits import stim_circuit, layout, logical_error_rate

DEFAULTS = {"p_phys_estimate": 0.01}


def load_profile(path="shared/hardware_profile.json"):
    """Read Person A's profile; fall back to defaults (and say so) if missing."""
    if os.path.exists(path):
        with open(path) as f:
            return {**DEFAULTS, **json.load(f)}
    print(f"[noise_models] {path} not found, using defaults {DEFAULTS}")
    return dict(DEFAULTS)


def make_circuit(distance=5, rounds=4, p=0.015, defects=None, link_p=None, drift_alpha=0.0):
    """Stim circuit with baseline noise p, optional planted defects, and temporal drift.

    defects: {(ctl, tgt): k} multiplies that link's CX error by k (planted truth).
    link_p:  {(ctl, tgt): prob} per-link overrides (used later for 'informed'
             weights built from Person A's estimates, not from the truth).
    drift_alpha: per-round multiplicative drift in error rates (α=0.08 per round).
    """
    _, _, pairs = layout(distance)
    cx_p = {pair: p for pair in pairs}
    if link_p:
        cx_p.update(link_p)
    for pair, k in (defects or {}).items():
        cx_p[pair] = min(cx_p[pair] * k, 0.75)
    return stim_circuit(distance, rounds, p_data=p, p_cx=p, p_meas=p, cx_p=cx_p, drift_alpha=drift_alpha)


if __name__ == "__main__":
    p = load_profile()["p_phys_estimate"]
    for k in [1, 2, 5, 10]:
        c = make_circuit(5, 4, p, defects={(2, 1): k}, drift_alpha=0.08)
        print(f"defect k={k}: logical error rate = {logical_error_rate(c):.5f}")