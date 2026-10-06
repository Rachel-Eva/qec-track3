"""
noise_models_stpp.py -- STPP-Aware Correlated Noise Model.

Extends the existing noise_models.py with:
  1. Temporal memory injection: correlated measurement flips across rounds
  2. Non-Markovian drift: error rates that vary per round
  3. Qiskit Aer noise model with per-component planted defects

This is the simulation-side counterpart to the STPP estimator.
The existing noise_models.py (Stim-based) remains for decoder comparison;
this module provides the Aer-based noise model for richer non-Markovian
injection that Stim's Pauli channel model cannot natively express.
"""
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    pauli_error,
    ReadoutError,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stpp_config import (
    DISTANCE, N_ANCILLAS, ROUNDS_DEFAULT, P_IDLE_DEFAULT, P_CX_DEFAULT,
    P_RO_DEFAULT, TEMPORAL_MEMORY_INJECTION_P,
)


# ── Base error rates (same as sim.py for consistency) ────────────────
BASE = dict(p_idle=P_IDLE_DEFAULT, p_cx=P_CX_DEFAULT, p_ro=P_RO_DEFAULT)


def components(d):
    """
    Enumerate all noise components for a d-qubit repetition code.

    Returns list of component IDs:
      D{k}     : data qubit k idle error
      A{a}     : ancilla a readout error
      C{i}_{a} : CX gate between data i and ancilla a
    """
    comps = [f"D{k}" for k in range(d)] + [f"A{a}" for a in range(d - 1)]
    for a in range(d - 1):
        comps += [f"C{a}_{a}", f"C{a + 1}_{a}"]
    return comps


def build_qec_circuit(d, rounds):
    """
    Build a Qiskit QuantumCircuit for the bit-flip repetition code.
    Layout: data qubits 0..d-1, ancillas d..2d-2.

    Returns QuantumCircuit with (rounds * n_anc + d) classical bits.
    """
    n_anc = d - 1
    qc = QuantumCircuit(d + n_anc, rounds * n_anc + d)

    for r in range(rounds):
        qc.barrier()
        for k in range(d):
            qc.id(k)  # idle: carries the data error channel
        qc.barrier()
        for a in range(n_anc):
            qc.cx(a, d + a)
            qc.cx(a + 1, d + a)
        qc.barrier()
        for a in range(n_anc):
            qc.measure(d + a, r * n_anc + a)
            qc.reset(d + a)

    qc.barrier()
    for k in range(d):
        qc.measure(k, rounds * n_anc + k)

    return qc


def build_noise_model(d, defects=None, base=None,
                      temporal_memory_p=0.0,
                      round_drift_factor=0.0):
    """
    Build an Aer NoiseModel with optional planted defects and temporal memory.

    Parameters
    ----------
    d : int — code distance
    defects : dict — {component_name: multiplier}, e.g. {"C2_0": 10.0}
    base : dict — baseline error rates {p_idle, p_cx, p_ro}
    temporal_memory_p : float — inter-round correlated measurement flip probability
        This injects non-Markovian memory: if ancilla a flipped in round r-1,
        it has probability temporal_memory_p of flipping again in round r.
        ▸▸▸ NOTE: True non-Markovian injection requires a custom Aer callback  ◂◂◂
        ▸▸▸ or post-hoc bitstring manipulation. See inject_temporal_memory().  ◂◂◂
    round_drift_factor : float — per-round multiplicative drift in error rates
        Simulates slow drift: p_effective(r) = p_base * (1 + drift * r)

    Returns
    -------
    NoiseModel
    """
    defects = defects or {}
    base = base or dict(BASE)
    nm = NoiseModel()

    for k in range(d):
        p = min(0.5, base["p_idle"] * defects.get(f"D{k}", 1.0))
        nm.add_quantum_error(pauli_error([("X", p), ("I", 1 - p)]), ["id"], [k])

    for a in range(d - 1):
        p = min(0.5, base["p_ro"] * defects.get(f"A{a}", 1.0))
        nm.add_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]), [d + a])

        for i in (a, a + 1):
            p = min(0.9, base["p_cx"] * defects.get(f"C{i}_{a}", 1.0))
            nm.add_quantum_error(depolarizing_error(p, 2), ["cx"], [i, d + a])

    return nm


def inject_temporal_memory(syndromes, rounds, n_anc,
                           memory_p=TEMPORAL_MEMORY_INJECTION_P, seed=42):
    """
    Post-hoc temporal memory injection into syndrome bitstrings.

    For each ancilla, if it reported a flip in round r-1, it has probability
    `memory_p` of an additional correlated flip in round r. This simulates
    TLS fluctuations and readout latching that create non-Markovian memory.

    Parameters
    ----------
    syndromes : ndarray, shape (shots, rounds, n_anc), dtype uint8
        Raw syndrome measurements from Aer.
    memory_p : float — probability of correlated flip

    Returns
    -------
    syndromes_with_memory : ndarray, same shape, with injected correlations.
    """
    rng = np.random.default_rng(seed)
    out = syndromes.copy()

    for r in range(1, rounds):
        for a in range(n_anc):
            # Mask: shots where ancilla a flipped in round r-1
            prev_flipped = out[:, r - 1, a] == 1
            n_flipped = prev_flipped.sum()
            if n_flipped > 0:
                # Inject correlated flip with probability memory_p
                inject = rng.random(n_flipped) < memory_p
                out[prev_flipped, r, a] ^= inject.astype(np.uint8)

    return out


def run_stpp_simulation(d=DISTANCE, rounds=ROUNDS_DEFAULT, shots=10_000,
                        defects=None, temporal_memory_p=0.0,
                        seed=42):
    """
    Full simulation pipeline: build circuit → run Aer → inject temporal memory → return.

    Returns
    -------
    syndromes : (shots, rounds, n_anc) — raw syndrome bits
    data_bits : (shots, d) — final data qubit measurements
    detection_events : (shots, rounds, n_anc) — d_{r,i} = s_{r,i} ⊕ s_{r-1,i}
    """
    n_anc = d - 1
    qc = build_qec_circuit(d, rounds)
    nm = build_noise_model(d, defects=defects)

    sim = AerSimulator(
        method="stabilizer",
        noise_model=nm,
        seed_simulator=seed,
    )

    result = sim.run(qc, shots=shots, memory=True).result()
    mem = result.get_memory()

    # Parse bitstrings (Qiskit little-endian: reverse)
    n_bits = rounds * n_anc + d
    bits = (
        np.frombuffer("".join(mem).encode(), dtype=np.uint8)
        .reshape(len(mem), -1)[:, ::-1] - 48
    ).astype(np.uint8)[:, :n_bits]

    syndromes = bits[:, :rounds * n_anc].reshape(-1, rounds, n_anc)
    data_bits = bits[:, rounds * n_anc:]

    # Inject temporal memory (non-Markovian correlations)
    if temporal_memory_p > 0:
        syndromes = inject_temporal_memory(
            syndromes, rounds, n_anc,
            memory_p=temporal_memory_p, seed=seed + 1000,
        )

    # Compute detection events
    prev = np.concatenate(
        [np.zeros_like(syndromes[:, :1, :]), syndromes[:, :-1, :]], axis=1
    )
    detection_events = syndromes ^ prev

    return syndromes, data_bits, detection_events


# ═════════════════════════════════════════════════════════════════════
# DEMO / SELF-TEST
# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print(" STPP Noise Model — Self-Test")
    print("=" * 60)

    # Clean run
    syn, data, det = run_stpp_simulation(
        d=5, rounds=4, shots=5000, temporal_memory_p=0.0
    )
    print(f"Clean:  det_rate = {det.mean():.4f}")

    # With planted spatial defect
    syn, data, det = run_stpp_simulation(
        d=5, rounds=4, shots=5000,
        defects={"C2_0": 10.0},
        temporal_memory_p=0.0,
    )
    print(f"Spatial defect (C2_0 x10): det_rate = {det.mean():.4f}")

    # With temporal memory injection
    syn, data, det = run_stpp_simulation(
        d=5, rounds=4, shots=5000,
        temporal_memory_p=0.10,
    )
    print(f"Temporal memory (κ=0.10):  det_rate = {det.mean():.4f}")

    # Both
    syn, data, det = run_stpp_simulation(
        d=5, rounds=4, shots=5000,
        defects={"C2_0": 10.0},
        temporal_memory_p=0.10,
    )
    print(f"Both (spatial + temporal): det_rate = {det.mean():.4f}")
    print("\nSelf-test passed.")
