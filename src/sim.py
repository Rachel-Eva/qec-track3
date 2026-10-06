"""Repetition-code memory experiment on Aer's stabilizer method with Pauli-only noise
and per-component defect injection. Bit-flip code, data prepared in |0...0>.

Layout: data qubits 0..d-1, ancillas d..2d-2. Ancilla a checks data a and a+1.
Components: D{k} (data idle bit-flip), A{a} (ancilla readout flip),
            C{i}_{a} (CX between data i and ancilla a, 2q depolarizing).
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error, ReadoutError

BASE = dict(p_idle=0.003, p_cx=0.005, p_ro=0.010)


def components(d):
    comps = [f"D{k}" for k in range(d)] + [f"A{a}" for a in range(d - 1)]
    for a in range(d - 1):
        comps += [f"C{a}_{a}", f"C{a + 1}_{a}"]
    return comps


def build_circuit(d, rounds):
    n_anc = d - 1
    qc = QuantumCircuit(d + n_anc, rounds * n_anc + d)
    for r in range(rounds):
        qc.barrier()
        for k in range(d):
            qc.id(k)                       # carries the data idle error
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


def noise_model(d, defects=None, base=BASE):
    """defects: dict component -> multiplier on its baseline error."""
    defects = defects or {}
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


def run_detection_events(d, rounds, shots, defects=None, seed=0, base=BASE):
    """Returns detection events, shape (shots, rounds+1, d-1), dtype uint8, and final data bits."""
    n_anc = d - 1
    sim = AerSimulator(method="stabilizer", noise_model=noise_model(d, defects, base),
                       seed_simulator=seed)
    qc = build_circuit(d, rounds)
    res = sim.run(qc, shots=shots, memory=True).result()
    mem = res.get_memory()
    shots_n = len(mem)
    # memory strings are MSB-first; reverse so column i == classical bit i
    bits = (np.frombuffer("".join(mem).encode(), dtype=np.uint8).reshape(shots_n, -1)[:, ::-1] - 48)
    syn = bits[:, : rounds * n_anc].reshape(shots, rounds, n_anc)
    data = bits[:, rounds * n_anc:]
    final_syn = data[:, :-1] ^ data[:, 1:]
    syn_all = np.concatenate([syn, final_syn[:, None, :]], axis=1)
    prev = np.concatenate([np.zeros_like(syn_all[:, :1]), syn_all[:, :-1]], axis=1)
    return syn_all ^ prev, data
