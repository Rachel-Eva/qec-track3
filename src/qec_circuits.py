"""Distance-d bit-flip repetition code: Qiskit (hardware/Aer) and Stim (decoding).

Line layout: d0 - a0 - d1 - a1 - ... - d_{n-1}
Stim qubit ids: data i -> 2i, ancilla i -> 2i+1 (ancilla i checks data i and i+1).
Experiment: prepare |0...0>, run R syndrome rounds, measure data in Z.
A logical error = decoded logical value != 0.
"""
import numpy as np
import stim
import pymatching
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def layout(distance):
    data = [2 * i for i in range(distance)]
    anc = [2 * i + 1 for i in range(distance - 1)]
    cx_pairs = []  # (control, target)
    for i, a in enumerate(anc):
        cx_pairs += [(data[i], a), (data[i + 1], a)]
    return data, anc, cx_pairs


# ---------------------------------------------------------------- Qiskit
def qiskit_circuit(distance=5, rounds=3):
    n_anc = distance - 1
    data = QuantumRegister(distance, "d")
    anc = QuantumRegister(n_anc, "a")
    syn = [ClassicalRegister(n_anc, f"syn_{r}") for r in range(rounds)]
    fin = ClassicalRegister(distance, "fin")
    qc = QuantumCircuit(data, anc, *syn, fin)
    for r in range(rounds):
        for i in range(n_anc):
            qc.cx(data[i], anc[i])
            qc.cx(data[i + 1], anc[i])
        qc.barrier()
        qc.measure(anc, syn[r])
        qc.reset(anc)
        qc.barrier()
    qc.measure(data, fin)
    return qc


# ------------------------------------------------------------------ Stim
def stim_circuit(distance=5, rounds=3, p_data=0.01, p_cx=0.01, p_meas=0.01,
                 cx_p=None):
    """cx_p: optional {(control, target): prob} overriding p_cx per link
    (the hook for planted defects, e.g. {(2, 1): 0.1})."""
    cx_p = cx_p or {}
    data, anc, cx_pairs = layout(distance)
    n_anc = len(anc)
    c = stim.Circuit()
    c.append("R", sorted(data + anc))
    for r in range(rounds):
        c.append("X_ERROR", data, p_data)
        for ctl, tgt in cx_pairs:
            c.append("CX", [ctl, tgt])
            c.append("DEPOLARIZE2", [ctl, tgt], cx_p.get((ctl, tgt), p_cx))
        c.append("MR", anc, p_meas)
        for i in range(n_anc):
            cur = stim.target_rec(-(n_anc - i))
            if r == 0:
                c.append("DETECTOR", [cur])
            else:
                c.append("DETECTOR", [cur, stim.target_rec(-(n_anc - i) - n_anc)])
    c.append("X_ERROR", data, p_data)
    c.append("M", data)
    for i in range(n_anc):
        last = stim.target_rec(-distance - (n_anc - i))
        c.append("DETECTOR", [last,
                              stim.target_rec(-distance + i),
                              stim.target_rec(-distance + i + 1)])
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-distance)], 0)
    return c


def logical_error_rate(circuit, shots=20000, matching=None):
    if matching is None:
        dem = circuit.detector_error_model(
            decompose_errors=True, ignore_decomposition_failures=True)
        matching = pymatching.Matching.from_detector_error_model(dem)
    det, obs = circuit.compile_detector_sampler().sample(
        shots, separate_observables=True)
    pred = matching.decode_batch(det)
    return float(np.mean(np.any(pred != obs, axis=1)))


def unprotected_error_rate(rounds=3, p=0.01):
    """One idle qubit, independent bit-flip prob p per round: P(odd # flips)."""
    return (1 - (1 - 2 * p) ** rounds) / 2


if __name__ == "__main__":
    c = stim_circuit(distance=5, rounds=3, p_data=0.02, p_cx=0.02, p_meas=0.02)
    print("decoded logical error rate:", logical_error_rate(c))
    print("unprotected (p=0.02):", unprotected_error_rate(3, 0.02))
    print(qiskit_circuit(5, 3).draw(fold=140))