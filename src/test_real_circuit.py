"""
test_real_circuit.py -- checks the REAL qec_circuits.qiskit_circuit against the
assumptions in qiskit_parse.py. Run before submitting anything to hardware.

Run from repo root (PowerShell):   python src/test_real_circuit.py

Checks
  1. structure: qubit count, register names/sizes, measure/reset counts
  2. noiseless Aer run: every syndrome, final bit and detection event is 0
  3. planted flip: X on data qubit 2 (= data index 1) applied before round 0 must give
       syndromes   [1,1,0,0] in every round   (ancillas on qubits 1 and 3)
       final data  [0,1,0,0,0]
       detection events: only round 0 = [1,1,0,0], all others 0
     This pins ancilla k -> syn_r[k] and data j -> fin[j] (a reversed mapping would give
     [0,0,1,1] and [0,0,0,1,0]).
Step 3 assumes data qubits are NOT reset after the flip. If it fails with all-zero
output, your circuit resets data at the start; tell me and I'll move the flip.
"""
import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qec_circuits import qiskit_circuit
from qiskit_parse import assemble, counts_to_shots, detection_events

D, ROUNDS, SHOTS = 5, 3, 200
N_ANC = D - 1


def run(qc):
    reg_names = [r.name for r in qc.cregs]
    counts = AerSimulator().run(qc, shots=SHOTS).result().get_counts()
    shots = counts_to_shots(counts, reg_names)
    syn, final = assemble(shots, ROUNDS)
    return syn, final, detection_events(syn, final), counts


def main():
    qc = qiskit_circuit(distance=D, rounds=ROUNDS)

    # 1. structure
    want_regs = [f"syn_{r}" for r in range(ROUNDS)] + ["fin"]
    got_regs = [(r.name, r.size) for r in qc.cregs]
    ops = qc.count_ops()
    print("qubits:", qc.num_qubits, "| cregs:", got_regs, "| ops:", dict(ops))
    assert qc.num_qubits == 2 * D - 1, "expected 2d-1 qubits"
    assert [n for n, _ in got_regs] == want_regs, f"register names/order {got_regs}"
    assert [s for _, s in got_regs] == [N_ANC] * ROUNDS + [D], "register sizes"
    assert ops.get("measure", 0) == N_ANC * ROUNDS + D, "measure count"
    assert ops.get("reset", 0) >= N_ANC * ROUNDS, "expected an ancilla reset per round"
    print("PASS  structure")

    # 2. noiseless
    syn, final, det, _ = run(qc)
    assert not syn.any() and not final.any() and not det.any(), "noiseless run is not all-zero"
    print("PASS  noiseless run is all-zero")

    # 3. planted flip on data qubit 2, before round 0
    flipped = QuantumCircuit(*qc.qregs, *qc.cregs)
    flipped.x(2)
    flipped.compose(qc, inplace=True)
    syn, final, det, counts = run(flipped)
    print("flipped counts key:", list(counts))
    want_syn = np.array([1, 1, 0, 0], dtype=np.uint8)
    want_fin = np.array([0, 1, 0, 0, 0], dtype=np.uint8)
    want_det = np.zeros((ROUNDS + 1, N_ANC), dtype=np.uint8)
    want_det[0] = want_syn
    assert (syn == want_syn).all(), f"syndromes {syn[0].tolist()} != {want_syn.tolist()} per round"
    assert (final == want_fin).all(), f"final data {final[0].tolist()} != {want_fin.tolist()}"
    assert (det == want_det).all(), f"detection events {det[0].tolist()}"
    print("PASS  planted flip parses to the expected syndromes / final data / detection events")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL ", e)
        sys.exit(1)