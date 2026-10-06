"""
test_qiskit_bit_order.py -- run BEFORE parsing any QPU output.

Builds a d=3, 2-round-shaped circuit (registers syn_0, syn_1 with 2 bits each, fin with
3 bits) in which three asymmetric bits are flipped on purpose:
    syn_0[0]=1, syn_1[1]=1, fin[2]=1
Everything else is 0. If any ordering assumption (within-register or between-register)
is wrong, the parsed arrays will not match the literals below.

Run from repo root (PowerShell):   python src/test_qiskit_bit_order.py
"""
import sys

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

from qiskit_parse import (assemble, counts_to_shots, detection_events,
                          pub_result_to_shots)

REGS = ["syn_0", "syn_1", "fin"]  # order ADDED to the circuit
SHOTS = 64

EXPECT = {
    "syn_0": [1, 0],
    "syn_1": [0, 1],
    "fin": [0, 0, 1],
}
# rounds 0,1 then final row; ancilla order ascending
EXPECT_DET = [[1, 0], [1, 1], [0, 0]]


def build():
    qr = QuantumRegister(7, "q")
    syn_0 = ClassicalRegister(2, "syn_0")
    syn_1 = ClassicalRegister(2, "syn_1")
    fin = ClassicalRegister(3, "fin")
    qc = QuantumCircuit(qr, syn_0, syn_1, fin)
    for q in (0, 3, 6):  # qubit -> clbit in creation order: 0,1 | 2,3 | 4,5,6
        qc.x(q)
    qc.measure(qr[0], syn_0[0])
    qc.measure(qr[1], syn_0[1])
    qc.measure(qr[2], syn_1[0])
    qc.measure(qr[3], syn_1[1])
    qc.measure(qr[4], fin[0])
    qc.measure(qr[5], fin[1])
    qc.measure(qr[6], fin[2])
    return qc


def check(shots_by_reg, label):
    for name, want in EXPECT.items():
        got = shots_by_reg[name]
        assert got.shape == (SHOTS, len(want)), f"{label}: {name} shape {got.shape}"
        assert (got == np.array(want)).all(), f"{label}: {name} parsed {got[0].tolist()} expected {want}"
    syn, final = assemble(shots_by_reg, rounds=2)
    det = detection_events(syn, final)
    assert (det == np.array(EXPECT_DET)).all(), f"{label}: detection events {det[0].tolist()} != {EXPECT_DET}"
    print(f"PASS  {label}")


def main():
    qc = build()

    # Path 1: backend.run().get_counts() -> space-separated keys
    counts = AerSimulator().run(qc, shots=SHOTS).result().get_counts()
    print("raw counts key (note register order / spaces):", list(counts))
    check(counts_to_shots(counts, REGS), "get_counts() path")

    # Path 2: SamplerV2 per-register path (what IBM hardware returns)
    try:
        from qiskit_aer.primitives import SamplerV2
    except ImportError:
        print("SKIP  SamplerV2 path (qiskit_aer.primitives.SamplerV2 not available)")
        return
    res = SamplerV2().run([qc], shots=SHOTS).result()[0]
    check(pub_result_to_shots(res, REGS), "SamplerV2 per-register path")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL ", e)
        sys.exit(1)