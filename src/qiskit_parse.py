"""
qiskit_parse.py -- the ONLY place that knows Qiskit's bit-ordering rules.

Qiskit conventions this module encodes (verified by src/test_qiskit_bit_order.py):
  * Bitstrings are little-endian: the RIGHTMOST character is clbit 0 of that register.
  * backend.run(...).get_counts() keys join registers with spaces, LAST-added register
    on the LEFT:  "fin syn_2 syn_1 syn_0".
  * SamplerV2 results are per register: result.data.<reg>.get_bitstrings(), same shot
    order across registers, each string little-endian within its register.

Convention for the repetition-code circuit (confirm it matches qec_circuits.py):
  syn_r clbit k  <- ancilla k (ascending qubit order) in round r
  fin   clbit j  <- data qubit j (ascending)
"""
import numpy as np


def strings_to_bits(strings):
    """list[str] (little-endian) -> uint8 array (shots, nbits); column j == clbit j."""
    return np.array([[int(c) for c in s[::-1]] for s in strings], dtype=np.uint8)


def counts_to_shots(counts, reg_names_in_order):
    """counts: {"fin syn_1 syn_0": n, ...} from get_counts().
    reg_names_in_order: registers in the order they were ADDED to the circuit.
    Returns {name: (shots, nbits) uint8}. Shot order is arbitrary but identical across
    registers, so shot-by-shot correlations are preserved."""
    cols = {n: [] for n in reg_names_in_order}
    for key, n in counts.items():
        parts = key.split()
        if len(parts) != len(reg_names_in_order):
            raise ValueError(
                f"key {key!r} has {len(parts)} fields, expected {len(reg_names_in_order)} "
                f"({reg_names_in_order}). Registers may have been merged or reordered."
            )
        for name, bits in zip(reg_names_in_order, parts[::-1]):
            cols[name].extend([bits] * n)
    return {name: strings_to_bits(v) for name, v in cols.items()}


def pub_result_to_shots(pub_result, reg_names):
    """SamplerV2 PubResult -> {name: (shots, nbits) uint8}."""
    return {n: strings_to_bits(getattr(pub_result.data, n).get_bitstrings())
            for n in reg_names}


def assemble(shots_by_reg, rounds):
    """-> syndromes (shots, rounds, n_anc), final_data (shots, d)."""
    syn = np.stack([shots_by_reg[f"syn_{r}"] for r in range(rounds)], axis=1)
    return syn, shots_by_reg["fin"]


def detection_events(syn, final):
    """Same definition as export_telemetry.check_ordering.
    Returns (shots, rounds+1, n_anc); last row = final-round detectors."""
    syn = syn.astype(np.uint8)
    final = final.astype(np.uint8)
    prev = np.zeros_like(syn[:, 0])
    rows = []
    for r in range(syn.shape[1]):
        rows.append(syn[:, r] ^ prev)
        prev = syn[:, r]
    rows.append(final[:, :-1] ^ final[:, 1:] ^ prev)
    return np.stack(rows, axis=1)