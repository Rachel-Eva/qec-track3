"""
export_telemetry.py  --  Person B, Hours 5-6

Samples the Stim repetition-code memory experiment and writes:

  shared/syndrome_telemetry.json        what Person A reads (NO planted truth in here)
  shared/telemetry_raw/<run_id>.npz     every shot, bit-packed, so it can be re-decoded
  shared/planted_truth.json             planted defects; for SCORING only, never for weights

Runs:
  baseline          clean circuit (no defects)      -> per-ancilla p0
  obs_a / obs_b / obs_c   1, 2 and 3 planted defects (ids do not reveal the count)

Everything numeric is computed from sampled shots. Nothing is hardcoded.

Usage (from repo root, PowerShell):
    python src/export_telemetry.py --shots 100000 --p 0.02

Layout assumption (checked below): 2d-1 qubits on a line, data on even indices,
ancillas on odd indices, link id = "data-ancilla" (e.g. "2-1").
Measurement order assumption (also checked below): per round, ancillas in
ascending qubit order; then the d final data measurements.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pymatching
import stim

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "shared"
RAW_DIR = SHARED / "telemetry_raw"

# Planted defects: link id -> CX error multiplier k.  Edit freely.
DEFECT_RUNS = {
    "obs_a": {"2-1": 10.0},
    "obs_b": {"2-1": 10.0, "6-7": 5.0},
    "obs_c": {"2-1": 10.0, "6-7": 5.0, "4-3": 8.0},
}


# --------------------------------------------------------------------------
# ADAPTER: the only place that touches src/noise_models.py. Edit to match.
# --------------------------------------------------------------------------
def build_circuit(p_phys, defects, d, rounds):
    """Return a stim.Circuit. `defects` is {"data-anc": k}; may be empty."""
    from noise_models import make_circuit  # src/ is on sys.path when run as a script

    # make_circuit(distance=5, rounds=3, p=0.01, defects=None, link_p=None)
    # defects keys are (ctl, tgt) = (data, ancilla), i.e. "2-1" -> (2, 1)
    defect_arg = {tuple(int(x) for x in k.split("-")): v for k, v in defects.items()}
    return make_circuit(distance=d, rounds=rounds, p=p_phys,
                        defects=defect_arg or None)


# --------------------------------------------------------------------------
def layout(d):
    n = 2 * d - 1
    data = list(range(0, n, 2))
    anc = list(range(1, n, 2))
    links = [f"{q}-{a}" for a in anc for q in (a - 1, a + 1)]
    return n, data, anc, links


def check_ordering(meas, dets, d, rounds):
    """Recompute detection events from raw syndromes and compare with Stim's.
    Passing validates BOTH the measurement-order and detector-order assumptions."""
    n_anc = d - 1
    s = meas[:, : n_anc * rounds].reshape(-1, rounds, n_anc).astype(np.uint8)
    data = meas[:, n_anc * rounds:].astype(np.uint8)
    prev = np.zeros_like(s[:, 0])
    rows = []
    for r in range(rounds):
        rows.append(s[:, r] ^ prev)
        prev = s[:, r]
    rows.append(data[:, :-1] ^ data[:, 1:] ^ prev)  # final round: ancilla k sees data k, k+1
    expected = np.concatenate(rows, axis=1).astype(bool)
    if expected.shape != dets.shape or not np.array_equal(expected, dets):
        raise RuntimeError(
            "Ordering check FAILED: detection events recomputed from raw syndromes do not "
            "match Stim's detectors. Ancillas may not be reset each round, or measurement/"
            "detector order differs from the assumption. Fix before sharing anything."
        )


def to_bitstrings(a):
    return ["".join(map(str, row)) for row in a.astype(np.uint8)]


def uniform_logical_errors(p_phys, dets, obs, d, rounds):
    """Baseline decoder: MWPM with weights from the NOMINAL (defect-free) model."""
    nominal = build_circuit(p_phys, {}, d, rounds)
    try:
        dem = nominal.detector_error_model(decompose_errors=True)
    except Exception:
        dem = nominal.detector_error_model()
    matching = pymatching.Matching.from_detector_error_model(dem)
    pred = matching.decode_batch(dets)
    return int(np.sum(np.any(pred != obs, axis=1)))


def run_one(run_id, role, p_phys, defects, shots, seed, d, rounds, n_inline):
    n, data_q, anc_q, links = layout(d)
    n_anc = len(anc_q)

    circ = build_circuit(p_phys, defects, d, rounds)
    assert circ.num_qubits == n, f"circuit has {circ.num_qubits} qubits, expected {n}"
    assert circ.num_measurements == n_anc * rounds + d, "unexpected measurement count"
    assert circ.num_detectors == n_anc * (rounds + 1), "unexpected detector count"
    for k in defects:
        assert k in links, f"defect {k} is not a link in {links}"

    meas = circ.compile_sampler(seed=seed).sample(shots)
    dets, obs = circ.compile_m2d_converter().convert(
        measurements=meas, separate_observables=True
    )
    check_ordering(meas, dets, d, rounds)

    synd = meas[:, : n_anc * rounds].reshape(shots, rounds, n_anc)
    final = meas[:, n_anc * rounds:]
    det_r = dets.reshape(shots, rounds + 1, n_anc)  # [shot][round (last = final)][ancilla]

    rate = det_r.mean(axis=0)  # [round][ancilla]
    logical_errors = uniform_logical_errors(p_phys, dets, obs, d, rounds)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"{run_id}.npz"
    np.savez_compressed(
        raw_path,
        measurements=np.packbits(meas, axis=1),
        detection_events=np.packbits(dets, axis=1),
        observable_flips=obs.astype(np.uint8),
        num_measurements=meas.shape[1],
        num_detectors=dets.shape[1],
        seed=seed,
    )

    inline = []
    for i in range(min(n_inline, shots)):
        inline.append({
            "syndromes": to_bitstrings(synd[i]),        # per round, ancillas ascending
            "final_data": "".join(map(str, final[i].astype(np.uint8))),
            "detection_events": to_bitstrings(det_r[i]),  # per round + final-round row
            "observable_flip": int(obs[i, 0]),
        })

    run = {
        "run_id": run_id,
        "role": role,  # "baseline" or "observed"
        "source": "stim_simulation",
        "p_phys": p_phys,
        "shots": shots,
        "seed": seed,
        "circuit_sha256": hashlib.sha256(str(circ).encode()).hexdigest()[:16],
        "decoder": "uniform_mwpm",
        "logical_errors": logical_errors,
        "logical_error_rate": logical_errors / shots,
        "detector_counts": dets.sum(axis=0).astype(int).tolist(),  # round-major, ancilla-minor
        "detection_rate_by_ancilla": {
            str(a): rate[:, j].tolist() for j, a in enumerate(anc_q)
        },  # list over rounds 0..rounds (last entry = final-round detectors)
        "detection_rate_mean_by_ancilla": {
            str(a): float(rate[:, j].mean()) for j, a in enumerate(anc_q)
        },
        "raw_file": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
        "inline_shots": inline,
    }
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--p", type=float, default=0.02)
    ap.add_argument("--shots", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--inline", type=int, default=500, help="shots embedded in the JSON")
    args = ap.parse_args()

    n, data_q, anc_q, links = layout(args.d)
    SHARED.mkdir(exist_ok=True)

    runs = [run_one("baseline", "baseline", args.p, {}, args.shots, args.seed,
                    args.d, args.rounds, args.inline)]
    for i, (rid, defects) in enumerate(DEFECT_RUNS.items(), start=1):
        runs.append(run_one(rid, "observed", args.p, defects, args.shots, args.seed + i,
                            args.d, args.rounds, args.inline))

    telemetry = {
        "schema_version": 1,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "versions": {"stim": stim.__version__, "pymatching": pymatching.__version__},
        "code": {
            "type": "bit_flip_repetition",
            "distance": args.d,
            "rounds": args.rounds,
            "n_qubits": n,
            "data_qubits": data_q,
            "ancilla_qubits": anc_q,
            "link_ids": links,  # Stim ids "data-ancilla"
            "detector_layout": "round-major, ancilla ascending; round index "
                               f"{args.rounds} = final-round detectors",
            "detection_event_definition": "syndrome(round r) XOR syndrome(round r-1); "
                                          "round 0 vs 0; final row = data parity XOR last syndrome",
        },
        "runs": runs,
    }
    (SHARED / "syndrome_telemetry.json").write_text(json.dumps(telemetry, indent=1))

    truth = {
        "WARNING": "Scoring only. Informed decoder weights must come from Person A's estimates.",
        "runs": {rid: {"defects": d_} for rid, d_ in DEFECT_RUNS.items()},
    }
    (SHARED / "planted_truth.json").write_text(json.dumps(truth, indent=1))

    print(f"Wrote {SHARED / 'syndrome_telemetry.json'}")
    for r in runs:
        means = {a: round(v, 4) for a, v in r["detection_rate_mean_by_ancilla"].items()}
        print(f"{r['run_id']:9s} shots={r['shots']} uniform logical_errors={r['logical_errors']} "
              f"({r['logical_error_rate']:.5f})  det-rate/ancilla={means}")


if __name__ == "__main__":
    main()