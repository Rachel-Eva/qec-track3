"""
run_hardware_ibm.py -- IBM Quantum Hardware Validation.

Executes the Pauli-twirled repetition code on real IBM Quantum hardware:
  1. Clean calibration run (1,000 shots)
  2. Normal operation run (3,000 shots)

Dumps raw bitstrings into shared/syndrome_telemetry.json for the STPP
estimator to consume.

▸▸▸ REQUIRES: IBM Quantum API token                                   ◂◂◂
▸▸▸ Set environment variable IBM_QUANTUM_TOKEN or edit stpp_config.py  ◂◂◂
▸▸▸ You will run this file separately with your API key.               ◂◂◂

Usage:
  # Option 1: environment variable
  $env:IBM_QUANTUM_TOKEN = "your-token-here"
  python src/run_hardware_ibm.py

  # Option 2: pass as argument
  python src/run_hardware_ibm.py --token "your-token-here"

  # Option 3: edit IBM_TOKEN in src/stpp_config.py
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stpp_config import (
    SHARED, DISTANCE, N_ANCILLAS, ROUNDS_DEFAULT,
    IBM_TOKEN, IBM_BACKEND_NAME,
    IBM_SHOTS_CALIBRATION, IBM_SHOTS_DEFECT,
    SYNDROME_TELEMETRY_PATH,
)
from qec_circuits import qiskit_circuit
from qiskit_parse import (
    pub_result_to_shots, counts_to_shots, assemble, detection_events,
)


def get_service(token=None):
    """
    Initialize the IBM Quantum Runtime Service.

    ▸▸▸ FILL: Your IBM Quantum token goes here (or via env var).      ◂◂◂
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    tok = token or IBM_TOKEN
    if tok is None:
        raise ValueError(
            "No IBM Quantum token found. Set IBM_QUANTUM_TOKEN env var, "
            "pass --token, or edit IBM_TOKEN in stpp_config.py"
        )

    # ▸▸▸ FILL: If you need a specific instance/channel, edit below.  ◂◂◂
    service = QiskitRuntimeService(
        channel="ibm_quantum",     # or "ibm_cloud"
        token=tok,
        # instance="ibm-q/open/main",  # ◂◂◂ FILL if needed
    )
    return service


def select_backend(service, backend_name=None):
    """
    Select the least-busy backend, or a specific one.

    ▸▸▸ FILL: Choose your preferred backend. Options:                 ◂◂◂
    ▸▸▸   ibm_brisbane, ibm_sherbrooke, ibm_kyiv, etc.                ◂◂◂
    """
    name = backend_name or IBM_BACKEND_NAME
    try:
        backend = service.backend(name)
        print(f"[IBM] Using backend: {backend.name}")
    except Exception:
        print(f"[IBM] Backend '{name}' not available. Listing alternatives...")
        backends = service.backends(
            simulator=False,
            operational=True,
            min_num_qubits=2 * DISTANCE - 1,
        )
        if not backends:
            raise RuntimeError("No suitable backends available.")
        backend = backends[0]
        print(f"[IBM] Falling back to: {backend.name}")

    return backend


def build_hardware_circuit(distance=DISTANCE, rounds=ROUNDS_DEFAULT):
    """
    Build the Qiskit circuit for hardware execution.

    Uses optimization_level=0 and barriers to preserve syndrome extraction
    structure. No Pauli twirling in this base version.

    ▸▸▸ OPTIONAL: Add Pauli twirl frames around CX gates.             ◂◂◂
    ▸▸▸ See add_pauli_twirling() below for the extension point.       ◂◂◂
    """
    qc = qiskit_circuit(distance=distance, rounds=rounds)
    return qc


def add_pauli_twirling(qc):
    """
    Insert randomized Pauli twirl frames around each CX gate.

    ▸▸▸ FILL: Implement Pauli twirling for rigorous STPP projection.  ◂◂◂
    ▸▸▸ For each CX, randomly choose Pauli pair (P_c, P_t) from the   ◂◂◂
    ▸▸▸ twirling group and insert P_c/P_t before and P_c'/P_t' after. ◂◂◂
    ▸▸▸ See: https://arxiv.org/abs/2305.13527 for the twirling group. ◂◂◂
    
    For now, this is a no-op pass-through.
    """
    # ── Pauli twirling group for CX: 16 elements ────────────────────
    # Each element: (pre_control, pre_target, post_control, post_target)
    # where each is in {I, X, Y, Z}
    # 
    # TWIRL_GROUP = [
    #     ("I", "I", "I", "I"),
    #     ("I", "X", "I", "X"),
    #     ("I", "Y", "Z", "Y"),
    #     ("I", "Z", "Z", "Z"),
    #     ("X", "I", "X", "X"),
    #     ("X", "X", "X", "I"),
    #     ("X", "Y", "Y", "Z"),
    #     ("X", "Z", "Y", "Y"),
    #     ("Y", "I", "Y", "X"),
    #     ("Y", "X", "Y", "I"),
    #     ("Y", "Y", "X", "Z"),
    #     ("Y", "Z", "X", "Y"),
    #     ("Z", "I", "Z", "I"),
    #     ("Z", "X", "Z", "X"),
    #     ("Z", "Y", "I", "Y"),
    #     ("Z", "Z", "I", "Z"),
    # ]
    #
    # TODO: Iterate through qc.data, find CX gates, insert twirl frames.

    return qc  # Pass-through for now


def run_on_hardware(backend, qc, shots, optimization_level=0):
    """
    Execute circuit on IBM Quantum hardware via Sampler.

    ▸▸▸ FILL: Adjust for your Qiskit Runtime version.                 ◂◂◂
    ▸▸▸ SamplerV2 vs SamplerV1 API may differ.                        ◂◂◂
    """
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    # Transpile with minimal optimization to preserve circuit structure
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
    )
    transpiled = pm.run(qc)
    print(f"[IBM] Transpiled circuit: {transpiled.num_qubits} qubits, "
          f"depth={transpiled.depth()}")

    sampler = Sampler(mode=backend)
    job = sampler.run([transpiled], shots=shots)
    print(f"[IBM] Job submitted: {job.job_id()}")
    print(f"[IBM] Waiting for results...")

    result = job.result()
    pub_result = result[0]

    # Parse results
    reg_names = [r.name for r in qc.cregs]
    shots_dict = pub_result_to_shots(pub_result, reg_names)

    return shots_dict


def run_on_aer_fallback(qc, shots, seed=42):
    """
    Fallback: run on Aer simulator if hardware is unavailable.
    """
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    result = sim.run(qc, shots=shots, seed_simulator=seed).result()
    counts = result.get_counts()
    reg_names = [r.name for r in qc.cregs]
    return counts_to_shots(counts, reg_names)


def process_results(shots_dict, rounds, distance):
    """
    Convert raw measurement results to syndromes and detection events.

    Returns (syndromes, final_data, det_events)
    """
    syn, final = assemble(shots_dict, rounds)
    det = detection_events(syn, final)
    return syn, final, det


def export_hardware_telemetry(run_data, output_path):
    """
    Write hardware results to the shared telemetry JSON.
    """
    output_path.parent.mkdir(exist_ok=True)

    # Load existing telemetry if present
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
    else:
        existing = {
            "schema_version": 2,
            "runs": [],
        }

    # Append the new run
    existing["runs"].append(run_data)
    existing["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"[IBM] Exported → {output_path}")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="IBM Quantum Hardware Validation")
    parser.add_argument("--token", type=str, default=None,
                        help="IBM Quantum API token")
    parser.add_argument("--backend", type=str, default=IBM_BACKEND_NAME,
                        help=f"Backend name (default: {IBM_BACKEND_NAME})")
    parser.add_argument("--distance", type=int, default=DISTANCE)
    parser.add_argument("--rounds", type=int, default=ROUNDS_DEFAULT)
    parser.add_argument("--shots-cal", type=int, default=IBM_SHOTS_CALIBRATION,
                        help="Shots for calibration run")
    parser.add_argument("--shots-main", type=int, default=IBM_SHOTS_DEFECT,
                        help="Shots for main run")
    parser.add_argument("--aer-fallback", action="store_true",
                        help="Use Aer simulator instead of hardware")
    args = parser.parse_args()

    print("=" * 72)
    print(" IBM Quantum Hardware Validation")
    print("=" * 72)

    # ── Build circuit ────────────────────────────────────────────────
    qc = build_hardware_circuit(distance=args.distance, rounds=args.rounds)
    print(f"[IBM] Circuit: {qc.num_qubits} qubits, "
          f"{args.rounds} rounds, d={args.distance}")

    # ── Optional: add Pauli twirling ─────────────────────────────────
    qc = add_pauli_twirling(qc)

    # ── Run ──────────────────────────────────────────────────────────
    if args.aer_fallback:
        print("[IBM] Using Aer simulator fallback.")
        backend_name = "aer_simulator"

        # Calibration run
        print(f"\n[1/2] Calibration run ({args.shots_cal} shots)...")
        cal_shots = run_on_aer_fallback(qc, args.shots_cal, seed=42)
        cal_syn, cal_final, cal_det = process_results(
            cal_shots, args.rounds, args.distance
        )

        # Main run
        print(f"\n[2/2] Main run ({args.shots_main} shots)...")
        main_shots = run_on_aer_fallback(qc, args.shots_main, seed=123)
        main_syn, main_final, main_det = process_results(
            main_shots, args.rounds, args.distance
        )
    else:
        # ▸▸▸ This path requires your IBM Quantum token ◂◂◂
        service = get_service(token=args.token)
        backend = select_backend(service, args.backend)
        backend_name = backend.name

        # Calibration run
        print(f"\n[1/2] Calibration run ({args.shots_cal} shots)...")
        cal_shots = run_on_hardware(backend, qc, args.shots_cal)
        cal_syn, cal_final, cal_det = process_results(
            cal_shots, args.rounds, args.distance
        )

        # Main run
        print(f"\n[2/2] Main run ({args.shots_main} shots)...")
        main_shots = run_on_hardware(backend, qc, args.shots_main)
        main_syn, main_final, main_det = process_results(
            main_shots, args.rounds, args.distance
        )

    # ── Summary statistics ───────────────────────────────────────────
    print(f"\n[IBM] Calibration: det_event_rate = {cal_det.mean():.4f}")
    print(f"[IBM] Main:        det_event_rate = {main_det.mean():.4f}")

    # ── Export ───────────────────────────────────────────────────────
    def to_bitstrings(arr):
        return ["".join(map(str, row)) for row in arr.astype(np.uint8)]

    def make_run_data(run_id, syn, final, det, n_shots, role):
        return {
            "run_id": run_id,
            "role": role,
            "source": "ibm_hardware" if not args.aer_fallback else "aer_fallback",
            "backend": backend_name,
            "distance": args.distance,
            "rounds": args.rounds,
            "shots": n_shots,
            "detection_rate_mean": float(det.mean()),
            "detection_rate_per_ancilla": det.mean(axis=(0, 1)).tolist(),
            "inline_shots": [
                {
                    "syndromes": to_bitstrings(syn[i]),
                    "final_data": "".join(map(str, final[i])),
                    "detection_events": to_bitstrings(det[i]),
                }
                for i in range(min(200, n_shots))
            ],
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    cal_data = make_run_data("hw_calibration", cal_syn, cal_final, cal_det,
                              args.shots_cal, "baseline")
    main_data = make_run_data("hw_main", main_syn, main_final, main_det,
                               args.shots_main, "observed")

    hw_telemetry_path = SHARED / "hardware_telemetry.json"
    export_hardware_telemetry(cal_data, hw_telemetry_path)
    export_hardware_telemetry(main_data, hw_telemetry_path)

    # Also save raw NPZ for the estimator
    from stpp_config import RAW_DIR
    for name, det_arr in [("hw_calibration", cal_det), ("hw_main", main_det)]:
        npz_path = RAW_DIR / f"{name}.npz"
        flat = det_arr.reshape(det_arr.shape[0], -1)
        np.savez_compressed(
            npz_path,
            detection_events=np.packbits(flat.astype(np.uint8), axis=1),
            num_detectors=flat.shape[1],
        )
        print(f"[IBM] Raw data → {npz_path}")

    print("\n[IBM] Done. Run 02_stpp_estimator.py on hw_calibration / hw_main.")


if __name__ == "__main__":
    main()
