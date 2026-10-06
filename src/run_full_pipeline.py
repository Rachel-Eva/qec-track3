"""
run_full_pipeline.py -- Master orchestrator for the STPP-Triage pipeline.

Runs all stages end-to-end in the correct dependency order:
  0. (Optional) Kaggle prior generation
  1. Simulation data generation (export_telemetry.py — already exists)
  2. STPP estimation and triage
  3. STPP-aware decoding benchmark
  4. Digital twin visualization
  5. Task 8 trade-off suite

Usage:
  python src/run_full_pipeline.py                     # full pipeline
  python src/run_full_pipeline.py --skip-kaggle        # skip step 0
  python src/run_full_pipeline.py --skip-sim           # skip step 1 (use existing data)
  python src/run_full_pipeline.py --only estimator     # run single stage
"""
import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_stage(name, func, skip=False):
    """Run a pipeline stage with timing and error handling."""
    if skip:
        print(f"\n{'═' * 72}")
        print(f" ⏭  SKIPPED: {name}")
        print(f"{'═' * 72}")
        return True

    print(f"\n{'═' * 72}")
    print(f" ▶  Stage: {name}")
    print(f"{'═' * 72}")

    t0 = time.perf_counter()
    try:
        func()
        elapsed = time.perf_counter() - t0
        print(f"\n ✓  {name} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"\n ✗  {name} FAILED after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="STPP-Triage: Full Pipeline Orchestrator"
    )
    parser.add_argument("--skip-kaggle", action="store_true",
                        help="Skip Kaggle prior generation")
    parser.add_argument("--skip-sim", action="store_true",
                        help="Skip simulation data generation (use existing)")
    parser.add_argument("--skip-tradeoffs", action="store_true",
                        help="Skip Task 8 trade-off suite (slow)")
    parser.add_argument("--only", type=str, default=None,
                        choices=["kaggle", "sim", "estimator", "decoder",
                                 "twin", "tradeoffs"],
                        help="Run only a single stage")
    args = parser.parse_args()

    t_total = time.perf_counter()
    print("╔" + "═" * 70 + "╗")
    print("║  STPP-Triage: In-Situ Characterization & Non-Markovian Decoding   ║")
    print("║  of Spatiotemporal Pauli Processes in Quantum Error Correction     ║")
    print("╚" + "═" * 70 + "╝")

    results = {}

    # ── Stage 0: Kaggle Prior ────────────────────────────────────────
    if args.only is None or args.only == "kaggle":
        from stpp_config import STPP_PRIOR_PATH

        def run_kaggle():
            from importlib import import_module
            mod = import_module("01_kaggle_stpp_prior")
            mod.main()

        results["kaggle"] = run_stage(
            "Kaggle STPP Prior Generation",
            run_kaggle,
            skip=(args.skip_kaggle and args.only is None),
        )

    # ── Stage 1: Simulation Data ─────────────────────────────────────
    if args.only is None or args.only == "sim":
        def run_sim():
            from importlib import import_module
            mod = import_module("export_telemetry")
            mod.main(argv=[])

        results["sim"] = run_stage(
            "Simulation Data Generation (export_telemetry)",
            run_sim,
            skip=(args.skip_sim and args.only is None),
        )

    # ── Stage 2: STPP Estimation ─────────────────────────────────────
    if args.only is None or args.only == "estimator":
        def run_estimator():
            from importlib import import_module
            mod = import_module("02_stpp_estimator")
            mod.main()

        results["estimator"] = run_stage(
            "STPP Estimation & Triage",
            run_estimator,
        )

    # ── Stage 3: STPP Decoder Benchmark ──────────────────────────────
    if args.only is None or args.only == "decoder":
        def run_decoder():
            from importlib import import_module
            mod = import_module("decoder_stpp")
            mod.main()

        results["decoder"] = run_stage(
            "STPP-Aware Decoder Benchmark",
            run_decoder,
        )

    # ── Stage 4: Digital Twin ────────────────────────────────────────
    if args.only is None or args.only == "twin":
        def run_twin():
            from importlib import import_module
            mod = import_module("03_stpp_digital_twin")
            mod.main()

        results["twin"] = run_stage(
            "STPP Digital Twin Visualization",
            run_twin,
        )

    # ── Stage 5: Task 8 Trade-offs ───────────────────────────────────
    if args.only is None or args.only == "tradeoffs":
        def run_tradeoffs():
            from importlib import import_module
            mod = import_module("04_task8_tradeoffs")
            mod.main()

        results["tradeoffs"] = run_stage(
            "Task 8 Trade-Off Suite",
            run_tradeoffs,
            skip=(args.skip_tradeoffs and args.only is None),
        )

    # ── Summary ──────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_total
    print(f"\n{'═' * 72}")
    print(f" Pipeline Summary ({elapsed_total:.1f}s total)")
    print(f"{'═' * 72}")
    for stage, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {stage:20s} {status}")

    print(f"\n Output directories:")
    from stpp_config import SHARED, FIGURES
    print(f"   shared/  → {SHARED}")
    print(f"   figures/ → {FIGURES}")

    if all(results.values()):
        print(f"\n 🎉 All stages passed!")
    else:
        failed = [s for s, ok in results.items() if not ok]
        print(f"\n ⚠  Failed stages: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
