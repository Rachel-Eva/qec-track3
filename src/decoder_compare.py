"""Sample from the TRUE defective circuit once, decode the same shots with
different matching graphs:
  uniform  - graph built assuming no defect
  informed - graph built from Person A's link_error_estimates (skipped if absent)
  oracle   - graph built from the true planted defect
All numbers are measured logical error rates; nothing is hardcoded.
"""
import numpy as np
import pymatching

from noise_models import make_circuit, load_profile


def matching_from(circuit):
    dem = circuit.detector_error_model(
        decompose_errors=True, ignore_decomposition_failures=True)
    return pymatching.Matching.from_detector_error_model(dem)


def parse_links(d):
    """{"2-1": 0.05} -> {(2, 1): 0.05}"""
    return {tuple(int(x) for x in k.split("-")): v for k, v in d.items()}


def compare(distance, rounds, p, defects, shots, est_links=None):
    true_c = make_circuit(distance, rounds, p, defects)
    det, obs = true_c.compile_detector_sampler().sample(
        shots, separate_observables=True)
    decoders = {
        "uniform": matching_from(make_circuit(distance, rounds, p)),
        "oracle": matching_from(true_c),
    }
    if est_links:
        decoders["informed"] = matching_from(
            make_circuit(distance, rounds, p, link_p=est_links))
    out = {}
    for name, m in decoders.items():
        pred = m.decode_batch(det)
        out[name] = float(np.mean(np.any(pred != obs, axis=1)))
    return out


if __name__ == "__main__":
    profile = load_profile()
    est = parse_links(profile.get("link_error_estimates", {}))
    p, shots = 0.02, 200_000
    for k in [1, 2, 5, 10]:
        res = compare(5, 3, p, {(2, 1): k}, shots, est or None)
        line = "  ".join(
            f"{n}={r:.5f}±{np.sqrt(r * (1 - r) / shots):.5f}"
            for n, r in res.items())
        print(f"k={k:>2}: {line}")
    if not est:
        print("(no link_error_estimates in profile: 'informed' skipped)")