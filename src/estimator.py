"""Syndrome-derived error estimation + fault isolation.

1. features(): detection-event statistics (node marginals + pairwise edge probabilities p_ij,
   the correlation estimator from the Google Quantum AI repetition-code work -- verify the
   citation/formula before presenting).
2. build_dictionary(): model-based fault dictionary, one column per component, obtained by
   simulating the nominal circuit with that component degraded (classic fault-isolation idea).
3. isolate(): non-negative least squares of observed feature excess onto the dictionary,
   bootstrap standard errors, Bonferroni-corrected z test.
"""
import numpy as np
from scipy.optimize import nnls
from scipy.stats import norm
import sim


def _pij(xi, xj, xij):
    num = xij - xi * xj
    den = 1 - 2 * xi - 2 * xj + 4 * xij
    return 0.5 - 0.5 * np.sqrt(np.clip(1 - 4 * num / den, 0, None))


def features(ev):
    """ev: (shots, R+1, n) detection events -> feature vector (bulk layers only)."""
    S, L, n = ev.shape
    B = ev[:, 1:L - 1, :].astype(np.float32)           # bulk layers, shape (S, R-1, n)
    m = B.mean((0, 1))
    f = [m]
    sp = []
    for a in range(n - 1):                              # space edge: data qubit a+1
        sp.append(_pij(m[a], m[a + 1], (B[:, :, a] * B[:, :, a + 1]).mean()))
    f.append(np.array(sp))
    tm = []
    for a in range(n):                                  # time edge: ancilla a
        x, y = B[:, :-1, a], B[:, 1:, a]
        tm.append(_pij(x.mean(), y.mean(), (x * y).mean()))
    f.append(np.array(tm))
    d1, d2 = [], []
    for a in range(n - 1):                              # diagonals (CX hook errors)
        x, y = B[:, :-1, a], B[:, 1:, a + 1]
        d1.append(_pij(x.mean(), y.mean(), (x * y).mean()))
        x, y = B[:, :-1, a + 1], B[:, 1:, a]
        d2.append(_pij(x.mean(), y.mean(), (x * y).mean()))
    f += [np.array(d1), np.array(d2)]
    return np.concatenate(f)


def build_dictionary(d, rounds, shots, mult=10.0, ref_ev=None, seed=100):
    comps = sim.components(d)
    if ref_ev is None:
        ref_ev, _ = sim.run_detection_events(d, rounds, shots, seed=seed)
    f0 = features(ref_ev)
    cols = []
    for i, c in enumerate(comps):
        ev, _ = sim.run_detection_events(d, rounds, shots, defects={c: mult}, seed=seed + 1 + i)
        cols.append((features(ev) - f0) / (mult - 1.0))
    return comps, np.array(cols).T, f0


def isolate(ev, ref_f, dict_mat, comps, n_boot=40, alpha=0.01, min_excess=0.8, seed=0):
    """Returns per-component weight (estimated multiplier-1), bootstrap se, z, flagged."""
    rng = np.random.default_rng(seed)
    S = ev.shape[0]

    def fit(e):
        delta = features(e) - ref_f
        w, _ = nnls(dict_mat, delta)
        return w

    w = fit(ev)
    boots = np.array([fit(ev[rng.integers(0, S, S)]) for _ in range(n_boot)])
    se = boots.std(0) + 1e-9
    z = w / se
    zcrit = norm.isf(alpha / len(comps))                # Bonferroni
    flagged = (z > zcrit) & (w > min_excess)
    return dict(w=w, se=se, z=z, flagged=flagged, comps=comps, zcrit=zcrit)
