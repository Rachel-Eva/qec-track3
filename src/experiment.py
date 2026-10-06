"""Caught / missed / wrongly-blamed sweeps against planted defects (simulator)."""
import sys, json, itertools, numpy as np
sys.path.insert(0, "src")
import sim, estimator as E

D, R = 5, 8


def load():
    z = np.load("shared/dictionary_d5.npz")
    return list(z["comps"]), z["M"], z["ref_f"]


def trial(planted, shots, seed, comps, M, ref_f):
    ev, _ = sim.run_detection_events(D, R, shots, defects=planted, seed=seed)
    out = E.isolate(ev, ref_f, M, comps, n_boot=30, seed=seed)
    flagged = {c for c, fl in zip(comps, out["flagged"]) if fl}
    truth = set(planted)
    return dict(caught=len(truth & flagged), planted=len(truth), wrong=len(flagged - truth),
                flagged=sorted(flagged), top=[comps[i] for i in np.argsort(-out["z"])[:3]])


def sweep(cases, shots_list, reps, comps, M, ref_f, tag):
    rows = []
    for name, planted in cases:
        for shots in shots_list:
            res = [trial(planted, shots, 1000 * r + 17, comps, M, ref_f) for r in range(reps)]
            rows.append(dict(case=name, planted=planted, shots=shots,
                             caught_rate=float(np.mean([r["caught"] / max(1, r["planted"]) for r in res])),
                             full_detect_rate=float(np.mean([r["caught"] == r["planted"] for r in res])),
                             wrong_per_run=float(np.mean([r["wrong"] for r in res])),
                             examples=res[0]["flagged"]))
            print(rows[-1]["case"], shots, "caught", round(rows[-1]["caught_rate"], 2),
                  "wrong/run", round(rows[-1]["wrong_per_run"], 2), flush=True)
    json.dump(rows, open(f"shared/results_{tag}.json", "w"), indent=1)
    return rows
