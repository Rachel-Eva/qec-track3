import sys, numpy as np
sys.path.insert(0, "src")
import sim, estimator as E
d, R, SHOTS = 5, 8, 16000
ref_ev, _ = sim.run_detection_events(d, R, 32000, seed=7)
comps, M, f0 = E.build_dictionary(d, R, SHOTS, mult=10.0, ref_ev=ref_ev, seed=100)
np.savez("shared/dictionary_d5.npz", comps=np.array(comps), M=M, ref_f=E.features(ref_ev))
print("saved", M.shape)
