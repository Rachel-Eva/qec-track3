import stim, pymatching, numpy as np

c = stim.Circuit.generated("repetition_code:memory",
                           distance=3, rounds=3,
                           before_round_data_depolarization=0.05)
dem = c.detector_error_model(decompose_errors=True)
m = pymatching.Matching.from_detector_error_model(dem)

det, obs = c.compile_detector_sampler().sample(5000, separate_observables=True)
pred = m.decode_batch(det)
print("logical error rate:", np.mean(np.any(pred != obs, axis=1)))

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2); qc.h(0); qc.cx(0, 1); qc.measure([0, 1], [0, 1])
print(AerSimulator().run(qc, shots=100).result().get_counts())