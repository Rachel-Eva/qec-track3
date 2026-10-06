# Shared data schema

Contract between Person A (ML, triage, dashboard) and Person B (quantum, decoder).
Change a field name here first, then in code. Draft: confirm with Person A.

## Naming decisions

- Logical error metric: `logical_errors` (a rate in 0-1, lower is better). Do not use `logical_fidelities`.
- Probabilities are fractions (0.01 = 1%), never percentages.
- Qubit and link IDs: integers; links as `"q_a-q_b"` strings, e.g. `"1-2"`.

## `shared/syndrome_telemetry.json` (Person B writes, Person A reads)

```json
{
  "source": "synthetic | aer | ibm_hardware",
  "backend": "name or null",
  "code": {"distance": 3, "rounds": 3},
  "shots": 3000,
  "planted_defect": {"link": "1-2", "k": 5},
  "syndromes": [[0, 1, 0, 0, 1, 0]],
  "logical_outcomes": [0],
  "logical_errors": {
    "unprotected": 0.0,
    "uniform": 0.0,
    "informed": 0.0,
    "oracle": 0.0
  }
}
```

- `planted_defect` is `null` when no defect is planted. Informed decoding must NOT read it.
- `syndromes` holds raw per-shot detector bits so shots can be re-decoded later.

## `shared/hardware_profile.json` (Person A writes, Person B reads)

```json
{
  "p_phys_estimate": 0.01,
  "link_error_estimates": {"0-1": 0.01, "1-2": 0.03},
  "noise_bias": "bitflip | dephasing | balanced"
}
```

- Informed decoder weights come from `link_error_estimates`.

## Open questions

- [ ] Confirm field names with Person A.
- [ ] Confirm syndrome bit ordering once the circuit is final.