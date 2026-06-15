# Budget dimension labeling (v3)

Rate each query on SIX independent axes. Output one integer per axis.

| # | axis | range | meaning |
|---|---|---|---|
| 1 | reasoning_depth | 0–4 | layers of reasoning / problem decomposition |
| 2 | verification_difficulty | 0–4 | how hard the answer is to check |
| 3 | constraint_density | 0–4 | how many constraints must hold at once |
| 4 | context_integration | 0–4 | how much context must be integrated |
| 5 | ambiguity | 0–3 | how many reasonable interpretations |
| 6 | error_cost | 0–4 | cost of being wrong |

Reply with EXACTLY one line, then one short reason line:

```
DIMS: <reasoning_depth> <verification_difficulty> <constraint_density> <context_integration> <ambiguity> <error_cost>
```

Rules:

- Judge the QUERY only — never see or run a model answer.
- One call returns all six integers (not six calls).
- Output file: `data/budget.<labeler>.jsonl`, one line per query:
  `{"query":"…","category":"…","dims":{"reasoning_depth":3,"verification_difficulty":2,"constraint_density":2,"context_integration":1,"ambiguity":1,"error_cost":2}}`
- Use ≥2 frontier labelers (e.g. claude, codex, gemma). Cross-labeler disagreement
  per axis is a signal, not noise (SPEC-v3 §6.2 / §8 axis B).

The six axes, scales, and weights match `SPEC-v3.md §4.1` and the in-code rubric in
`crates/trainer/src/budget_label.rs` (`build_dims_prompt`). The map from these integers
to a routing decision (budget_score → R0–R4 → tier) is `SPEC-v3.md §4`–`§6`.
