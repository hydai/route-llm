# route-llm — Budget Dimension Labeling Prompt (v3, portable)

> Paste this whole file into another tool (Claude Code, Codex, etc.) and attach the
> queries you want labeled. The tool may use its own subagents to process in bulk.
> Its job is to produce **data only** — `route-llm` does the math (`fit-budget`/`eval-budget`) itself.

You are labeling a corpus of user queries with six **reasoning-budget** scores for the
`route-llm` project. `route-llm` is a *predictive* LLM router: it estimates how much
reasoning a query needs so it can recommend an appropriately-powerful model **without
calling one**. Your ratings become training labels, so **consistency matters more than
any single judgment** — apply the rubric the same way every time.

## Rubric — six axes (integers)

| # | axis | range | meaning |
|---|---|---|---|
| 1 | reasoning_depth | 0–4 | layers of reasoning / problem decomposition |
| 2 | verification_difficulty | 0–4 | how hard the answer is to check |
| 3 | constraint_density | 0–4 | how many constraints must hold at once |
| 4 | context_integration | 0–4 | how much context must be integrated |
| 5 | ambiguity | 0–3 | how many reasonable interpretations |
| 6 | error_cost | 0–4 | cost of being wrong |

Judge the **query**, not the topic's fame or your interest in it. `"hi cats"` is all
zeros even though feline biology is deep. A non-English query is rated by its task
difficulty, not by the language.

## Input

JSONL, one object per line — exactly what `route-llm`'s `synth` step emits and what
lives in `data/corpus.jsonl`:

```jsonl
{"query":"hi cats","category":"chat"}
{"query":"Implement a binary search in Rust.","category":"code"}
{"query":"Prove by induction a statement about primes","category":"math"}
```

Categories are: `chat`, `extraction`, `multilingual`, `code`, `math`, `reasoning`.

## Output — STRICT

Emit **JSONL only** — exactly **one line per input line, in the same order**:

```jsonl
{"query":"hi cats","category":"chat","dims":{"reasoning_depth":0,"verification_difficulty":0,"constraint_density":0,"context_integration":0,"ambiguity":0,"error_cost":0}}
{"query":"Implement a binary search in Rust.","category":"code","dims":{"reasoning_depth":2,"verification_difficulty":1,"constraint_density":1,"context_integration":0,"ambiguity":1,"error_cost":1}}
{"query":"Prove by induction a statement about primes","category":"math","dims":{"reasoning_depth":4,"verification_difficulty":4,"constraint_density":1,"context_integration":0,"ambiguity":1,"error_cost":2}}
```

Rules:

- Copy `query` and `category` **byte-for-byte** from the input.
- Each axis is an integer in range (`ambiguity` 0–3; all others 0–4).
- Save the output to **`data/budget.<labeler>.jsonl`** (e.g. `data/budget.claude.jsonl`).
- Output **only** JSONL. No prose, no markdown fences, no commentary, no blank lines.
- **Do not drop, merge, reorder, or invent** queries. N input lines → N output lines.

## Bulk processing (subagents)

If you split the work across subagents/batches, **concatenate the results in the
original input order** and verify the final line count equals the input line count
before returning. A labeled file with a different line count than the corpus will be
rejected downstream.

## Determinism

Treat this as a measurement, not a creative task. Prefer the most consistent ratings
for a given query; do not vary between runs (the reference local-LLM labeler runs at
`temperature=0` — match that spirit). Run each labeler **independently**: cross-labeler
disagreement per axis is the signal v3 wants (`SPEC-v3.md §6.2 / §8`), so don't let
labelers see each other's answers.
