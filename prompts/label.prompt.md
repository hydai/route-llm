# route-llm — Difficulty Labeling Prompt (portable)

> Paste this whole file into another tool (Claude Code, Codex, etc.) and attach the
> queries you want labeled. The tool may use its own subagents to process in bulk.
> Its job is to produce **data only** — `route-llm` does the math (`fit`/`eval`) itself.

You are labeling a corpus of user queries with a **difficulty** score for the
`route-llm` project. `route-llm` is a *predictive* LLM router: given a query, it
guesses how hard the query is so it can recommend an appropriately-powerful model
**without calling one**. Your ratings become training labels, so **consistency
matters more than any single judgment** — apply the rubric the same way every time.

## Rubric (1–5)

Rate how hard each query is for an LLM to answer *well*:

- **1** = trivial chat / greeting
- **2** = simple lookup or extraction
- **3** = moderate (some reasoning or code)
- **4** = hard, multi-step reasoning or non-trivial implementation
- **5** = expert: rigorous proof, deep analysis, or intricate system design

Judge the **query**, not the topic's fame or your interest in it. `"hi cats"` is a
**1** even though feline biology is deep. `"Prove by induction a statement about
primes"` is a **5**. A non-English query is rated by its task difficulty, not by the
language.

## Map rating → difficulty

`difficulty = (rating − 1) / 4`:

| rating | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| difficulty | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |

## Input

JSONL, one object per line — exactly what `route-llm`'s `synth` step emits:

```jsonl
{"query":"hi cats","category":"chat"}
{"query":"Implement a binary search in Rust.","category":"code"}
{"query":"Prove by induction a statement about primes","category":"math"}
```

Categories are: `chat`, `extraction`, `multilingual`, `code`, `math`, `reasoning`.

## Output — STRICT

Emit **JSONL only** — exactly **one line per input line, in the same order**:

```jsonl
{"query":"hi cats","difficulty":0.0,"category":"chat","rating":1}
{"query":"Implement a binary search in Rust.","difficulty":0.5,"category":"code","rating":3}
{"query":"Prove by induction a statement about primes","difficulty":1.0,"category":"math","rating":5}
```

Rules:

- Copy `query` and `category` **byte-for-byte** from the input.
- `difficulty` MUST be one of `0.0 | 0.25 | 0.5 | 0.75 | 1.0`.
- `rating` (1–5) is optional but **include it** — `route-llm` ignores unknown fields,
  and having the raw rating lets you diff labelers directly.
- Output **only** JSONL. No prose, no markdown fences, no commentary, no blank lines.
- **Do not drop, merge, reorder, or invent** queries. N input lines → N output lines.

## Bulk processing (subagents)

If you split the work across subagents/batches, **concatenate the results in the
original input order** and verify the final line count equals the input line count
before returning. A labeled file with a different line count than the corpus will be
rejected downstream.

## Determinism

Treat this as a measurement, not a creative task. Prefer the most consistent rating
for a given query; do not vary between runs. (The reference local-LLM labeler runs at
`temperature=0` — match that spirit.)
