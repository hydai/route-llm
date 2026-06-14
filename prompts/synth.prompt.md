# route-llm — Corpus Synthesis Prompt (portable)

> Paste this into another tool to generate **new** corpus queries. Output is
> queries-only (no labels) — labeling is a separate step (`label.prompt.md`).
> The tool may use its own subagents to generate in bulk.

Generate **new** user queries to expand the `route-llm` training corpus.
`route-llm` is a *predictive* LLM router that guesses a query's difficulty to
recommend a model. For the learned model to have signal, the corpus must span a
**wide difficulty range within each category** — not just across categories.

## Categories (use all six)

- **chat** — greetings, small talk, opinions
- **extraction** — lookups, definitions, summaries, translations, entity pulls
- **multilingual** — non-English queries (especially Traditional Chinese), mixed difficulty
- **code** — from typo-fixes / hello-world up to lock-free data structures, profiling, refactoring
- **math** — from arithmetic up to proofs, convergence analysis, derivations
- **reasoning** — from one-liners up to "design X, prove correctness, analyze failure modes"

## Difficulty spread (critical)

Within **each** category, include the full easy→hard range:

- some **trivial** (would map to rating 1–2),
- some **moderate** (rating 3),
- some **hard/expert** (rating 4–5).

Do **not** make any category uniformly easy or uniformly hard. The intra-category
spread is the whole point — it's what the labeler turns into signal.

## Quantity & balance

Produce **<N>** queries (default **300**), roughly balanced across the six
categories (~50 each). Make them **diverse and natural** — vary phrasing, topics,
and length. Avoid template-y near-duplicates (don't just swap one noun).

## Avoid duplicates

If an existing `corpus.jsonl` is attached, do **not** repeat queries already present
or trivial paraphrases of them. Aim for genuinely new phrasings and topics.

## Output — STRICT

Emit **JSONL only**, one object per line — nothing else (no prose, no fences):

```jsonl
{"query":"got a minute to chat about board games?","category":"chat"}
{"query":"Implement a wait-free single-producer single-consumer ring buffer in C and explain the memory ordering.","category":"code"}
{"query":"請證明任意連通無環圖的邊數等於節點數減一。","category":"reasoning"}
```

Rules:

- `query` = the literal text a user would type.
- `category` = exactly one of `chat | extraction | multilingual | code | math | reasoning`.
- **No labels / difficulty here** — labeling is a separate step.
