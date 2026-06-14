# route-llm — portable prompt pack

These prompts let you run `route-llm`'s **LLM-native** pipeline stages inside *other*
tools (Claude Code, Codex, a hosted model, …) when those tools can't be embedded as
a service. You paste a prompt, attach the data, the tool (optionally via its own
subagents) returns JSONL, and you feed that JSONL back into this repo.

## The pipeline: who does what

```
synth ──▶ corpus.jsonl ──▶ label ──▶ labeled.jsonl ──▶ fit  ──▶ weights.rs   (ships the router)
                                                   └──▶ eval ──▶ metrics      (the verdict)
   └ LLM-native ┘            └ LLM-native ┘          └────── deterministic ──────┘
```

- **`synth`** and **`label`** depend on model *judgment* → portable prompts live here
  (`synth.prompt.md`, `label.prompt.md`). This is what differs between labelers.
- **`fit`** and **`eval`** are pure math (logistic regression; Spearman/ordinal/cost).
  They stay **in this repo** as the single source of truth. External tools never run
  them — otherwise you couldn't tell whether a metric moved because the *labeler* was
  better or because the *math* differed.

So an external tool's only job is to produce **data files** (queries and/or labels).

## File formats (the contract)

Every line is one JSON object (JSONL), serialized with no spaces.

| File | Stage | Schema | Example line |
|---|---|---|---|
| `data/corpus.jsonl` | `synth` out / `label` in | `{query, category}` | `{"query":"Implement a binary search in Rust.","category":"code"}` |
| `data/label_cache.jsonl` | `label` cache | `{key, rating}` | `{"key":"3f9a…e1","rating":3}` |
| `data/labeled.jsonl` | `label` out / `fit`+`eval` in | `{query, difficulty, category}` | `{"query":"Implement a binary search in Rust.","difficulty":0.5,"category":"code"}` |
| `crates/core/src/learned/weights.rs` | `fit` out | Rust source (`bias` + weights) | *generated; compiled into the core crate* |

Notes:

- `category` ∈ `chat | extraction | multilingual | code | math | reasoning`.
- `difficulty = (rating − 1) / 4` → one of `{0.0, 0.25, 0.5, 0.75, 1.0}`.
- `label_cache.jsonl` stores the **raw 1–5 rating**, keyed by
  `sha256(model · 0x00 · query)`. The model is in the key, so different labelers
  never collide in one cache file.
- `labeled.jsonl` has `#[serde(default)]` on `category` (optional) and **ignores
  unknown fields** — so an external labeler may add `"rating":3` for auditing.

## How to run each stage in another tool

### Expand the corpus (`synth`)

1. Paste `synth.prompt.md`. Set `<N>` (default 300). Optionally attach the current
   `data/corpus.jsonl` so it avoids duplicates.
2. Save the JSONL output to `data/corpus.<source>.jsonl` (e.g. `corpus.claude.jsonl`).
3. Merge + dedupe into the canonical corpus, then commit it (the committed file is
   the frozen corpus — reproducibility comes from committing the artifact, not from
   the generator being deterministic):

   ```sh
   cat data/corpus.jsonl data/corpus.claude.jsonl | sort -u > data/corpus.merged.jsonl
   mv data/corpus.merged.jsonl data/corpus.jsonl
   ```

### Label the corpus (`label`)

1. Paste `label.prompt.md`. Attach `data/corpus.jsonl` as the input.
2. Save the JSONL output to **`data/labeled.<labeler>.jsonl`**:
   - `data/labeled.jsonl` — the canonical (local gemma) set
   - `data/labeled.claude.jsonl` — labeled by Claude Code
   - `data/labeled.codex.jsonl` — labeled by Codex
3. Sanity-check it parses and has the right line count (see below).

## Feeding results back & validating

A foreign labeled set must (a) be valid JSONL, (b) have one line per corpus query,
(c) have `difficulty` ∈ `[0,1]`. `eval` already fails loudly with a line number on
malformed JSON, so the quickest validation is to just run it. Quick local checks:

```sh
wc -l data/corpus.jsonl data/labeled.claude.jsonl     # line counts should match
jq -c 'select(.difficulty < 0 or .difficulty > 1)' data/labeled.claude.jsonl   # must print nothing
```

## fit / eval runbook (stays in this repo)

`eval` is **read-only** (it does its own internal 80/20 split-fit and writes no
artifacts), so it's safe to run per labeler. `fit` **overwrites `weights.rs`** — only
run it on the labeler you decide to ship.

Eval any single labeled set without disturbing the canonical file:

```sh
cargo run -p route-llm-trainer -- eval --in data/labeled.claude.jsonl
```

Compare all labelers side-by-side in one table (the learned router's metrics on
each set's holdout — no file swapping, safe to run while a `label` job is going):

```sh
cargo run -p route-llm-trainer -- compare \
  data/labeled.gemma.jsonl data/labeled.claude.jsonl data/labeled.codex.jsonl
```

To ship a winner:

```sh
cp data/labeled.<winner>.jsonl data/labeled.jsonl
cargo run -p route-llm-trainer -- fit      # regenerates crates/core/src/learned/weights.rs
cargo test -p route-llm-core learned
```

## Gold verdict (v2.2) — a label-independent yardstick

`compare`'s spearman/ordinal are measured against *each labeler's own labels*
(self-consistency, not correctness). v2.2 adds a **human** gold set to break that
self-reference, focusing judgment on the queries that actually discriminate routers.

```sh
# 1. Build the BLIND pool: the queries where claude and codex disagree (~143).
cargo run --release -p route-llm-trainer -- gold-pool      # -> data/gold.unlabeled.jsonl

# 2. A HUMAN hand-labels it (blind — do NOT feed it to a model; that would just
#    add a 4th labeler and re-introduce the bias). Use the label.prompt.md 1–5
#    rubric. Save as data/gold.jsonl, one line per input, SAME order:
#      {"query":...,"difficulty":0.0|0.25|0.5|0.75|1.0,"category":...,"rating":1..5}

# 3. Score every router on the SAME human labels (label-independent verdict):
cargo run --release -p route-llm-trainer -- compare --gold data/gold.jsonl \
  data/labeled.codex.jsonl data/labeled.claude.jsonl data/labeled.gemma.jsonl
cargo run --release -p route-llm-trainer -- eval --gold data/gold.jsonl   # shipped router only

# 4. (diagnostic) label-transfer matrix across labelers — no human, no network:
cargo run --release -p route-llm-trainer -- crosseval
```

The gold set is **hard-cases-only** (chat/extraction have no disagreements), so it
judges ranking quality on contested queries — the real test for learned-vs-heuristic.

## Comparing labelers — a fairness caveat

`eval`'s Spearman/ordinal are measured against **that labeler's own labels** as
truth, so a higher score means "this labeler's signal is more self-consistent /
learnable," **not** "this labeler is more correct." For a fair cross-labeler verdict
you want a shared yardstick — e.g. score every resulting router against one small
hand-checked validation set, or compare the cost/adequacy behavior the routers
produce on a shared query set. The `compare` subcommand does exactly the latter:
it evals each `labeled.*.jsonl` and prints one table, where **`avg_cost` is the
label-independent column** (same holdout queries, different routers) while
`sp_learn`/`ordinal` are each measured against that set's own labels.
