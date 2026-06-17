# route-llm

A Rust HTTP service that **predicts a recommended ordering of candidate LLMs** for a
given query — without calling any LLM. Inspired by [RouteLLM](https://github.com/lm-sys/RouteLLM):
it scores query difficulty heuristically and ranks models by a cost-quality tradeoff.

See `SPEC.md` for the design and `PLAN.md` for the build steps.

## Run

```bash
cargo run --release -p route-llm-server
# listens on http://0.0.0.0:8080 (override with ROUTE_LLM_HOST / ROUTE_LLM_PORT)
```

## Endpoints

- `GET /health` — liveness.
- `GET /v1/models` — list the builtin model registry.
- `POST /v1/recommend` — native: `{query, models, preferences}` → `{difficulty, ranking}`.
- `POST /v1/chat/completions` — OpenAI-shaped (candidate list via `models` extra field);
  returns a `chat.completion` envelope whose `model` is the top pick and `route_llm` holds
  the full ranking.
- `POST /v1/messages` — Anthropic-shaped equivalent (`type: "message"` envelope).

### Example

```bash
curl -s localhost:8080/v1/recommend -H 'content-type: application/json' -d '{
  "query": "Summarize this text in one sentence.",
  "models": [{"id": "claude-haiku-4-5"}, {"id": "claude-opus-4-8"}],
  "preferences": {"cost_bias": 0.5}
}'
```

## Test

```bash
cargo test
```

## v2: Learned router

`route-llm` ships two routing strategies behind the same API:

- `heuristic` — v1's hand-tuned difficulty scorer (frozen).
- `learned` — a logistic model over richer features, trained offline (default).

Select at startup:

```bash
ROUTE_LLM_ROUTER=learned   cargo run --release -p route-llm-server   # default
ROUTE_LLM_ROUTER=heuristic cargo run --release -p route-llm-server   # v1 fallback
ROUTE_LLM_ROUTER=budget    cargo run --release -p route-llm-server   # v3 Reasoning Budget Router
# Budget policy (only for the budget router): balanced (default) | strict | cheap
ROUTE_LLM_ROUTER=budget ROUTE_LLM_POLICY=strict cargo run --release -p route-llm-server
```

### Retraining (offline)

```bash
cargo run -p route-llm-trainer -- synth   # regenerate data/*.jsonl
cargo run -p route-llm-trainer -- eval    # learned vs heuristic on a holdout
cargo run -p route-llm-trainer -- fit     # regenerate crates/core/src/learned/weights.rs
```

Inference never calls an LLM or the network; only the (deferred) `label` step would.
