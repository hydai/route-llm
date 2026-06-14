# route-llm v2.1 — Real Local-LLM Difficulty Labeling Implementation Plan

> **Status: ✅ Complete (T1–T7).** Verdict — learned beats heuristic on all three labelers (gemma / claude / codex) across Spearman, ordinal, and cost-at-ceiling; default stays `learned` (no server change). Shipped `weights.rs` fit on codex labels. See `SPEC-v2.1.md` §16. (PR #9)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace v2's category-fixed synthetic difficulty labels with per-query judgments from a local OpenAI-compatible LLM (e.g. LM Studio) over an expanded ~1000-query corpus, then re-fit and re-eval so the verdict can set the default router.

**Architecture:** All changes live in `crates/trainer`. `synth` becomes a deterministic combinatorial query generator (queries-only `corpus.jsonl`); a new `label` step calls a local OpenAI-compatible LLM (e.g. LM Studio) (`localhost`) to rate each query 1–5 → `[0,1]`, writing `labeled.jsonl` with a hash-cache; `fit`/`eval` are unchanged. Network (`reqwest`) is confined to the trainer; inference stays zero-network. v1 and v2's `features`/`model`/`ranker` are frozen — only the embedded `weights.rs` regenerates from better labels.

**Tech Stack:** Rust 2021; `reqwest` (blocking, json) + `sha2` added to the trainer only; local OpenAI-compatible chat API. See `SPEC-v2.1.md`.

---

## Conventions (apply to every task)

- **TDD:** failing test first → watch it fail → minimal implementation → watch it pass → commit.
- **Before each commit:** `cargo fmt --all`, `cargo test` green, `cargo clippy -- -D warnings` clean, `lineguard` on touched files. A PreToolUse hook also auto-runs format/lint/test on `git commit`.
- **Commits:** Conventional Commits; end every message with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- **Branch:** work on `spec/v2.1-real-labeling` (PR #9). Never commit to `master`.
- **Isolation (do NOT modify):** v1 frozen files (`crates/core/src/difficulty.rs`, `ranker.rs`, `registry.rs`, `router.rs`); v2's `crates/core/src/learned/{features.rs, model.rs, mod.rs}`. The only `core` file that changes is the generated `crates/core/src/learned/weights.rs` (via `fit`, Task 6). `crates/core/Cargo.toml` and `crates/server/` are untouched except the optional one-line default flip in `crates/server/src/main.rs` (Task 7).
- **Network rule:** `reqwest` only in `crates/trainer/Cargo.toml`. The OpenAI-compatible chat call appears only in `label.rs`. No network in `cargo test` (the real call is `#[ignore]`d).

## File Structure (decomposition)

```
crates/trainer/
  Cargo.toml                + reqwest {blocking,json}, sha2                 (T5)
  src/
    dataset.rs   (modify)   + CorpusQuery + corpus jsonl load/save         (T1)
    corpus.rs    (modify)   combinatorial build() -> Vec<CorpusQuery>;
                            synth writes queries-only corpus.jsonl ★       (T2)
    label.rs     (new)      parse_rating + rating_to_difficulty (T3),
                            LabelCache (T4), OpenAI-compatible chat client + run() (T5)
    main.rs      (modify)   "label" => label::run(); usage string          (T5)
data/
  corpus.jsonl   (regen, queries-only)                                     (T6)
  labeled.jsonl  (regen by label, real labels)                            (T6)
  label_cache.jsonl (new)                                                  (T6)
crates/core/src/learned/weights.rs  (regen by fit)                         (T6)
crates/server/src/main.rs  (default flip, only if verdict says so)         (T7)
SPEC-v2.1.md §16  (verdict)                                                (T7)
```

★ = learning contribution point (T2 corpus patterns; T5 rubric prompt).

### Spec coverage map

| SPEC-v2.1 section | Task |
|---|---|
| §4 combinatorial synth, queries-only corpus | 1, 2 |
| §5 rubric parse/map | 3 |
| §5 cache | 4 |
| §5 OpenAI-compatible chat client + label run; §10 deps | 5 |
| §6 reproducible data; §3 pipeline | 6 |
| §7 fit/eval; §8 verdict + default flip; §16 | 6, 7 |
| §11 testing | every task |
| §12 backward compat / isolation | conventions + 7 |

---

## Task 1: Corpus query type + I/O

**Files:**
- Modify: `crates/trainer/src/dataset.rs`

- [x] **Step 1: Write the failing test**

Append to the `#[cfg(test)] mod tests` block in `crates/trainer/src/dataset.rs` (add new tests; keep existing ones):
```rust
    #[test]
    fn corpus_query_round_trip() {
        let items = vec![
            CorpusQuery { query: "hi".into(), category: "chat".into() },
            CorpusQuery { query: "prove X".into(), category: "math".into() },
        ];
        let s = to_corpus_jsonl(&items);
        assert_eq!(parse_corpus_jsonl(&s).unwrap(), items);
    }

    #[test]
    fn corpus_query_skips_blank_lines() {
        let s = "{\"query\":\"a\",\"category\":\"chat\"}\n\n";
        assert_eq!(parse_corpus_jsonl(s).unwrap().len(), 1);
    }
```

- [x] **Step 2: Run to verify it fails**

Run: `cargo test -p route-llm-trainer corpus_query`
Expected: FAILS to compile (`CorpusQuery`, `to_corpus_jsonl`, `parse_corpus_jsonl` not found).

- [x] **Step 3: Implement the type + I/O**

Prepend to `crates/trainer/src/dataset.rs` (after the existing `use` line, alongside `LabeledExample`):
```rust
/// A corpus query with no label — what `synth` produces and `label` consumes.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CorpusQuery {
    pub query: String,
    pub category: String,
}

pub fn parse_corpus_jsonl(text: &str) -> Result<Vec<CorpusQuery>, String> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let q: CorpusQuery =
            serde_json::from_str(line).map_err(|e| format!("line {}: {e}", i + 1))?;
        out.push(q);
    }
    Ok(out)
}

pub fn to_corpus_jsonl(items: &[CorpusQuery]) -> String {
    let mut s = items
        .iter()
        .map(|x| serde_json::to_string(x).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    s.push('\n');
    s
}

pub fn load_corpus(path: &str) -> Result<Vec<CorpusQuery>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    parse_corpus_jsonl(&text)
}

pub fn save_corpus(path: &str, items: &[CorpusQuery]) -> Result<(), String> {
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
    }
    std::fs::write(path, to_corpus_jsonl(items)).map_err(|e| format!("write {path}: {e}"))
}
```

- [x] **Step 4: Run to verify it passes**

Run: `cargo test -p route-llm-trainer`
Expected: PASS (existing dataset tests + 2 new corpus tests).

- [x] **Step 5: Commit**

```bash
cargo fmt --all
cargo clippy -- -D warnings
lineguard crates/trainer/src/dataset.rs
git add crates/trainer/src/dataset.rs
git commit -m "feat(trainer): add CorpusQuery type and corpus jsonl I/O"
```

---

## Task 2: Combinatorial synth (queries-only corpus) ★

> ★ The patterns and topic pools are owner-tunable business content (SPEC-v2.1 §4). The reference below is a complete, deterministic generator that reaches ~1000 queries with intra-category spread; expand/improve the pattern/topic lists to raise corpus quality. Keep `build()` returning `Vec<CorpusQuery>` and deterministic.

**Files:**
- Modify: `crates/trainer/src/corpus.rs`

- [x] **Step 1: Replace the corpus generator + tests**

Replace the entire contents of `crates/trainer/src/corpus.rs` with:
```rust
use crate::dataset::{self, CorpusQuery};

/// ★ (category, patterns with a single `{}` slot, topic fills). Owner-tunable.
/// Patterns within a category deliberately range easy→hard; the LLM assigns the
/// actual difficulty at label time, so intra-category spread becomes signal.
fn specs() -> Vec<(&'static str, Vec<&'static str>, Vec<&'static str>)> {
    vec![
        (
            "chat",
            vec![
                "hi {}", "thanks, {}!", "what's up with {}?", "tell me about {}",
                "good morning, any thoughts on {}?", "quick question about {}",
                "how do you feel about {}?", "small talk about {}",
                "say hello and mention {}", "got a minute to chat about {}?",
            ],
            vec![
                "the weather", "your day", "coffee", "weekend plans", "music",
                "movies", "cats", "the news", "lunch", "travel", "books",
                "sports", "the office", "hobbies", "nothing much", "games", "food",
            ],
        ),
        (
            "extraction",
            vec![
                "What is {}?", "Define {} in one sentence.", "Summarize {} briefly.",
                "List three facts about {}.", "Translate '{}' to Spanish.",
                "Extract the key entities from a passage about {}.",
                "Give a one-line summary of {}.", "When did {} happen?",
                "Reformat this note about {} as bullet points.",
                "What does the acronym {} stand for?",
            ],
            vec![
                "photosynthesis", "the Eiffel Tower", "JSON", "the water cycle",
                "World War II", "HTTP", "the stock market", "DNA", "gravity",
                "the internet", "machine learning", "the French Revolution",
                "blockchain", "the solar system", "TCP", "REST", "OAuth",
            ],
        ),
        (
            "multilingual",
            vec![
                "請用一句話說明 {}。", "比較 {} 的優缺點並舉例。",
                "逐步解釋 {} 的運作原理。", "分析 {} 的效能瓶頸並提出優化。",
                "設計一個與 {} 相關的系統並說明取捨。",
                "證明關於 {} 的一個重要性質。", "為什麼 {} 重要?請推導。",
                "用中文比較 {} 的兩種實作方式。",
            ],
            vec![
                "遞迴", "快速排序", "TCP 與 UDP", "梯度下降", "分散式快取",
                "資料庫索引", "一致性雜湊", "垃圾回收", "微服務架構",
                "並行控制", "B+ 樹", "向量時鐘", "共識演算法", "RSA 加密",
            ],
        ),
        (
            "code",
            vec![
                "Fix this typo in {} code.", "Write a hello-world in {}.",
                "Explain what this {} snippet does.",
                "Implement a binary search in {}.",
                "Write unit tests for a {} function.",
                "Implement a thread-safe LRU cache in {}.",
                "Design and implement a rate limiter in {}.",
                "Implement a lock-free concurrent queue in {} and discuss ABA.",
                "Refactor a tangled {} module and justify each change.",
                "Profile and optimize a hot loop in {}.",
            ],
            vec![
                "Rust", "Python", "TypeScript", "Go", "Java", "C++", "Ruby",
                "Kotlin", "Scala", "Swift", "Elixir", "Haskell", "C", "SQL",
            ],
        ),
        (
            "math",
            vec![
                "Compute {} + 7.", "Simplify the expression for {}.",
                "Solve a basic equation involving {}.",
                "Differentiate a function of {}.",
                "Prove a standard identity about {}.",
                "Derive the closed form for {} from first principles.",
                "Prove by induction a statement about {}.",
                "Analyze the convergence of a series involving {}.",
            ],
            vec![
                "x", "a quadratic", "a geometric series", "the sine function",
                "primes", "the harmonic series", "a 2x2 matrix", "logarithms",
                "the binomial coefficients", "an integral of x^2", "eigenvalues",
                "the Fibonacci sequence", "modular arithmetic", "a limit",
            ],
        ),
        (
            "reasoning",
            vec![
                "Briefly: why might {} matter?",
                "Compare two approaches to {}.",
                "Analyze the trade-offs in {} and recommend one.",
                "Prove a key property of {} and derive its complexity.",
                "Design {}, prove its correctness, and analyze failure modes.",
                "Step by step, derive and justify the design of {} under partitions.",
                "Prove the lower bound for {} and design an optimal strategy.",
            ],
            vec![
                "consensus protocols", "the CAP theorem", "A* search",
                "Byzantine fault tolerance", "quicksort's worst case",
                "Dijkstra's algorithm", "two-phase commit", "garbage collection",
                "a distributed lock", "MVCC", "Raft", "a bloom filter",
                "rate limiting at scale", "leader election", "a CRDT",
            ],
        ),
    ]
}

/// Build the corpus deterministically: for each category, every pattern × every
/// topic fill. Stable iteration order → reproducible corpus.jsonl.
pub fn build() -> Vec<CorpusQuery> {
    let mut out = Vec::new();
    for (category, patterns, fills) in specs() {
        for pat in &patterns {
            for fill in &fills {
                out.push(CorpusQuery {
                    query: pat.replace("{}", fill),
                    category: category.to_string(),
                });
            }
        }
    }
    out
}

/// `synth` subcommand: write the queries-only corpus. (No labels — `label` adds those.)
pub fn run() {
    let items = build();
    dataset::save_corpus("data/corpus.jsonl", &items).expect("write data/corpus.jsonl");
    eprintln!("synth: wrote {} queries to data/corpus.jsonl", items.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_reaches_target_size() {
        let n = build().len();
        assert!((900..=1200).contains(&n), "corpus size {n} outside ~1000 target");
    }

    #[test]
    fn deterministic() {
        assert_eq!(build(), build());
    }

    #[test]
    fn every_category_present_and_nonempty() {
        let items = build();
        for cat in ["chat", "extraction", "multilingual", "code", "math", "reasoning"] {
            assert!(items.iter().any(|q| q.category == cat), "missing category {cat}");
        }
    }

    #[test]
    fn queries_are_unique_enough() {
        let items = build();
        let mut q: Vec<&str> = items.iter().map(|x| x.query.as_str()).collect();
        q.sort_unstable();
        q.dedup();
        // No `{}` slots left unfilled.
        assert!(items.iter().all(|x| !x.query.contains("{}")));
        // Mostly-unique queries (combinatorial fill shouldn't collide much).
        assert!(q.len() as f64 > items.len() as f64 * 0.95);
    }
}
```

- [x] **Step 2: Run to verify it fails, then passes**

Run: `cargo test -p route-llm-trainer corpus`
Expected: after the replacement compiles, PASS (4 tests). If it fails to compile because other code referenced the old `corpus::build()` return type (`LabeledExample`), that is expected — Task 5 updates `main.rs`'s `synth` arm (it already calls `corpus::run()`, which still exists, so no change needed). Confirm `main.rs`'s `"synth" => corpus::run()` still compiles.

- [x] **Step 3: Run the whole trainer suite**

Run: `cargo test -p route-llm-trainer`
Expected: PASS. (The old `corpus` tests are replaced; `dataset`, `logreg`, `emit`, `eval` tests unchanged.)

- [x] **Step 4: Commit**

```bash
cargo fmt --all
cargo clippy -- -D warnings
lineguard crates/trainer/src/corpus.rs
git add crates/trainer/src/corpus.rs
git commit -m "feat(trainer): combinatorial synth generating ~1000 queries-only corpus"
```

---

## Task 3: Rubric parse + difficulty mapping (pure)

**Files:**
- Create: `crates/trainer/src/label.rs`
- Modify: `crates/trainer/src/main.rs`

- [x] **Step 1: Write the failing test**

Create `crates/trainer/src/label.rs`:
```rust
//! Local-LLM difficulty labeling for the learned router via an OpenAI-compatible
//! chat API (e.g. LM Studio, Ollama's /v1, llama.cpp server, vLLM). Only this
//! module talks to the network, and only to a local server at request time of
//! the offline `label` step — never in the inference path.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rating_extracts_1_to_5() {
        assert_eq!(parse_rating("RATING: 3\nbecause ..."), Some(3));
        assert_eq!(parse_rating("I'd say rating: 5 (very hard)"), Some(5));
        assert_eq!(parse_rating("1"), Some(1));
        assert_eq!(parse_rating("The difficulty is 4 out of 5."), Some(4));
    }

    #[test]
    fn parse_rating_rejects_out_of_range_or_missing() {
        assert_eq!(parse_rating("RATING: 9"), None);
        assert_eq!(parse_rating("no number here"), None);
        assert_eq!(parse_rating("0"), None);
    }

    #[test]
    fn rating_maps_to_unit_interval() {
        assert_eq!(rating_to_difficulty(1), 0.0);
        assert_eq!(rating_to_difficulty(3), 0.5);
        assert_eq!(rating_to_difficulty(5), 1.0);
    }
}
```

- [x] **Step 2: Run to verify it fails**

Run: `cargo test -p route-llm-trainer label`
Expected: FAILS to compile (`label` module not declared; `parse_rating`/`rating_to_difficulty` missing).

- [x] **Step 3: Implement the pure functions + declare the module**

Prepend to `crates/trainer/src/label.rs` (above the test module):
```rust
/// Extract the first standalone 1–5 integer from model output. Prefers a
/// `rating: N` cue but falls back to the first 1–5 token. Returns None if no
/// valid 1–5 rating is present.
pub fn parse_rating(output: &str) -> Option<u8> {
    let lower = output.to_lowercase();
    // Prefer an explicit "rating: N" / "rating N".
    if let Some(idx) = lower.find("rating") {
        if let Some(n) = first_1_to_5(&output[idx..]) {
            return Some(n);
        }
    }
    first_1_to_5(output)
}

fn first_1_to_5(s: &str) -> Option<u8> {
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_ascii_digit() {
            // Only accept a single-digit 1..=5 not glued to another digit.
            let next_is_digit = chars.peek().map(|c| c.is_ascii_digit()).unwrap_or(false);
            if !next_is_digit {
                if let Some(n) = c.to_digit(10) {
                    if (1..=5).contains(&n) {
                        return Some(n as u8);
                    }
                }
            }
        }
    }
    None
}

/// Map a 1–5 rating to difficulty in {0.0, 0.25, 0.5, 0.75, 1.0}.
pub fn rating_to_difficulty(rating: u8) -> f64 {
    (rating.clamp(1, 5) as f64 - 1.0) / 4.0
}
```

In `crates/trainer/src/main.rs`, add `mod label;` with the other `mod` declarations (below `mod eval;`).

- [x] **Step 4: Run to verify it passes**

Run: `cargo test -p route-llm-trainer label`
Expected: PASS (3 tests).

- [x] **Step 5: Commit**

```bash
cargo fmt --all
cargo clippy -- -D warnings
lineguard crates/trainer/src/label.rs crates/trainer/src/main.rs
git add crates/trainer/src/label.rs crates/trainer/src/main.rs
git commit -m "feat(trainer): add rubric parse and 1-5 to difficulty mapping"
```

---

## Task 4: Label cache

**Files:**
- Modify: `crates/trainer/src/label.rs`
- Modify: `crates/trainer/Cargo.toml` (add `sha2`)

- [x] **Step 1: Add `sha2` to the trainer manifest**

In `crates/trainer/Cargo.toml`, under `[dependencies]`, add:
```toml
sha2 = "0.10"
```

- [x] **Step 2: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `crates/trainer/src/label.rs`:
```rust
    #[test]
    fn cache_key_is_deterministic_and_model_sensitive() {
        let a = cache_key("hi", "m1");
        assert_eq!(a, cache_key("hi", "m1"));
        assert_ne!(a, cache_key("hi", "m2"));
        assert_ne!(a, cache_key("bye", "m1"));
    }

    #[test]
    fn cache_round_trips_and_looks_up() {
        let mut c = LabelCache::default();
        c.insert("k1".into(), 3);
        c.insert("k2".into(), 5);
        let restored = LabelCache::from_jsonl(&c.to_jsonl());
        assert_eq!(restored.get("k1"), Some(3));
        assert_eq!(restored.get("k2"), Some(5));
        assert_eq!(restored.get("missing"), None);
    }
```

- [x] **Step 3: Run to verify it fails**

Run: `cargo test -p route-llm-trainer cache`
Expected: FAILS to compile (`cache_key`, `LabelCache` not found).

- [x] **Step 4: Implement the cache**

Prepend to `crates/trainer/src/label.rs` (above the test module, below the parse functions):
```rust
use std::collections::HashMap;

use sha2::{Digest, Sha256};

/// Stable cache key for a (query, model) pair. Including the model means
/// switching models naturally invalidates old labels.
pub fn cache_key(query: &str, model: &str) -> String {
    let mut h = Sha256::new();
    h.update(model.as_bytes());
    h.update([0u8]);
    h.update(query.as_bytes());
    format!("{:x}", h.finalize())
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct CacheEntry {
    key: String,
    rating: u8,
}

/// In-memory label cache, persisted as jsonl (one CacheEntry per line).
#[derive(Debug, Default)]
pub struct LabelCache {
    map: HashMap<String, u8>,
}

impl LabelCache {
    pub fn get(&self, key: &str) -> Option<u8> {
        self.map.get(key).copied()
    }

    pub fn insert(&mut self, key: String, rating: u8) {
        self.map.insert(key, rating);
    }

    pub fn from_jsonl(text: &str) -> Self {
        let mut map = HashMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(e) = serde_json::from_str::<CacheEntry>(line) {
                map.insert(e.key, e.rating);
            }
        }
        Self { map }
    }

    /// Deterministic (key-sorted) jsonl so the committed cache file is stable.
    pub fn to_jsonl(&self) -> String {
        let mut entries: Vec<(&String, &u8)> = self.map.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        let mut s = entries
            .iter()
            .map(|(k, r)| serde_json::to_string(&CacheEntry { key: (*k).clone(), rating: **r }).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        if !s.is_empty() {
            s.push('\n');
        }
        s
    }

    pub fn load(path: &str) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::from_jsonl(&text),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        if let Some(dir) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
        }
        std::fs::write(path, self.to_jsonl()).map_err(|e| format!("write {path}: {e}"))
    }
}
```

- [x] **Step 5: Run to verify it passes**

Run: `cargo test -p route-llm-trainer`
Expected: PASS (label parse + cache tests + existing suite).

- [x] **Step 6: Commit**

```bash
cargo fmt --all
cargo clippy -- -D warnings
lineguard crates/trainer/src/label.rs crates/trainer/Cargo.toml
git add crates/trainer/src/label.rs crates/trainer/Cargo.toml
git commit -m "feat(trainer): add hash-keyed label cache with stable jsonl"
```

---

## Task 5: OpenAI-compatible chat client + `label` run

**Files:**
- Modify: `crates/trainer/src/label.rs`
- Modify: `crates/trainer/Cargo.toml` (add `reqwest`)
- Modify: `crates/trainer/src/main.rs` (wire `label` arm + usage)

- [x] **Step 1: Add `reqwest` to the trainer manifest**

In `crates/trainer/Cargo.toml`, under `[dependencies]`, add:
```toml
reqwest = { version = "0.12", default-features = false, features = ["blocking", "json", "rustls-tls"] }
```
(`reqwest` lives ONLY in the trainer — `core`/`server` gain no network deps.)

- [x] **Step 2: Write the failing test (config + an ignored integration test)**

Add to the `#[cfg(test)] mod tests` block in `crates/trainer/src/label.rs`:
```rust
    #[test]
    fn config_defaults_match_spec() {
        // With env unset, defaults come from the spec.
        std::env::remove_var("ROUTE_LLM_LABEL_URL");
        std::env::remove_var("ROUTE_LLM_LABEL_MODEL");
        let cfg = LabelConfig::from_env();
        assert_eq!(cfg.url, "http://localhost:1234/v1");
        assert_eq!(cfg.model, "google/gemma-4-31b-qat");
    }

    #[test]
    fn prompt_includes_query_and_scale() {
        let p = build_prompt("reverse a linked list");
        assert!(p.contains("reverse a linked list"));
        assert!(p.contains("1-5") || p.contains("1–5"));
        assert!(p.to_lowercase().contains("rating"));
    }

    #[test]
    #[ignore = "requires a running local OpenAI-compatible LLM server (e.g. LM Studio) with the model loaded"]
    fn chat_complete_round_trip_smoke() {
        // Run manually: `cargo test -p route-llm-trainer -- --ignored chat_complete`
        let cfg = LabelConfig::from_env();
        let out = chat_complete(&cfg, &build_prompt("hi")).expect("chat completion call");
        assert!(parse_rating(&out).is_some(), "expected a 1-5 rating, got: {out}");
    }
```

- [x] **Step 3: Run to verify it fails**

Run: `cargo test -p route-llm-trainer config_defaults`
Expected: FAILS to compile (`LabelConfig`, `build_prompt`, `chat_complete` not found).

- [x] **Step 4: Implement the client, prompt, and `run()`**

Prepend to `crates/trainer/src/label.rs` (above the test module):
```rust
use crate::dataset::{self, LabeledExample};

/// Where to reach the local model. Defaults per SPEC-v2.1 §5.
pub struct LabelConfig {
    pub url: String,
    pub model: String,
}

impl LabelConfig {
    pub fn from_env() -> Self {
        Self {
            url: std::env::var("ROUTE_LLM_LABEL_URL")
                .unwrap_or_else(|_| "http://localhost:1234/v1".to_string()),
            model: std::env::var("ROUTE_LLM_LABEL_MODEL")
                .unwrap_or_else(|_| "google/gemma-4-31b-qat".to_string()),
        }
    }
}

/// ★ Rubric prompt (owner-tunable, SPEC-v2.1 §5). Asks for `RATING: <n>` + reason.
pub fn build_prompt(query: &str) -> String {
    format!(
        "You are rating how hard a user query is for an LLM to answer *well*.\n\
         Use a 1-5 scale:\n\
         1 = trivial chat/greeting\n\
         2 = simple lookup or extraction\n\
         3 = moderate (some reasoning or code)\n\
         4 = hard, multi-step reasoning or non-trivial implementation\n\
         5 = expert: rigorous proof, deep analysis, or intricate system design\n\
         Reply with EXACTLY one line `RATING: <n>` (n in 1..5), then one short reason line.\n\n\
         Query: {query}\n"
    )
}

/// One OpenAI-compatible chat completion → assistant message text. NETWORK; not
/// unit-tested. Works with LM Studio (default), Ollama's /v1, llama.cpp, vLLM.
fn chat_complete(cfg: &LabelConfig, prompt: &str) -> Result<String, String> {
    let body = serde_json::json!({
        "model": cfg.model,
        "messages": [{ "role": "user", "content": prompt }],
        "temperature": 0,
        "stream": false
    });
    let resp = reqwest::blocking::Client::new()
        .post(format!("{}/chat/completions", cfg.url))
        .json(&body)
        .send()
        .map_err(|e| {
            format!(
                "LLM request to {}/chat/completions failed: {e}\n\
                 Is your OpenAI-compatible server running (e.g. LM Studio at {}) with `{}` loaded?",
                cfg.url, cfg.url, cfg.model
            )
        })?;
    let v: serde_json::Value = resp
        .json()
        .map_err(|e| format!("LLM returned bad JSON: {e}"))?;
    v["choices"][0]["message"]["content"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| format!("LLM response missing choices[0].message.content: {v}"))
}

/// Label one query: cache hit → reuse; else call the LLM (retry once) → parse.
/// `Ok(Some(r))` on success, `Ok(None)` if the output can't be parsed after a
/// retry (skip), and `Err(e)` if the server is unreachable (a network failure
/// won't fix on retry within the same run, so abort).
fn label_one(cfg: &LabelConfig, cache: &mut LabelCache, query: &str) -> Result<Option<u8>, String> {
    let key = cache_key(query, &cfg.model);
    if let Some(r) = cache.get(&key) {
        return Ok(Some(r));
    }
    for _ in 0..2 {
        match chat_complete(cfg, &build_prompt(query)) {
            Ok(out) => {
                if let Some(r) = parse_rating(&out) {
                    cache.insert(key.clone(), r);
                    return Ok(Some(r));
                }
            }
            Err(e) => return Err(e),
        }
    }
    Ok(None)
}

/// `label` subcommand: read corpus.jsonl, label each query via the local
/// OpenAI-compatible LLM, write labeled.jsonl (+ persist the cache). The only
/// networked step.
pub fn run() {
    let cfg = LabelConfig::from_env();
    let corpus = dataset::load_corpus("data/corpus.jsonl")
        .expect("load data/corpus.jsonl (run `trainer synth` first)");
    let mut cache = LabelCache::load("data/label_cache.jsonl");
    let mut labeled: Vec<LabeledExample> = Vec::new();
    let (mut ok, mut skipped) = (0usize, 0usize);

    for (i, q) in corpus.iter().enumerate() {
        match label_one(&cfg, &mut cache, &q.query) {
            Ok(Some(r)) => {
                labeled.push(LabeledExample {
                    query: q.query.clone(),
                    difficulty: rating_to_difficulty(r),
                    category: q.category.clone(),
                });
                ok += 1;
            }
            Ok(None) => {
                eprintln!("skip (unparseable) [{i}]: {}", q.query);
                skipped += 1;
            }
            Err(e) => {
                let _ = cache.save("data/label_cache.jsonl");
                eprintln!("\nlabel aborted after {ok} labeled / {skipped} skipped: {e}");
                std::process::exit(1);
            }
        }
        if (i + 1) % 50 == 0 {
            eprintln!("labeled {}/{} (model={})", i + 1, corpus.len(), cfg.model);
            let _ = cache.save("data/label_cache.jsonl"); // periodic checkpoint
        }
    }

    cache
        .save("data/label_cache.jsonl")
        .expect("save label cache");
    dataset::save("data/labeled.jsonl", &labeled).expect("save data/labeled.jsonl");
    eprintln!("label: {ok} labeled, {skipped} skipped -> data/labeled.jsonl");
}
```

In `crates/trainer/src/main.rs`, replace the `"label"` arm and the usage string:
```rust
        "label" => label::run(),
```
```rust
                eprintln!("usage: trainer <synth|label|fit|eval>");
```

- [x] **Step 5: Run to verify it passes (pure tests only; ignored test skipped)**

Run: `cargo test -p route-llm-trainer`
Expected: PASS. The `chat_complete_round_trip_smoke` test is `#[ignore]`d (no server in CI). `cargo clippy -- -D warnings` clean.

- [x] **Step 6: Commit**

```bash
cargo fmt --all
cargo clippy -- -D warnings
lineguard crates/trainer/src/label.rs crates/trainer/src/main.rs crates/trainer/Cargo.toml
git add crates/trainer/src/label.rs crates/trainer/src/main.rs crates/trainer/Cargo.toml
git commit -m "feat(trainer): add OpenAI-compatible chat client and label subcommand"
```

---

## Task 6: Run the pipeline for real (REQUIRES a local OpenAI-compatible LLM server, e.g. LM Studio)

> This task needs a running local OpenAI-compatible LLM server (e.g. LM Studio) with `google/gemma-4-31b-qat` loaded, and labels ~1000 queries (minutes–hours, cached). Run it on a machine with the LLM server. The committed `labeled.jsonl` + `weights.rs` are the reproducible artifacts; CI/`fit` never need the LLM server.

**Files:**
- Create/regenerate: `data/corpus.jsonl`, `data/labeled.jsonl`, `data/label_cache.jsonl`
- Regenerate: `crates/core/src/learned/weights.rs`

- [x] **Step 1: Prepare the local LLM server (LM Studio)**

```bash
# In LM Studio: download/load `google/gemma-4-31b-qat`, then start the local server
# (Developer tab → Start Server; default http://localhost:1234).
# Override if needed: export ROUTE_LLM_LABEL_URL / ROUTE_LLM_LABEL_MODEL
# (model id must match what GET http://localhost:1234/v1/models reports)
```
Expected: model available locally.

- [x] **Step 2: Generate the corpus**

Run: `cargo run -p route-llm-trainer -- synth`
Expected: writes `data/corpus.jsonl` with ~1000 queries.

- [x] **Step 3: Label (the slow, networked, local step)**

Run: `cargo run -p route-llm-trainer -- label`
Expected: progress every 50; writes `data/labeled.jsonl` + `data/label_cache.jsonl`. Re-running resumes from cache. If many lines say "skip (unparseable)", tune the prompt in `build_prompt` (★) and re-run.

- [x] **Step 4: Eval (record the verdict numbers BEFORE committing weights)**

Run: `cargo run -p route-llm-trainer -- eval`
Expected: prints learned vs heuristic vs always-strongest (Spearman, ordinal, cost@adequacy). Copy this output for Task 7.

- [x] **Step 5: Fit and format**

```bash
cargo run -p route-llm-trainer -- fit
cargo fmt --all
cargo test -p route-llm-core learned
```
Expected: regenerates `crates/core/src/learned/weights.rs`; `trivial_query_is_easier_than_hard_query` still passes with the trained weights. (If it fails, the labels/corpus need work — tune and re-run.)

- [x] **Step 6: Full suite + commit artifacts**

```bash
cargo test
cargo clippy -- -D warnings
lineguard data/corpus.jsonl data/labeled.jsonl data/label_cache.jsonl crates/core/src/learned/weights.rs
git add data/corpus.jsonl data/labeled.jsonl data/label_cache.jsonl crates/core/src/learned/weights.rs
git commit -m "feat(core): retrain learned weights on real local-LLM difficulty labels"
```

---

## Task 7: Record verdict + set default router

**Files:**
- Modify: `SPEC-v2.1.md` (§16)
- Modify (only if learned lost): `crates/server/src/main.rs`

- [x] **Step 1: Decide the verdict from Task 6 Step 4**

Apply SPEC-v2.1 §8: learned **wins** iff `Spearman(learned) ≥ Spearman(heuristic)` AND `ordinal(learned) ≥ ordinal(heuristic)` AND cost(learned) not worse than cost(heuristic) at fixed adequacy.

- [x] **Step 2: If learned LOST — flip the deployment default to heuristic**

In `crates/server/src/main.rs`, in `choose_router`, change the unset/default arm from `learned` to `heuristic`:
```rust
        // Genuinely unset → default to heuristic (v2.1 verdict: learned did not beat it)
        Err(std::env::VarError::NotPresent) => Ok("heuristic"),
```
Then update the `unset_defaults_to_learned` test to `unset_defaults_to_heuristic` asserting `Ok("heuristic")`, and run `cargo test -p route-llm-server`.

> If learned WON, skip this step — the default stays `learned`, now justified by evidence.

- [x] **Step 3: Record the verdict in the spec**

Replace SPEC-v2.1 §16's placeholder with the actual numbers and decision, e.g.:
```markdown
## 16. 驗收結論

eval（holdout n=…）：
- spearman   learned=… heuristic=…
- ordinal    learned=… heuristic=…
- cost@adeq  learned=… heuristic=… always-strongest=…

判定：learned <勝出 / 未勝出>。預設 router = <learned / heuristic>。
```

- [x] **Step 4: Commit**

```bash
cargo fmt --all
cargo clippy -- -D warnings
lineguard SPEC-v2.1.md crates/server/src/main.rs
git add SPEC-v2.1.md crates/server/src/main.rs
git commit -m "docs: record v2.1 eval verdict and set default router accordingly"
```

---

## Final verification checklist

- [x] `cargo build --release` — clean.
- [x] `cargo test` — all crates green (the LLM smoke test stays `#[ignore]`d).
- [x] `cargo clippy -- -D warnings` — clean.
- [x] v1 frozen files and v2 `learned/{features,model,mod}.rs` **unmodified**; only `weights.rs` changed in `core`.
- [x] `reqwest` appears **only** in `crates/trainer/Cargo.toml`; `core`/`server` have no network deps.
- [x] `data/labeled.jsonl` is committed; `fit` is deterministic on it (no LLM server needed for `fit`/CI).
- [x] `eval` verdict recorded in SPEC-v2.1 §16; default router matches the verdict.
- [x] Inference performs no network I/O.
