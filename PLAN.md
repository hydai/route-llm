# route-llm v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Rust HTTP service that predicts (without calling) a recommended ordering of candidate LLMs for a given query, using a RouteLLM-inspired heuristic difficulty router.

**Architecture:** A Cargo workspace splits pure routing logic (`route-llm-core`, no I/O, fully unit-testable) from the HTTP layer (`route-llm-server`, axum). Three request dialects (native, OpenAI-shaped, Anthropic-shaped) each extract a query + candidate list, then call one shared `Router::recommend`. The router scores query difficulty heuristically and ranks models by a cost-quality tradeoff `adequacy(m) − λ·cost(m)`.

**Tech Stack:** Rust (edition 2021), axum 0.7, tokio, serde, thiserror, tracing; axum-test for HTTP integration tests.

---

## Conventions (apply to every task)

- **TDD:** write the failing test first, watch it fail, write the minimal implementation, watch it pass.
- **Before each commit:** run `cargo fmt --all`, ensure `cargo test` is green, and run `lineguard` on touched files. A PreToolUse hook also auto-runs format/lint/test on `git commit`.
- **Commits:** Conventional Commits. End each commit message with the trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- **Build:** prefer `cargo build --release` for manual runs; tests use plain `cargo test`.
- Floating-point assertions use ranges/ordering, never exact equality (difficulty/score values are tuning-dependent; see SPEC §9 note).

## File Structure (decomposition)

```
route-llm/
├── Cargo.toml                         # workspace + dev debug profile        (Task 1)
├── .gitignore                                                                (Task 1)
├── crates/
│   ├── core/
│   │   ├── Cargo.toml                                                        (Task 1)
│   │   └── src/
│   │       ├── lib.rs                 # module decls + re-exports     (Task 2,3,4,5,6)
│   │       ├── model.rs               # domain types                        (Task 2)
│   │       ├── difficulty.rs          # sigmoid + heuristic scorer ★         (Task 3)
│   │       ├── registry.rs            # builtin table + resolve()           (Task 4)
│   │       ├── ranker.rs              # cost-quality ranking ★              (Task 5)
│   │       └── router.rs              # Router trait + HeuristicRouter      (Task 6)
│   └── server/
│       ├── Cargo.toml                                                        (Task 1)
│       ├── src/
│       │   ├── main.rs                # bin: env config + axum serve        (Task 7)
│       │   ├── lib.rs                 # app() builder + module decls   (Task 7,8,9,10,11)
│       │   ├── error.rs               # ApiError + IntoResponse             (Task 8)
│       │   ├── dto.rs                 # request/response serde structs (Task 8,10,11)
│       │   └── handlers.rs            # process() + per-dialect handlers (Task 8,9,10,11)
│       └── tests/
│           ├── health.rs                                                    (Task 7)
│           ├── recommend.rs           # native + error cases                (Task 8)
│           ├── models.rs              # GET /v1/models                      (Task 9)
│           ├── openai.rs                                                    (Task 10)
│           ├── anthropic.rs                                                 (Task 11)
│           └── consistency.rs         # cross-dialect parity                (Task 12)
```

★ = learning contribution point (Task 3, Task 5).

### Spec coverage map

| SPEC section | Task |
|---|---|
| §4 core types | 2 |
| §5 difficulty | 3 |
| §6 registry | 4 |
| §7 ranker | 5 |
| §8 Router trait | 6 |
| §9.1 /health, §11 config | 7 |
| §9.3 native, §10 errors | 8 |
| §9.2 /v1/models | 9 |
| §9.4 OpenAI | 10 |
| §9.5 Anthropic | 11 |
| §13 cross-dialect parity | 12 |

---

## Task 1: Workspace scaffold

**Files:**
- Create: `Cargo.toml`
- Create: `.gitignore`
- Create: `crates/core/Cargo.toml`
- Create: `crates/core/src/lib.rs`
- Create: `crates/server/Cargo.toml`
- Create: `crates/server/src/lib.rs`
- Create: `crates/server/src/main.rs`

- [x] **Step 1: Create the workspace manifest**

`Cargo.toml`:
```toml
[workspace]
resolver = "2"
members = ["crates/core", "crates/server"]

[profile.dev.package."*"]
debug = false
```

- [x] **Step 2: Create `.gitignore`**

`.gitignore`:
```gitignore
/target
**/*.rs.bk
Cargo.lock
```

- [x] **Step 3: Create the core crate manifest**

`crates/core/Cargo.toml`:
```toml
[package]
name = "route-llm-core"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1", features = ["derive"] }

[dev-dependencies]
serde_json = "1"
```

- [x] **Step 4: Create a placeholder core lib**

`crates/core/src/lib.rs`:
```rust
//! route-llm core: pure routing logic (no I/O).
```

- [x] **Step 5: Create the server crate manifest**

`crates/server/Cargo.toml`:
```toml
[package]
name = "route-llm-server"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "route-llm"
path = "src/main.rs"

[dependencies]
route-llm-core = { path = "../core" }
axum = "0.7"
tokio = { version = "1", features = ["rt-multi-thread", "macros", "net"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
thiserror = "1"

[dev-dependencies]
axum-test = "14"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
serde_json = "1"
```

- [x] **Step 6: Create a placeholder server lib + bin**

`crates/server/src/lib.rs`:
```rust
//! route-llm HTTP server.
```

`crates/server/src/main.rs`:
```rust
fn main() {
    println!("route-llm placeholder");
}
```

- [x] **Step 7: Verify the workspace builds**

Run: `cargo build`
Expected: compiles successfully (both crates), no errors.

- [x] **Step 8: Commit**

```bash
cargo fmt --all
lineguard Cargo.toml .gitignore crates/core/Cargo.toml crates/core/src/lib.rs crates/server/Cargo.toml crates/server/src/lib.rs crates/server/src/main.rs
git add -A
git commit -m "chore: scaffold cargo workspace with core and server crates"
```

---

## Task 2: Core domain types

**Files:**
- Create: `crates/core/src/model.rs`
- Modify: `crates/core/src/lib.rs`

- [x] **Step 1: Write the failing test**

Append to `crates/core/src/model.rs`:
```rust
use serde::{Deserialize, Serialize};

/// A candidate model's capability/cost profile; `quality` and `cost` are normalized to 0.0..=1.0.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelProfile {
    pub id: String,
    pub quality: f64,
    pub cost: f64,
}

/// Routing preferences (the tunable knob).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RoutingPreferences {
    /// 0.0 = quality-first, 1.0 = cost-first.
    pub cost_bias: f64,
}

impl Default for RoutingPreferences {
    fn default() -> Self {
        Self { cost_bias: 0.5 }
    }
}

/// Estimated query difficulty.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Difficulty {
    pub score: f64,
    pub signals: Vec<String>,
}

/// One model's ranking result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedModel {
    pub id: String,
    pub score: f64,
    pub reason: String,
}

/// The router's final output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Recommendation {
    pub difficulty: Difficulty,
    pub ranking: Vec<RankedModel>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_cost_bias_is_half() {
        assert_eq!(RoutingPreferences::default().cost_bias, 0.5);
    }

    #[test]
    fn recommendation_serializes_to_expected_shape() {
        let rec = Recommendation {
            difficulty: Difficulty { score: 0.5, signals: vec!["code".into()] },
            ranking: vec![RankedModel { id: "m".into(), score: 0.4, reason: "r".into() }],
        };
        let v = serde_json::to_value(&rec).unwrap();
        assert_eq!(v["difficulty"]["score"], 0.5);
        assert_eq!(v["ranking"][0]["id"], "m");
    }
}
```

- [x] **Step 2: Wire the module**

Replace `crates/core/src/lib.rs` with:
```rust
//! route-llm core: pure routing logic (no I/O).

pub mod model;

pub use model::{Difficulty, ModelProfile, RankedModel, Recommendation, RoutingPreferences};
```

- [x] **Step 3: Run the tests**

Run: `cargo test -p route-llm-core`
Expected: PASS (2 tests).

- [x] **Step 4: Commit**

```bash
cargo fmt --all
lineguard crates/core/src/model.rs crates/core/src/lib.rs
git add crates/core/src/model.rs crates/core/src/lib.rs
git commit -m "feat(core): add domain types for models, difficulty, and recommendations"
```

---

## Task 3: Difficulty scorer ★ (learning contribution point)

> ★ During execution this is a good place for the project owner to author/tune the feature weights and detection rules (SPEC §5). The implementation below is the spec's default and serves as the reference/fallback. Keep the function signature and `signals` names stable so later tasks/tests still match.

**Files:**
- Create: `crates/core/src/difficulty.rs`
- Modify: `crates/core/src/lib.rs`

- [x] **Step 1: Write the failing test**

Create `crates/core/src/difficulty.rs`:
```rust
use crate::model::Difficulty;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_midpoint() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn trivial_query_is_easy() {
        let d = score("hi");
        assert!(d.score < 0.4, "score was {}", d.score);
    }

    #[test]
    fn hard_query_scores_high_with_expected_signals() {
        let q = "Explain why the Paxos consensus algorithm guarantees safety, derive the invariant it maintains, and analyze step by step how it handles a network partition.";
        let d = score(q);
        assert!(d.score > 0.6 && d.score < 0.85, "score was {}", d.score);
        assert!(d.signals.contains(&"reasoning".to_string()));
        assert!(d.signals.contains(&"explanation_request".to_string()));
    }

    #[test]
    fn code_query_flags_code_signal() {
        let q = "Write a function in Rust: ```fn main() {}``` and optimize it.";
        let d = score(q);
        assert!(d.signals.contains(&"code".to_string()));
        assert!(d.score > 0.5, "score was {}", d.score);
    }
}
```

- [x] **Step 2: Run the test to verify it fails**

Run: `cargo test -p route-llm-core difficulty`
Expected: FAILS to compile (`score`/`sigmoid` not found).

- [x] **Step 3: Write the implementation (reference default)**

Prepend to `crates/core/src/difficulty.rs` (above the `#[cfg(test)]` module):
```rust
pub(crate) fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Estimate query difficulty. Pure and deterministic. See SPEC §5.
pub fn score(query: &str) -> Difficulty {
    let lower = query.to_lowercase();
    let mut sum = -1.0_f64; // base bias
    let mut signals: Vec<String> = Vec::new();

    // Length: estimate tokens ~= chars/4; +0.001 per token, capped at +1.2.
    let est_tokens = query.chars().count() as f64 / 4.0;
    let length_contrib = (est_tokens * 0.001).min(1.2);
    sum += length_contrib;
    if length_contrib > 0.3 {
        signals.push("long_form".into());
    }

    // Code.
    let code_markers = ["```", "fn ", "def ", "class ", "import ", "function", "select "];
    if code_markers.iter().any(|m| lower.contains(m)) {
        sum += 1.0;
        signals.push("code".into());
    }

    // Math / LaTeX.
    let math_markers = ["\\frac", "\\sum", "\\int", "∑", "∫"];
    if query.matches('$').count() >= 2 || math_markers.iter().any(|m| query.contains(m)) {
        sum += 0.8;
        signals.push("math".into());
    }

    // Reasoning keywords: +0.5 each, capped at +1.5.
    let reasoning = [
        "prove", "derive", "step by step", "analyze", "analyse", "design", "explain why",
        "optimize", "optimise", "compare", "證明", "推導", "逐步", "分析", "設計", "比較",
    ];
    let hits = reasoning.iter().filter(|k| lower.contains(&k.to_lowercase())).count();
    if hits > 0 {
        sum += (hits as f64 * 0.5).min(1.5);
        signals.push("reasoning".into());
    }

    // Multi-part constraints: numbered list >= 3 items, or >= 3 questions.
    let numbered = count_numbered_items(query);
    let questions = query.matches('?').count() + query.matches('？').count();
    if numbered >= 3 || questions >= 3 {
        sum += 0.6;
        signals.push("multi_constraint".into());
    }

    // Structured output request.
    let structured = ["json", "table", "schema", "yaml", "csv", "格式", "表格"];
    if structured.iter().any(|s| lower.contains(s)) {
        sum += 0.4;
        signals.push("structured_output".into());
    }

    // Explanation request.
    let explain = ["explain", "說明", "為什麼", "how does", "怎麼", "what is"];
    if explain.iter().any(|s| lower.contains(&s.to_lowercase())) {
        sum += 0.4;
        signals.push("explanation_request".into());
    }

    Difficulty { score: sigmoid(sum), signals }
}

/// Count lines that begin with `<digit>.` or `<digit>)`.
fn count_numbered_items(query: &str) -> usize {
    query
        .lines()
        .filter(|line| {
            let mut chars = line.trim_start().chars();
            match chars.next() {
                Some(c) if c.is_ascii_digit() => matches!(chars.next(), Some('.') | Some(')')),
                _ => false,
            }
        })
        .count()
}
```

- [x] **Step 4: Wire the module**

In `crates/core/src/lib.rs`, add `pub mod difficulty;` below `pub mod model;`:
```rust
//! route-llm core: pure routing logic (no I/O).

pub mod difficulty;
pub mod model;

pub use model::{Difficulty, ModelProfile, RankedModel, Recommendation, RoutingPreferences};
```

- [x] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p route-llm-core difficulty`
Expected: PASS (4 tests).

- [x] **Step 6: Commit**

```bash
cargo fmt --all
lineguard crates/core/src/difficulty.rs crates/core/src/lib.rs
git add crates/core/src/difficulty.rs crates/core/src/lib.rs
git commit -m "feat(core): add heuristic query difficulty scorer"
```

---

## Task 4: Model registry

**Files:**
- Create: `crates/core/src/registry.rs`
- Modify: `crates/core/src/lib.rs`

- [x] **Step 1: Write the failing test**

Create `crates/core/src/registry.rs`:
```rust
use crate::model::ModelProfile;

#[cfg(test)]
mod tests {
    use super::*;

    fn input(id: &str, quality: Option<f64>, cost: Option<f64>) -> CandidateInput {
        CandidateInput { id: id.into(), quality, cost }
    }

    #[test]
    fn known_id_resolves_to_builtin() {
        let got = resolve(&[input("claude-haiku-4-5", None, None)]).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].quality, 0.75);
        assert_eq!(got[0].cost, 0.12);
    }

    #[test]
    fn full_override_accepts_unknown_id() {
        let got = resolve(&[input("brand-new", Some(0.5), Some(0.2))]).unwrap();
        assert_eq!(got[0], ModelProfile { id: "brand-new".into(), quality: 0.5, cost: 0.2 });
    }

    #[test]
    fn partial_override_on_known_merges() {
        let got = resolve(&[input("gpt-4o-mini", None, Some(0.05))]).unwrap();
        assert_eq!(got[0].quality, 0.62); // builtin
        assert_eq!(got[0].cost, 0.05); // overridden
    }

    #[test]
    fn unknown_without_full_override_errors() {
        let err = resolve(&[input("nope", None, None)]).unwrap_err();
        assert_eq!(err, vec!["nope".to_string()]);

        let err2 = resolve(&[input("nope", Some(0.5), None)]).unwrap_err();
        assert_eq!(err2, vec!["nope".to_string()]);
    }

    #[test]
    fn duplicate_ids_keep_last() {
        let got = resolve(&[
            input("gpt-4o-mini", None, None),
            input("gpt-4o-mini", None, Some(0.99)),
        ])
        .unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].cost, 0.99);
    }
}
```

- [x] **Step 2: Run the test to verify it fails**

Run: `cargo test -p route-llm-core registry`
Expected: FAILS to compile (`CandidateInput`/`resolve`/`builtin` not found).

- [x] **Step 3: Write the implementation**

Prepend to `crates/core/src/registry.rs` (above the `#[cfg(test)]` module):
```rust
/// A requested candidate: id plus optional overrides.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateInput {
    pub id: String,
    pub quality: Option<f64>,
    pub cost: Option<f64>,
}

/// Built-in model table (seed values; approximate, see SPEC §6).
pub fn builtin() -> Vec<ModelProfile> {
    vec![
        ModelProfile { id: "claude-opus-4-8".into(), quality: 0.97, cost: 0.90 },
        ModelProfile { id: "claude-sonnet-4-6".into(), quality: 0.90, cost: 0.45 },
        ModelProfile { id: "claude-haiku-4-5".into(), quality: 0.75, cost: 0.12 },
        ModelProfile { id: "gpt-4o".into(), quality: 0.88, cost: 0.50 },
        ModelProfile { id: "gpt-4o-mini".into(), quality: 0.62, cost: 0.10 },
        ModelProfile { id: "gemini-1.5-pro".into(), quality: 0.85, cost: 0.40 },
    ]
}

fn lookup(id: &str) -> Option<ModelProfile> {
    builtin().into_iter().find(|m| m.id == id)
}

/// Resolve candidates against the builtin table + overrides.
/// `Err` carries the list of ids that could not be resolved.
pub fn resolve(candidates: &[CandidateInput]) -> Result<Vec<ModelProfile>, Vec<String>> {
    let mut resolved: Vec<ModelProfile> = Vec::new();
    let mut unknown: Vec<String> = Vec::new();

    for c in candidates {
        let profile = match (c.quality, c.cost) {
            (Some(q), Some(co)) => Some(ModelProfile { id: c.id.clone(), quality: q, cost: co }),
            _ => lookup(&c.id).map(|mut base| {
                if let Some(q) = c.quality {
                    base.quality = q;
                }
                if let Some(co) = c.cost {
                    base.cost = co;
                }
                base
            }),
        };

        match profile {
            Some(p) => match resolved.iter_mut().find(|m| m.id == p.id) {
                Some(existing) => *existing = p, // dedup: keep last
                None => resolved.push(p),
            },
            None => {
                if !unknown.contains(&c.id) {
                    unknown.push(c.id.clone());
                }
            }
        }
    }

    if unknown.is_empty() {
        Ok(resolved)
    } else {
        Err(unknown)
    }
}
```

- [x] **Step 4: Wire the module**

In `crates/core/src/lib.rs`, add `pub mod registry;` and re-export `CandidateInput`:
```rust
//! route-llm core: pure routing logic (no I/O).

pub mod difficulty;
pub mod model;
pub mod registry;

pub use model::{Difficulty, ModelProfile, RankedModel, Recommendation, RoutingPreferences};
pub use registry::CandidateInput;
```

- [x] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p route-llm-core registry`
Expected: PASS (5 tests).

- [x] **Step 6: Commit**

```bash
cargo fmt --all
lineguard crates/core/src/registry.rs crates/core/src/lib.rs
git add crates/core/src/registry.rs crates/core/src/lib.rs
git commit -m "feat(core): add model registry with override merge and resolution"
```

---

## Task 5: Cost-quality ranker ★ (learning contribution point)

> ★ During execution this is the other place for the project owner to author/tune the scoring formula and reason thresholds (SPEC §7). The implementation below is the spec's default. Keep `rank`'s signature stable.

**Files:**
- Create: `crates/core/src/ranker.rs`
- Modify: `crates/core/src/lib.rs`

- [x] **Step 1: Write the failing test**

Create `crates/core/src/ranker.rs`:
```rust
use crate::difficulty::sigmoid;
use crate::model::{Difficulty, ModelProfile, RankedModel, RoutingPreferences};

#[cfg(test)]
mod tests {
    use super::*;

    fn diff(score: f64) -> Difficulty {
        Difficulty { score, signals: vec![] }
    }
    fn m(id: &str, quality: f64, cost: f64) -> ModelProfile {
        ModelProfile { id: id.into(), quality, cost }
    }
    fn prefs(cost_bias: f64) -> RoutingPreferences {
        RoutingPreferences { cost_bias }
    }

    #[test]
    fn easy_query_prefers_cheaper_adequate_model() {
        let out = rank(&diff(0.1), &[m("strong", 0.9, 0.9), m("cheap", 0.5, 0.1)], &prefs(0.5));
        assert_eq!(out[0].id, "cheap");
    }

    #[test]
    fn hard_query_prefers_stronger_model() {
        let out = rank(&diff(0.9), &[m("strong", 0.9, 0.9), m("cheap", 0.5, 0.1)], &prefs(0.5));
        assert_eq!(out[0].id, "strong");
    }

    #[test]
    fn cost_bias_changes_ordering() {
        let models = [m("strong", 0.9, 0.9), m("cheap", 0.6, 0.1)];
        let quality_first = rank(&diff(0.5), &models, &prefs(0.0));
        let cost_first = rank(&diff(0.5), &models, &prefs(1.0));
        assert_eq!(quality_first[0].id, "strong");
        assert_eq!(cost_first[0].id, "cheap");
    }

    #[test]
    fn ties_break_deterministically_by_id() {
        let out = rank(&diff(0.5), &[m("b", 0.7, 0.3), m("a", 0.7, 0.3)], &prefs(0.5));
        assert_eq!(out[0].id, "a");
    }

    #[test]
    fn inadequate_model_gets_capability_reason() {
        let out = rank(&diff(0.95), &[m("weak", 0.3, 0.1)], &prefs(0.5));
        assert!(out[0].reason.contains("能力可能不足"));
    }
}
```

- [x] **Step 2: Run the test to verify it fails**

Run: `cargo test -p route-llm-core ranker`
Expected: FAILS to compile (`rank` not found).

- [x] **Step 3: Write the implementation (reference default)**

Prepend to `crates/core/src/ranker.rs` (above the `#[cfg(test)]` module):
```rust
const K: f64 = 8.0;

/// Rank candidate models for a query difficulty under the given preferences. See SPEC §7.
pub fn rank(
    difficulty: &Difficulty,
    profiles: &[ModelProfile],
    prefs: &RoutingPreferences,
) -> Vec<RankedModel> {
    let d = difficulty.score;
    let lambda = prefs.cost_bias;

    // (profile, adequacy, score)
    let mut scored: Vec<(ModelProfile, f64, f64)> = profiles
        .iter()
        .map(|m| {
            let adequacy = sigmoid(K * (m.quality - d));
            let score = adequacy - lambda * m.cost;
            (m.clone(), adequacy, score)
        })
        .collect();

    // score desc, then quality desc, then cost asc, then id asc (deterministic).
    scored.sort_by(|a, b| {
        use std::cmp::Ordering::Equal;
        b.2.partial_cmp(&a.2).unwrap_or(Equal)
            .then(b.0.quality.partial_cmp(&a.0.quality).unwrap_or(Equal))
            .then(a.0.cost.partial_cmp(&b.0.cost).unwrap_or(Equal))
            .then(a.0.id.cmp(&b.0.id))
    });

    let max_quality = profiles.iter().map(|m| m.quality).fold(f64::MIN, f64::max);
    let cheapest_adequate_cost = scored
        .iter()
        .filter(|(_, adequacy, _)| *adequacy >= 0.5)
        .map(|(m, _, _)| m.cost)
        .fold(f64::MAX, f64::min);

    scored
        .into_iter()
        .map(|(m, adequacy, score)| {
            let reason = make_reason(d, &m, adequacy, max_quality, cheapest_adequate_cost);
            RankedModel { id: m.id, score, reason }
        })
        .collect()
}

fn make_reason(
    d: f64,
    m: &ModelProfile,
    adequacy: f64,
    max_quality: f64,
    cheapest_adequate_cost: f64,
) -> String {
    if adequacy < 0.5 {
        format!("能力可能不足以可靠處理此難度 (difficulty {:.2})", d)
    } else if d >= 0.6 && (m.quality - max_quality).abs() < f64::EPSILON {
        "高難度，最強模型最可靠".into()
    } else if d < 0.4 && (m.cost - cheapest_adequate_cost).abs() < f64::EPSILON {
        "低難度，便宜且足夠".into()
    } else {
        "在品質與成本間取得平衡".into()
    }
}
```

- [x] **Step 4: Wire the module**

In `crates/core/src/lib.rs`, add `pub mod ranker;`:
```rust
//! route-llm core: pure routing logic (no I/O).

pub mod difficulty;
pub mod model;
pub mod ranker;
pub mod registry;

pub use model::{Difficulty, ModelProfile, RankedModel, Recommendation, RoutingPreferences};
pub use registry::CandidateInput;
```

- [x] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p route-llm-core ranker`
Expected: PASS (5 tests).

- [x] **Step 6: Commit**

```bash
cargo fmt --all
lineguard crates/core/src/ranker.rs crates/core/src/lib.rs
git add crates/core/src/ranker.rs crates/core/src/lib.rs
git commit -m "feat(core): add cost-quality ranker with deterministic tie-break"
```

---

## Task 6: Router trait + HeuristicRouter

**Files:**
- Create: `crates/core/src/router.rs`
- Modify: `crates/core/src/lib.rs`

- [x] **Step 1: Write the failing test**

Create `crates/core/src/router.rs`:
```rust
use crate::model::{ModelProfile, Recommendation, RoutingPreferences};
use crate::{difficulty, ranker};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_router_orders_spec_example() {
        // SPEC §9 example: hard query, cost_bias 0.3 -> opus > haiku > gpt-4o-mini.
        let query = "Explain why the Paxos consensus algorithm guarantees safety, derive the invariant it maintains, and analyze step by step how it handles a network partition.";
        let models = [
            ModelProfile { id: "claude-opus-4-8".into(), quality: 0.97, cost: 0.90 },
            ModelProfile { id: "claude-haiku-4-5".into(), quality: 0.75, cost: 0.12 },
            ModelProfile { id: "gpt-4o-mini".into(), quality: 0.55, cost: 0.10 },
        ];
        let rec = HeuristicRouter.recommend(query, &models, &RoutingPreferences { cost_bias: 0.3 });

        let order: Vec<&str> = rec.ranking.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(order, ["claude-opus-4-8", "claude-haiku-4-5", "gpt-4o-mini"]);
        assert!(rec.difficulty.score > 0.6 && rec.difficulty.score < 0.85);
        assert!(rec.difficulty.signals.contains(&"reasoning".to_string()));
    }
}
```

- [x] **Step 2: Run the test to verify it fails**

Run: `cargo test -p route-llm-core router`
Expected: FAILS to compile (`HeuristicRouter`/`Router` not found).

- [x] **Step 3: Write the implementation**

Prepend to `crates/core/src/router.rs` (above the `#[cfg(test)]` module):
```rust
/// A routing strategy. v1 ships one implementation; future strategies plug in here.
pub trait Router {
    fn recommend(
        &self,
        query: &str,
        models: &[ModelProfile],
        prefs: &RoutingPreferences,
    ) -> Recommendation;
}

/// v1's first strategy: heuristic difficulty scoring + cost-quality ranking.
pub struct HeuristicRouter;

impl Router for HeuristicRouter {
    fn recommend(
        &self,
        query: &str,
        models: &[ModelProfile],
        prefs: &RoutingPreferences,
    ) -> Recommendation {
        let difficulty = difficulty::score(query);
        let ranking = ranker::rank(&difficulty, models, prefs);
        Recommendation { difficulty, ranking }
    }
}
```

- [x] **Step 4: Wire the module**

Replace `crates/core/src/lib.rs` with:
```rust
//! route-llm core: pure routing logic (no I/O).

pub mod difficulty;
pub mod model;
pub mod ranker;
pub mod registry;
pub mod router;

pub use model::{Difficulty, ModelProfile, RankedModel, Recommendation, RoutingPreferences};
pub use registry::CandidateInput;
pub use router::{HeuristicRouter, Router};
```

- [x] **Step 5: Run the full core test suite**

Run: `cargo test -p route-llm-core`
Expected: PASS (all core tests; includes the new router test).

- [x] **Step 6: Commit**

```bash
cargo fmt --all
lineguard crates/core/src/router.rs crates/core/src/lib.rs
git add crates/core/src/router.rs crates/core/src/lib.rs
git commit -m "feat(core): add Router trait and HeuristicRouter"
```

---

## Task 7: Server app builder + /health + config

**Files:**
- Modify: `crates/server/src/lib.rs`
- Modify: `crates/server/src/main.rs`
- Create: `crates/server/src/handlers.rs`
- Create: `crates/server/tests/health.rs`

- [x] **Step 1: Write the failing test**

Create `crates/server/tests/health.rs`:
```rust
use axum_test::TestServer;
use route_llm_server::app;
use serde_json::json;

#[tokio::test]
async fn health_returns_ok() {
    let server = TestServer::new(app()).unwrap();
    let res = server.get("/health").await;
    res.assert_status_ok();
    res.assert_json(&json!({ "status": "ok" }));
}
```

- [x] **Step 2: Run the test to verify it fails**

Run: `cargo test -p route-llm-server --test health`
Expected: FAILS to compile (`route_llm_server::app` not found).

- [x] **Step 3: Write the handler**

Create `crates/server/src/handlers.rs`:
```rust
use axum::Json;
use serde_json::{json, Value};

pub async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}
```

- [x] **Step 4: Write the app builder**

Replace `crates/server/src/lib.rs` with:
```rust
//! route-llm HTTP server.

pub mod handlers;

use axum::routing::get;
use axum::Router;

/// Build the axum application (used by both `main` and integration tests).
pub fn app() -> Router {
    Router::new().route("/health", get(handlers::health))
}
```

- [x] **Step 5: Write the binary entrypoint**

Replace `crates/server/src/main.rs` with:
```rust
use std::net::SocketAddr;

use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    let host = std::env::var("ROUTE_LLM_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("ROUTE_LLM_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(8080);
    let addr: SocketAddr = format!("{host}:{port}").parse().expect("invalid ROUTE_LLM_HOST/PORT");

    let app = route_llm_server::app();
    tracing::info!("route-llm listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.expect("failed to bind");
    axum::serve(listener, app).await.expect("server error");
}
```

- [x] **Step 6: Run the test to verify it passes**

Run: `cargo test -p route-llm-server --test health`
Expected: PASS (1 test).

- [x] **Step 7: Commit**

```bash
cargo fmt --all
lineguard crates/server/src/lib.rs crates/server/src/main.rs crates/server/src/handlers.rs crates/server/tests/health.rs
git add crates/server/src/lib.rs crates/server/src/main.rs crates/server/src/handlers.rs crates/server/tests/health.rs
git commit -m "feat(server): add axum app builder, /health, and env-based config"
```

---

## Task 8: Native /v1/recommend + error handling

**Files:**
- Create: `crates/server/src/error.rs`
- Create: `crates/server/src/dto.rs`
- Modify: `crates/server/src/handlers.rs`
- Modify: `crates/server/src/lib.rs`
- Create: `crates/server/tests/recommend.rs`

- [x] **Step 1: Write the failing test**

Create `crates/server/tests/recommend.rs`:
```rust
use axum_test::TestServer;
use route_llm_server::app;
use serde_json::json;

fn server() -> TestServer {
    TestServer::new(app()).unwrap()
}

#[tokio::test]
async fn recommend_happy_path_orders_models() {
    let res = server()
        .post("/v1/recommend")
        .json(&json!({
            "query": "Explain why the Paxos consensus algorithm guarantees safety, derive the invariant it maintains, and analyze step by step how it handles a network partition.",
            "models": [
                {"id": "claude-opus-4-8"},
                {"id": "claude-haiku-4-5"},
                {"id": "gpt-4o-mini", "quality": 0.55, "cost": 0.10}
            ],
            "preferences": {"cost_bias": 0.3}
        }))
        .await;
    res.assert_status_ok();
    let body: serde_json::Value = res.json();
    assert_eq!(body["ranking"][0]["id"], "claude-opus-4-8");
    let score = body["difficulty"]["score"].as_f64().unwrap();
    assert!(score > 0.6 && score < 0.85);
}

#[tokio::test]
async fn empty_query_is_rejected() {
    let res = server()
        .post("/v1/recommend")
        .json(&json!({ "query": "  ", "models": [{"id": "gpt-4o-mini"}] }))
        .await;
    res.assert_status_bad_request();
    assert_eq!(res.json::<serde_json::Value>()["error"]["code"], "empty_query");
}

#[tokio::test]
async fn empty_candidates_is_rejected() {
    let res = server()
        .post("/v1/recommend")
        .json(&json!({ "query": "hi", "models": [] }))
        .await;
    res.assert_status_bad_request();
    assert_eq!(res.json::<serde_json::Value>()["error"]["code"], "empty_candidates");
}

#[tokio::test]
async fn unknown_model_is_rejected_with_details() {
    let res = server()
        .post("/v1/recommend")
        .json(&json!({ "query": "hi", "models": [{"id": "does-not-exist"}] }))
        .await;
    res.assert_status_bad_request();
    let body: serde_json::Value = res.json();
    assert_eq!(body["error"]["code"], "unknown_models");
    assert_eq!(body["error"]["details"]["unknown"][0], "does-not-exist");
}

#[tokio::test]
async fn invalid_cost_bias_is_rejected() {
    let res = server()
        .post("/v1/recommend")
        .json(&json!({
            "query": "hi",
            "models": [{"id": "gpt-4o-mini"}],
            "preferences": {"cost_bias": 1.5}
        }))
        .await;
    res.assert_status_bad_request();
    assert_eq!(res.json::<serde_json::Value>()["error"]["code"], "invalid_preferences");
}

#[tokio::test]
async fn malformed_json_is_rejected() {
    let res = server()
        .post("/v1/recommend")
        .text("{ not json")
        .content_type("application/json")
        .await;
    res.assert_status_bad_request();
    assert_eq!(res.json::<serde_json::Value>()["error"]["code"], "invalid_json");
}
```

- [x] **Step 2: Run the test to verify it fails**

Run: `cargo test -p route-llm-server --test recommend`
Expected: FAILS to compile / route 404 (handler + route not defined yet).

- [x] **Step 3: Write the error type**

Create `crates/server/src/error.rs`:
```rust
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("invalid JSON body: {0}")]
    InvalidJson(String),
    #[error("query text is empty")]
    EmptyQuery,
    #[error("no candidate models provided")]
    EmptyCandidates,
    #[error("unknown models: {0:?}")]
    UnknownModels(Vec<String>),
    #[error("{0}")]
    InvalidPreferences(String),
}

impl ApiError {
    fn code(&self) -> &'static str {
        match self {
            ApiError::InvalidJson(_) => "invalid_json",
            ApiError::EmptyQuery => "empty_query",
            ApiError::EmptyCandidates => "empty_candidates",
            ApiError::UnknownModels(_) => "unknown_models",
            ApiError::InvalidPreferences(_) => "invalid_preferences",
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let details = match &self {
            ApiError::UnknownModels(ids) => json!({ "unknown": ids }),
            _ => json!({}),
        };
        let body = json!({
            "error": { "code": self.code(), "message": self.to_string(), "details": details }
        });
        (StatusCode::BAD_REQUEST, Json(body)).into_response()
    }
}
```

- [x] **Step 4: Write the DTOs**

Create `crates/server/src/dto.rs`:
```rust
use route_llm_core::{CandidateInput, RoutingPreferences};
use serde::Deserialize;

/// A candidate model entry in a request (id + optional overrides).
#[derive(Debug, Deserialize)]
pub struct ModelInput {
    pub id: String,
    #[serde(default)]
    pub quality: Option<f64>,
    #[serde(default)]
    pub cost: Option<f64>,
}

impl From<ModelInput> for CandidateInput {
    fn from(m: ModelInput) -> Self {
        CandidateInput { id: m.id, quality: m.quality, cost: m.cost }
    }
}

/// Optional routing preferences in a request body.
#[derive(Debug, Deserialize)]
pub struct PrefsInput {
    pub cost_bias: f64,
}

impl From<PrefsInput> for RoutingPreferences {
    fn from(p: PrefsInput) -> Self {
        RoutingPreferences { cost_bias: p.cost_bias }
    }
}

/// Native `/v1/recommend` request.
#[derive(Debug, Deserialize)]
pub struct RecommendRequest {
    pub query: String,
    #[serde(default)]
    pub models: Vec<ModelInput>,
    #[serde(default)]
    pub preferences: Option<PrefsInput>,
}
```

- [x] **Step 5: Write the shared processing + native handler**

Replace `crates/server/src/handlers.rs` with:
```rust
use axum::extract::rejection::JsonRejection;
use axum::Json;
use serde_json::{json, Value};

use route_llm_core::{registry, CandidateInput, HeuristicRouter, Recommendation, Router, RoutingPreferences};

use crate::dto::{ModelInput, PrefsInput, RecommendRequest};
use crate::error::ApiError;

pub async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

/// Merge a candidate list with an optional standard `model` field (hint).
pub(crate) fn collect_candidates(model: Option<String>, models: Vec<ModelInput>) -> Vec<CandidateInput> {
    let mut out: Vec<CandidateInput> = models.into_iter().map(Into::into).collect();
    if let Some(id) = model {
        if !id.is_empty() && !out.iter().any(|c| c.id == id) {
            out.push(CandidateInput { id, quality: None, cost: None });
        }
    }
    out
}

pub(crate) fn prefs_or_default(p: Option<PrefsInput>) -> RoutingPreferences {
    p.map(Into::into).unwrap_or_default()
}

/// Shared across all three dialects: validate, resolve, route.
pub(crate) fn process(
    query: &str,
    candidates: Vec<CandidateInput>,
    prefs: RoutingPreferences,
) -> Result<Recommendation, ApiError> {
    if query.trim().is_empty() {
        return Err(ApiError::EmptyQuery);
    }
    if candidates.is_empty() {
        return Err(ApiError::EmptyCandidates);
    }
    if !(0.0..=1.0).contains(&prefs.cost_bias) {
        return Err(ApiError::InvalidPreferences("cost_bias must be in 0.0..=1.0".into()));
    }
    let profiles = registry::resolve(&candidates).map_err(ApiError::UnknownModels)?;
    Ok(HeuristicRouter.recommend(query, &profiles, &prefs))
}

pub async fn recommend(
    payload: Result<Json<RecommendRequest>, JsonRejection>,
) -> Result<Json<Recommendation>, ApiError> {
    let Json(req) = payload.map_err(|e| ApiError::InvalidJson(e.body_text()))?;
    let candidates = collect_candidates(None, req.models);
    let rec = process(&req.query, candidates, prefs_or_default(req.preferences))?;
    Ok(Json(rec))
}
```

- [x] **Step 6: Wire modules and the new route**

Replace `crates/server/src/lib.rs` with:
```rust
//! route-llm HTTP server.

pub mod dto;
pub mod error;
pub mod handlers;

use axum::routing::{get, post};
use axum::Router;

/// Build the axum application (used by both `main` and integration tests).
pub fn app() -> Router {
    Router::new()
        .route("/health", get(handlers::health))
        .route("/v1/recommend", post(handlers::recommend))
}
```

- [x] **Step 7: Run the tests to verify they pass**

Run: `cargo test -p route-llm-server --test recommend`
Expected: PASS (6 tests).

- [x] **Step 8: Commit**

```bash
cargo fmt --all
lineguard crates/server/src/error.rs crates/server/src/dto.rs crates/server/src/handlers.rs crates/server/src/lib.rs crates/server/tests/recommend.rs
git add crates/server/src/error.rs crates/server/src/dto.rs crates/server/src/handlers.rs crates/server/src/lib.rs crates/server/tests/recommend.rs
git commit -m "feat(server): add native /v1/recommend endpoint with structured errors"
```

---

## Task 9: GET /v1/models

**Files:**
- Modify: `crates/server/src/handlers.rs`
- Modify: `crates/server/src/lib.rs`
- Create: `crates/server/tests/models.rs`

- [x] **Step 1: Write the failing test**

Create `crates/server/tests/models.rs`:
```rust
use axum_test::TestServer;
use route_llm_server::app;

#[tokio::test]
async fn lists_builtin_models() {
    let server = TestServer::new(app()).unwrap();
    let res = server.get("/v1/models").await;
    res.assert_status_ok();
    let body: serde_json::Value = res.json();
    let models = body["models"].as_array().unwrap();
    assert_eq!(models.len(), 6);
    assert!(models.iter().any(|m| m["id"] == "claude-opus-4-8"));
    assert!(models[0]["quality"].is_number());
    assert!(models[0]["cost"].is_number());
}
```

- [x] **Step 2: Run the test to verify it fails**

Run: `cargo test -p route-llm-server --test models`
Expected: FAIL (404 / route not found).

- [x] **Step 3: Add the handler**

Append to `crates/server/src/handlers.rs`:
```rust
pub async fn list_models() -> Json<Value> {
    Json(json!({ "models": registry::builtin() }))
}
```

- [x] **Step 4: Add the route**

In `crates/server/src/lib.rs`, add the `/v1/models` route to `app()`:
```rust
pub fn app() -> Router {
    Router::new()
        .route("/health", get(handlers::health))
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/recommend", post(handlers::recommend))
}
```

- [x] **Step 5: Run the test to verify it passes**

Run: `cargo test -p route-llm-server --test models`
Expected: PASS (1 test).

- [x] **Step 6: Commit**

```bash
cargo fmt --all
lineguard crates/server/src/handlers.rs crates/server/src/lib.rs crates/server/tests/models.rs
git add crates/server/src/handlers.rs crates/server/src/lib.rs crates/server/tests/models.rs
git commit -m "feat(server): add GET /v1/models to list the builtin registry"
```

---

## Task 10: OpenAI-shaped /v1/chat/completions

**Files:**
- Modify: `crates/server/src/dto.rs`
- Modify: `crates/server/src/handlers.rs`
- Modify: `crates/server/src/lib.rs`
- Create: `crates/server/tests/openai.rs`

- [x] **Step 1: Write the failing test**

Create `crates/server/tests/openai.rs`:
```rust
use axum_test::TestServer;
use route_llm_server::app;
use serde_json::json;

#[tokio::test]
async fn chat_completions_returns_completion_envelope() {
    let server = TestServer::new(app()).unwrap();
    let res = server
        .post("/v1/chat/completions")
        .json(&json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Explain why the Paxos consensus algorithm guarantees safety, derive the invariant it maintains, and analyze step by step how it handles a network partition."}],
            "models": [
                {"id": "claude-opus-4-8"},
                {"id": "claude-haiku-4-5"},
                {"id": "gpt-4o-mini", "quality": 0.55, "cost": 0.10}
            ],
            "preferences": {"cost_bias": 0.3}
        }))
        .await;
    res.assert_status_ok();
    let body: serde_json::Value = res.json();
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "claude-opus-4-8");
    assert_eq!(body["route_llm"]["ranking"][0]["id"], "claude-opus-4-8");
    assert_eq!(body["usage"]["total_tokens"], 0);
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(content.starts_with("Recommended:"));
}

#[tokio::test]
async fn model_field_alone_is_used_as_candidate() {
    let server = TestServer::new(app()).unwrap();
    let res = server
        .post("/v1/chat/completions")
        .json(&json!({
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .await;
    res.assert_status_ok();
    let body: serde_json::Value = res.json();
    assert_eq!(body["route_llm"]["ranking"].as_array().unwrap().len(), 1);
    assert_eq!(body["model"], "claude-haiku-4-5");
}
```

- [x] **Step 2: Run the test to verify it fails**

Run: `cargo test -p route-llm-server --test openai`
Expected: FAIL (404 / route not found).

- [x] **Step 3: Add the OpenAI DTOs**

Append to `crates/server/src/dto.rs`:
```rust
use route_llm_core::Recommendation;
use serde::Serialize;

/// A chat message (request side). We only support string `content` in v1.
#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub models: Vec<ModelInput>,
    #[serde(default)]
    pub preferences: Option<PrefsInput>,
}

#[derive(Debug, Serialize)]
pub struct ChatRespMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatRespMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct OpenAiUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: OpenAiUsage,
    pub route_llm: Recommendation,
}
```

- [x] **Step 4: Add shared helpers + the handler**

Append to `crates/server/src/handlers.rs`:
```rust
use std::sync::atomic::{AtomicU64, Ordering};

use crate::dto::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatRespMessage, OpenAiUsage,
};

static ID_COUNTER: AtomicU64 = AtomicU64::new(1);

pub(crate) fn next_id() -> String {
    format!("rec-{:016x}", ID_COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// Human-readable one-liner describing the recommendation.
pub(crate) fn summary_line(rec: &Recommendation) -> String {
    let order = rec
        .ranking
        .iter()
        .map(|r| r.id.as_str())
        .collect::<Vec<_>>()
        .join(" > ");
    let top = rec.ranking.first().map(|r| r.id.as_str()).unwrap_or("(none)");
    format!("Recommended: {} (difficulty {:.2}). Order: {}.", top, rec.difficulty.score, order)
}

pub async fn chat_completions(
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Json<ChatCompletionResponse>, ApiError> {
    let Json(req) = payload.map_err(|e| ApiError::InvalidJson(e.body_text()))?;
    let query = req
        .messages
        .iter()
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let candidates = collect_candidates(req.model, req.models);
    let rec = process(&query, candidates, prefs_or_default(req.preferences))?;
    let top = rec.ranking.first().map(|r| r.id.clone()).unwrap_or_default();

    let resp = ChatCompletionResponse {
        id: next_id(),
        object: "chat.completion",
        model: top,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatRespMessage { role: "assistant", content: summary_line(&rec) },
            finish_reason: "stop",
        }],
        usage: OpenAiUsage { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
        route_llm: rec,
    };
    Ok(Json(resp))
}
```

- [x] **Step 5: Add the route**

In `crates/server/src/lib.rs`, add the `/v1/chat/completions` route to `app()`:
```rust
pub fn app() -> Router {
    Router::new()
        .route("/health", get(handlers::health))
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/recommend", post(handlers::recommend))
        .route("/v1/chat/completions", post(handlers::chat_completions))
}
```

- [x] **Step 6: Run the tests to verify they pass**

Run: `cargo test -p route-llm-server --test openai`
Expected: PASS (2 tests).

- [x] **Step 7: Commit**

```bash
cargo fmt --all
lineguard crates/server/src/dto.rs crates/server/src/handlers.rs crates/server/src/lib.rs crates/server/tests/openai.rs
git add crates/server/src/dto.rs crates/server/src/handlers.rs crates/server/src/lib.rs crates/server/tests/openai.rs
git commit -m "feat(server): add OpenAI-shaped /v1/chat/completions endpoint"
```

---

## Task 11: Anthropic-shaped /v1/messages

**Files:**
- Modify: `crates/server/src/dto.rs`
- Modify: `crates/server/src/handlers.rs`
- Modify: `crates/server/src/lib.rs`
- Create: `crates/server/tests/anthropic.rs`

- [x] **Step 1: Write the failing test**

Create `crates/server/tests/anthropic.rs`:
```rust
use axum_test::TestServer;
use route_llm_server::app;
use serde_json::json;

#[tokio::test]
async fn messages_returns_message_envelope() {
    let server = TestServer::new(app()).unwrap();
    let res = server
        .post("/v1/messages")
        .json(&json!({
            "model": "claude-haiku-4-5",
            "system": "You are concise.",
            "messages": [{"role": "user", "content": "Explain why the Paxos consensus algorithm guarantees safety, derive the invariant it maintains, and analyze step by step how it handles a network partition."}],
            "models": [
                {"id": "claude-opus-4-8"},
                {"id": "claude-haiku-4-5"},
                {"id": "gpt-4o-mini", "quality": 0.55, "cost": 0.10}
            ],
            "preferences": {"cost_bias": 0.3}
        }))
        .await;
    res.assert_status_ok();
    let body: serde_json::Value = res.json();
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert_eq!(body["model"], "claude-opus-4-8");
    assert_eq!(body["content"][0]["type"], "text");
    assert_eq!(body["route_llm"]["ranking"][0]["id"], "claude-opus-4-8");
    assert_eq!(body["usage"]["output_tokens"], 0);
}
```

- [x] **Step 2: Run the test to verify it fails**

Run: `cargo test -p route-llm-server --test anthropic`
Expected: FAIL (404 / route not found).

- [x] **Step 3: Add the Anthropic DTOs**

Append to `crates/server/src/dto.rs`:
```rust
#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub models: Vec<ModelInput>,
    #[serde(default)]
    pub preferences: Option<PrefsInput>,
}

#[derive(Debug, Serialize)]
pub struct AnthropicContent {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<AnthropicContent>,
    pub stop_reason: &'static str,
    pub usage: AnthropicUsage,
    pub route_llm: Recommendation,
}
```

- [x] **Step 4: Add the handler**

Append to `crates/server/src/handlers.rs`:
```rust
use crate::dto::{AnthropicContent, AnthropicUsage, MessagesRequest, MessagesResponse};

pub async fn messages(
    payload: Result<Json<MessagesRequest>, JsonRejection>,
) -> Result<Json<MessagesResponse>, ApiError> {
    let Json(req) = payload.map_err(|e| ApiError::InvalidJson(e.body_text()))?;
    let mut parts: Vec<String> = Vec::new();
    if let Some(sys) = req.system {
        parts.push(sys);
    }
    parts.extend(req.messages.into_iter().map(|m| m.content));
    let query = parts.join("\n");

    let candidates = collect_candidates(req.model, req.models);
    let rec = process(&query, candidates, prefs_or_default(req.preferences))?;
    let top = rec.ranking.first().map(|r| r.id.clone()).unwrap_or_default();

    let resp = MessagesResponse {
        id: next_id(),
        kind: "message",
        role: "assistant",
        model: top,
        content: vec![AnthropicContent { kind: "text", text: summary_line(&rec) }],
        stop_reason: "end_turn",
        usage: AnthropicUsage { input_tokens: 0, output_tokens: 0 },
        route_llm: rec,
    };
    Ok(Json(resp))
}
```

- [x] **Step 5: Add the route**

In `crates/server/src/lib.rs`, add the `/v1/messages` route to `app()`:
```rust
pub fn app() -> Router {
    Router::new()
        .route("/health", get(handlers::health))
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/recommend", post(handlers::recommend))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/messages", post(handlers::messages))
}
```

- [x] **Step 6: Run the tests to verify they pass**

Run: `cargo test -p route-llm-server --test anthropic`
Expected: PASS (1 test).

- [x] **Step 7: Commit**

```bash
cargo fmt --all
lineguard crates/server/src/dto.rs crates/server/src/handlers.rs crates/server/src/lib.rs crates/server/tests/anthropic.rs
git add crates/server/src/dto.rs crates/server/src/handlers.rs crates/server/src/lib.rs crates/server/tests/anthropic.rs
git commit -m "feat(server): add Anthropic-shaped /v1/messages endpoint"
```

---

## Task 12: Cross-dialect consistency + README

**Files:**
- Create: `crates/server/tests/consistency.rs`
- Create: `README.md`

- [x] **Step 1: Write the failing test**

Create `crates/server/tests/consistency.rs`:
```rust
use axum_test::TestServer;
use route_llm_server::app;
use serde_json::{json, Value};

const QUERY: &str = "Explain why the Paxos consensus algorithm guarantees safety, derive the invariant it maintains, and analyze step by step how it handles a network partition.";

fn models() -> Value {
    json!([
        {"id": "claude-opus-4-8"},
        {"id": "claude-haiku-4-5"},
        {"id": "gpt-4o-mini", "quality": 0.55, "cost": 0.10}
    ])
}

fn order(ranking: &Value) -> Vec<String> {
    ranking
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap().to_string())
        .collect()
}

#[tokio::test]
async fn all_three_dialects_agree_on_ranking() {
    let server = TestServer::new(app()).unwrap();

    let native = server
        .post("/v1/recommend")
        .json(&json!({ "query": QUERY, "models": models(), "preferences": {"cost_bias": 0.3} }))
        .await
        .json::<Value>();

    let openai = server
        .post("/v1/chat/completions")
        .json(&json!({
            "messages": [{"role": "user", "content": QUERY}],
            "models": models(),
            "preferences": {"cost_bias": 0.3}
        }))
        .await
        .json::<Value>();

    let anthropic = server
        .post("/v1/messages")
        .json(&json!({
            "messages": [{"role": "user", "content": QUERY}],
            "models": models(),
            "preferences": {"cost_bias": 0.3}
        }))
        .await
        .json::<Value>();

    let native_order = order(&native["ranking"]);
    assert_eq!(native_order, order(&openai["route_llm"]["ranking"]));
    assert_eq!(native_order, order(&anthropic["route_llm"]["ranking"]));
    assert_eq!(native_order[0], "claude-opus-4-8");
}
```

- [x] **Step 2: Run the test to verify it passes**

Run: `cargo test -p route-llm-server --test consistency`
Expected: PASS (1 test). (All endpoints already exist; this is a regression guard for the shared core.)

- [x] **Step 3: Write the README**

Create `README.md`:
```markdown
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
```

- [x] **Step 4: Run the full workspace test suite**

Run: `cargo test`
Expected: PASS (all core + server tests).

- [x] **Step 5: Commit**

```bash
cargo fmt --all
lineguard crates/server/tests/consistency.rs README.md
git add crates/server/tests/consistency.rs README.md
git commit -m "test(server): assert cross-dialect ranking parity; add README"
```

---

## Final verification

- [x] Run `cargo build --release` — clean release build.
- [x] Run `cargo test` — all tests green.
- [x] Manual smoke test:
  ```bash
  cargo run --release -p route-llm-server &
  curl -s localhost:8080/health
  curl -s localhost:8080/v1/recommend -H 'content-type: application/json' \
    -d '{"query":"hi","models":[{"id":"gpt-4o-mini"},{"id":"claude-opus-4-8"}]}'
  ```
  Expected: `/health` → `{"status":"ok"}`; `/v1/recommend` → 200 with a `ranking` array (cheaper model first for the trivial query).
```
