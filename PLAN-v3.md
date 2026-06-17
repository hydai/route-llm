# route-llm v3 — Reasoning Budget Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third routing strategy `BudgetRouter` that decomposes a prompt into six frontier-LLM-labeled, offline-learned dimensions → a weighted `budget_score` → R0–R4 + a deterministic decision layer (risk floors, tool/latest-info, two-estimator disagreement, policy modes), while still emitting the unified `Recommendation` (ranking) plus an additive optional `budget` block.

**Architecture:** New isolated `crates/core/src/budget/` module (mirrors `learned/`). Six `LinearModel` heads share v2's feature vector; `budget_score` and level thresholds follow the RBC formula (SPEC §4–§5). Inference is pure & deterministic — frontier LLMs only label offline (`trainer label --dims` → `data/budget.*.jsonl` → `fit-budget` → `budget/weights.rs`). v1/v2 cores frozen; `RoutingPreferences` and the `Router` trait untouched (`Policy` is a `BudgetRouter` startup config via `ROUTE_LLM_POLICY`). The difficulty backbone is **gold-gated** on the v2.2 143-query human set (SPEC §8) — decided post-labeling, not pre-committed.

**Tech Stack:** Rust (cargo workspace); `serde`/`serde_json`; existing `route-llm-core` `learned::{features, model::LinearModel}` + `ranker`; existing `crates/trainer` `dataset`/`logreg`/`emit`/`eval`/`label`. `reqwest` (already present) for offline labeling only. No new crates; zero new runtime dependencies.

**Branch:** `spec/v3-reasoning-budget` (already created; never commit to `master`). Release builds only (`cargo … --release`).

**Spec:** `SPEC-v3.md` (approved).

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `crates/core/src/model.rs` | Add `Policy`, `DimensionScores`, `BudgetBreakdown`; `Recommendation.budget: Option<…>` | **Modify (additive)** |
| `crates/core/src/budget/dims.rs` | Dimension names/scales/weights; six-head scoring; `budget_score` | **Create** |
| `crates/core/src/budget/level.rs` | `Level` R0–R4, thresholds, tier, raw/floor difficulty | **Create** |
| `crates/core/src/budget/escalation.rs` | Risk floors, latest-info, confidence, disagreement, policy modes | **Create** |
| `crates/core/src/budget/weights.rs` | Shipped six dimension heads (GENERATED; zero placeholder until fit) | **Create (generated)** |
| `crates/core/src/budget/mod.rs` | `BudgetRouter` (implements `Router`) | **Create** |
| `crates/core/src/lib.rs` | `pub mod budget;` + re-exports | **Modify** |
| `crates/core/src/router.rs` | `HeuristicRouter` returns `budget: None` (one line) | **Modify (additive)** |
| `crates/core/src/learned/mod.rs` | `LearnedRouter` returns `budget: None` (one line) | **Modify (additive)** |
| `crates/server/src/main.rs` | `choose_router` accepts `budget`; `choose_policy(ROUTE_LLM_POLICY)` | **Modify** |
| `crates/server/src/handlers.rs` | `summary_line` appends level/tier when budget present | **Modify (minimal)** |
| `crates/server/tests/budget.rs` | Server-level budget routing test | **Create** |
| `crates/trainer/src/dataset.rs` | `DimScores`, `DimsExample` + jsonl I/O | **Modify (additive)** |
| `crates/trainer/src/label.rs` | Make `http_client`/`chat_complete` `pub(crate)` | **Modify (minimal)** |
| `crates/trainer/src/budget_label.rs` | 6-dim prompt/parse/`run_dims`; `fit_dims` | **Create** |
| `crates/trainer/src/emit.rs` | `budget_weights_rs` renderer | **Modify (additive)** |
| `crates/trainer/src/eval.rs` | `run_eval_budget`, `crosseval_dims` | **Modify (additive)** |
| `crates/trainer/src/main.rs` | Dispatch `label --dims` / `fit-budget` / `eval-budget` / `crosseval --dims` | **Modify** |
| `prompts/label.budget.prompt.md` | Portable 6-dim labeling rubric | **Create** |
| `README.md` | Document `ROUTE_LLM_ROUTER=budget` + `ROUTE_LLM_POLICY` | **Modify (docs)** |
| `data/budget.{claude,codex,gemma}.jsonl` | 6-dim labels (frontier LLMs) | **Create (manual)** |
| `data/budget.jsonl` | Shipped-labeler copy for `fit-budget` | **Create (manual)** |
| `SPEC-v3.md` | Fill §16 verdict after the run | **Modify (post-run)** |

Tasks 1–14 are pure engineering (do now). Task 15 is a **manual offline step** (frontier-LLM labeling). Task 16 is post-labeling (fit + verdict + ship).

---

## Phase A — core `budget/` module

## Task 1: `model.rs` — additive output types + `Recommendation.budget`

**Files:**
- Modify: `crates/core/src/model.rs`
- Modify: `crates/core/src/router.rs` (one line)
- Modify: `crates/core/src/learned/mod.rs` (one line)

- [ ] **Step 1: Write the failing test**

Append to the `tests` module in `crates/core/src/model.rs`:

```rust
    #[test]
    fn recommendation_without_budget_omits_the_field() {
        let rec = Recommendation {
            difficulty: Difficulty { score: 0.3, signals: vec![] },
            ranking: vec![],
            budget: None,
        };
        let v = serde_json::to_value(&rec).unwrap();
        assert!(v.get("budget").is_none(), "budget must be omitted when None");
    }

    #[test]
    fn budget_breakdown_serializes_expected_shape() {
        let b = BudgetBreakdown {
            level: "R3".into(),
            budget_score: 13.4,
            recommended_model_tier: "strong".into(),
            confidence: 0.78,
            dimensions: DimensionScores {
                reasoning_depth: 3.0,
                verification_difficulty: 2.0,
                constraint_density: 2.0,
                context_integration: 1.0,
                ambiguity: 1.0,
                error_cost: 2.0,
            },
            reason_codes: vec!["multi_step_reasoning".into()],
            needs_tool: false,
            tool_type: None,
            requires_verifier: false,
            fallback_policy: "none".into(),
        };
        let v = serde_json::to_value(&b).unwrap();
        assert_eq!(v["level"], "R3");
        assert_eq!(v["dimensions"]["reasoning_depth"], 3.0);
        assert!(v.get("tool_type").is_some()); // present-but-null is fine
    }

    #[test]
    fn policy_default_is_balanced() {
        assert_eq!(Policy::default(), Policy::Balanced);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p route-llm-core --release recommendation_without_budget`
Expected: **compile error** — `Policy`, `BudgetBreakdown`, `DimensionScores`, and the `budget` field don't exist yet.

- [ ] **Step 3: Add the types and the field**

In `crates/core/src/model.rs`, after the existing `RoutingPreferences` block, add:

```rust
/// Routing policy for the budget strategy. A `BudgetRouter` startup config
/// (`ROUTE_LLM_POLICY`); not part of `RoutingPreferences`. See SPEC-v3 §6.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Policy {
    Balanced,
    Strict,
    Cheap,
}

impl Default for Policy {
    fn default() -> Self {
        Policy::Balanced
    }
}

/// The six reasoning-budget dimensions, each on its own integer scale (SPEC-v3 §4.1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DimensionScores {
    pub reasoning_depth: f64,
    pub verification_difficulty: f64,
    pub constraint_density: f64,
    pub context_integration: f64,
    pub ambiguity: f64,
    pub error_cost: f64,
}

/// The BudgetRouter's intermediate output. Additive: only the budget strategy
/// fills it; other strategies leave it `None` (and it is omitted from JSON).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetBreakdown {
    pub level: String,
    pub budget_score: f64,
    pub recommended_model_tier: String,
    pub confidence: f64,
    pub dimensions: DimensionScores,
    pub reason_codes: Vec<String>,
    pub needs_tool: bool,
    pub tool_type: Option<String>,
    pub requires_verifier: bool,
    pub fallback_policy: String,
}
```

Then change the `Recommendation` struct to add the optional field:

```rust
/// The router's final output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Recommendation {
    pub difficulty: Difficulty,
    pub ranking: Vec<RankedModel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget: Option<BudgetBreakdown>,
}
```

Update the existing `recommendation_serializes_to_expected_shape` test's literal to add `budget: None`:

```rust
        let rec = Recommendation {
            difficulty: Difficulty {
                score: 0.5,
                signals: vec!["code".into()],
            },
            ranking: vec![RankedModel {
                id: "m".into(),
                score: 0.4,
                reason: "r".into(),
            }],
            budget: None,
        };
```

- [ ] **Step 4: Add `budget: None` to the two existing routers**

In `crates/core/src/router.rs`, in `HeuristicRouter::recommend`, change the returned literal:

```rust
        Recommendation {
            difficulty,
            ranking,
            budget: None,
        }
```

In `crates/core/src/learned/mod.rs`, in `LearnedRouter::recommend`, change the returned literal the same way:

```rust
        Recommendation {
            difficulty,
            ranking,
            budget: None,
        }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p route-llm-core --release`
Expected: all core tests **pass** (new + existing, including the frozen heuristic/learned tests).

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/model.rs crates/core/src/router.rs crates/core/src/learned/mod.rs
git commit -m "$(cat <<'EOF'
feat(core): additive budget output types + Recommendation.budget

Add Policy, DimensionScores, BudgetBreakdown and an optional, skip-if-none
`budget` field on Recommendation. v1/v2 routers return `budget: None`; their
serialized output is byte-identical. No behavior change.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `budget/dims.rs` — six dimensions + budget score

**Files:**
- Create: `crates/core/src/budget/dims.rs`
- Modify: `crates/core/src/lib.rs` (add `pub mod budget;` and `crates/core/src/budget/mod.rs` stub)

- [ ] **Step 1: Create the module tree**

In `crates/core/src/lib.rs`, add `pub mod budget;` after `pub mod ranker;` (keep alphabetical-ish with the others):

```rust
pub mod budget;
pub mod difficulty;
pub mod learned;
pub mod model;
pub mod ranker;
pub mod registry;
pub mod router;
```

Create `crates/core/src/budget/mod.rs` with only the submodule declarations for now:

```rust
//! Budget router subsystem (v3). Isolated from v1 heuristic & v2 learned cores.
pub mod dims;
```

- [ ] **Step 2: Write the failing tests**

Create `crates/core/src/budget/dims.rs`:

```rust
//! The six reasoning-budget dimensions: canonical order, scales, weights, and
//! scoring via six per-dimension LinearModels that share v2's feature vector.
//! Pure & deterministic. See SPEC-v3 §4.

use crate::learned::model::LinearModel;
use crate::model::DimensionScores;

/// Bump when the dimension label schema or order changes; `weights.rs` matches.
pub const BUDGET_SCHEMA_VERSION: u32 = 1;

/// Number of dimensions.
pub const N_DIMS: usize = 6;

/// Canonical dimension order. Load-bearing: heads, scales, and weights index by it.
pub const DIM_NAMES: [&str; N_DIMS] = [
    "reasoning_depth",
    "verification_difficulty",
    "constraint_density",
    "context_integration",
    "ambiguity",
    "error_cost",
];

/// Max integer each dimension's human rubric uses (SPEC-v3 §4.1).
pub const DIM_SCALES: [f64; N_DIMS] = [4.0, 4.0, 4.0, 4.0, 3.0, 4.0];

/// RBC weights (SPEC-v3 §4.1 / §6).
pub const DIM_WEIGHTS: [f64; N_DIMS] = [1.4, 1.1, 1.0, 1.0, 0.8, 1.2];

/// Theoretical maximum budget_score = Σ weight_i · scale_i = 25.2 (SPEC-v3 §4.1).
pub const MAX_BUDGET: f64 = 25.2;

/// Score all six dimensions for a query: each head outputs p∈(0,1) (logistic),
/// rescaled to its integer scale.
pub fn score_dims(models: &[LinearModel], query: &str) -> [f64; N_DIMS] {
    debug_assert_eq!(models.len(), N_DIMS, "expected six dimension heads");
    let mut out = [0.0; N_DIMS];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = models[i].difficulty(query).score * DIM_SCALES[i];
    }
    out
}

/// Weighted budget score from dimension values (SPEC-v3 §4.1 formula).
pub fn budget_score(dims: &[f64; N_DIMS]) -> f64 {
    (0..N_DIMS).map(|i| DIM_WEIGHTS[i] * dims[i]).sum()
}

/// Per-dimension contribution weight_i·dim_i (for reason_codes).
pub fn contributions(dims: &[f64; N_DIMS]) -> [f64; N_DIMS] {
    let mut c = [0.0; N_DIMS];
    for (i, slot) in c.iter_mut().enumerate() {
        *slot = DIM_WEIGHTS[i] * dims[i];
    }
    c
}

/// Convert the fixed array to the named output struct (canonical order).
pub fn to_scores(dims: &[f64; N_DIMS]) -> DimensionScores {
    DimensionScores {
        reasoning_depth: dims[0],
        verification_difficulty: dims[1],
        constraint_density: dims[2],
        context_integration: dims[3],
        ambiguity: dims[4],
        error_cost: dims[5],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::features::{feature_count, SCHEMA_VERSION};

    fn head_with(weight_idx: usize, w: f64) -> LinearModel {
        let n = feature_count();
        let mut weights = vec![0.0; n];
        weights[weight_idx] = w;
        LinearModel {
            schema_version: SCHEMA_VERSION,
            weights,
            bias: 0.0,
            means: vec![0.0; n],
            stds: vec![1.0; n],
        }
    }

    #[test]
    fn budget_score_matches_weighted_sum() {
        let dims = DIM_SCALES; // all-max dimensions
        assert!((budget_score(&dims) - MAX_BUDGET).abs() < 1e-9);
    }

    #[test]
    fn score_dims_stays_in_each_scale() {
        let models: Vec<LinearModel> = (0..N_DIMS).map(|_| head_with(0, 0.0)).collect();
        let d = score_dims(&models, "hello world");
        for i in 0..N_DIMS {
            assert!(d[i] >= 0.0 && d[i] <= DIM_SCALES[i], "dim {i} = {}", d[i]);
        }
    }

    #[test]
    fn positive_reasoning_head_raises_that_dimension() {
        // feature index 3 == reasoning_hits (see learned::features BASE_FEATURE_NAMES).
        let mut models: Vec<LinearModel> = (0..N_DIMS).map(|_| head_with(0, 0.0)).collect();
        models[0] = head_with(3, 5.0);
        let hard = score_dims(&models, "prove and derive step by step; analyze")[0];
        let easy = score_dims(&models, "hello")[0];
        assert!(hard > easy, "reasoning dim hard {hard} vs easy {easy}");
    }
}
```

- [ ] **Step 3: Run tests to verify they fail, then pass**

Run: `cargo test -p route-llm-core --release budget::dims`
Expected: compiles and **3 tests pass** (the module is fully implemented above; this task is test-first by construction — if anything fails, fix `dims.rs` before continuing).

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/lib.rs crates/core/src/budget/mod.rs crates/core/src/budget/dims.rs
git commit -m "$(cat <<'EOF'
feat(core): budget dimensions — scales, weights, six-head scoring

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `budget/level.rs` — R0–R4, thresholds, tier, difficulty mapping

**Files:**
- Create: `crates/core/src/budget/level.rs`
- Modify: `crates/core/src/budget/mod.rs` (add `pub mod level;`)

- [ ] **Step 1: Register the module**

In `crates/core/src/budget/mod.rs`, add under the existing `pub mod dims;`:

```rust
pub mod level;
```

- [ ] **Step 2: Write the module + tests**

Create `crates/core/src/budget/level.rs`:

```rust
//! Map budget_score → R0..R4 → model tier → ranker difficulty. See SPEC-v3 §5.

use crate::budget::dims::MAX_BUDGET;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    R0,
    R1,
    R2,
    R3,
    R4,
}

/// Upper thresholds (exclusive) separating R0|R1|R2|R3; R4 is the open top (SPEC-v3 §5.1).
const THRESHOLDS: [f64; 4] = [4.0, 8.0, 12.0, 17.0];

impl Level {
    pub fn index(self) -> usize {
        match self {
            Level::R0 => 0,
            Level::R1 => 1,
            Level::R2 => 2,
            Level::R3 => 3,
            Level::R4 => 4,
        }
    }

    pub fn from_index(i: usize) -> Level {
        match i {
            0 => Level::R0,
            1 => Level::R1,
            2 => Level::R2,
            3 => Level::R3,
            _ => Level::R4,
        }
    }

    pub fn label(self) -> &'static str {
        ["R0", "R1", "R2", "R3", "R4"][self.index()]
    }

    pub fn tier(self) -> &'static str {
        ["tiny", "small", "medium", "strong", "best"][self.index()]
    }

    /// Step up/down, clamped to R0..R4.
    pub fn shift(self, delta: i32) -> Level {
        Level::from_index((self.index() as i32 + delta).clamp(0, 4) as usize)
    }

    /// The higher of two levels.
    pub fn max_with(self, other: Level) -> Level {
        Level::from_index(self.index().max(other.index()))
    }
}

/// Bucket a budget_score into a level.
pub fn level_of(score: f64) -> Level {
    if score < THRESHOLDS[0] {
        Level::R0
    } else if score < THRESHOLDS[1] {
        Level::R1
    } else if score < THRESHOLDS[2] {
        Level::R2
    } else if score < THRESHOLDS[3] {
        Level::R3
    } else {
        Level::R4
    }
}

/// `[lower, upper)` budget bounds of a level (R4 upper = MAX_BUDGET).
pub fn bounds(level: Level) -> (f64, f64) {
    match level {
        Level::R0 => (0.0, THRESHOLDS[0]),
        Level::R1 => (THRESHOLDS[0], THRESHOLDS[1]),
        Level::R2 => (THRESHOLDS[1], THRESHOLDS[2]),
        Level::R3 => (THRESHOLDS[2], THRESHOLDS[3]),
        Level::R4 => (THRESHOLDS[3], MAX_BUDGET),
    }
}

/// Raw estimator difficulty for the ranker: budget_score normalized to [0,1]
/// (SPEC-v3 §5.3). Monotonic in budget_score.
pub fn raw_difficulty(score: f64) -> f64 {
    (score / MAX_BUDGET).clamp(0.0, 1.0)
}

/// Difficulty floor implied by a (possibly escalated) level's lower bound.
pub fn level_floor_difficulty(level: Level) -> f64 {
    (bounds(level).0 / MAX_BUDGET).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholds_bucket_correctly() {
        assert_eq!(level_of(3.9), Level::R0);
        assert_eq!(level_of(4.0), Level::R1);
        assert_eq!(level_of(11.9), Level::R2);
        assert_eq!(level_of(12.0), Level::R3);
        assert_eq!(level_of(16.9), Level::R3);
        assert_eq!(level_of(17.0), Level::R4);
        assert_eq!(level_of(99.0), Level::R4);
    }

    #[test]
    fn tier_and_label_track_index() {
        assert_eq!(Level::R0.tier(), "tiny");
        assert_eq!(Level::R3.tier(), "strong");
        assert_eq!(Level::R4.label(), "R4");
    }

    #[test]
    fn shift_and_max_clamp() {
        assert_eq!(Level::R0.shift(-1), Level::R0);
        assert_eq!(Level::R4.shift(1), Level::R4);
        assert_eq!(Level::R1.shift(2), Level::R3);
        assert_eq!(Level::R1.max_with(Level::R3), Level::R3);
    }

    #[test]
    fn difficulty_is_in_unit_interval_and_monotonic() {
        assert!((raw_difficulty(0.0) - 0.0).abs() < 1e-9);
        assert!((raw_difficulty(MAX_BUDGET) - 1.0).abs() < 1e-9);
        assert!(raw_difficulty(20.0) > raw_difficulty(5.0));
        assert!(level_floor_difficulty(Level::R3) > level_floor_difficulty(Level::R1));
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p route-llm-core --release budget::level`
Expected: **4 tests pass**.

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/budget/mod.rs crates/core/src/budget/level.rs
git commit -m "$(cat <<'EOF'
feat(core): budget level R0–R4 — thresholds, tier, difficulty mapping

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `budget/escalation.rs` — decision layer

**Files:**
- Create: `crates/core/src/budget/escalation.rs`
- Modify: `crates/core/src/budget/mod.rs` (add `pub mod escalation;`)

- [ ] **Step 1: Register the module**

In `crates/core/src/budget/mod.rs`, add:

```rust
pub mod escalation;
```

- [ ] **Step 2: Write the module + tests**

Create `crates/core/src/budget/escalation.rs`:

```rust
//! Deterministic decision layer: risk floors, latest-info → tool, confidence,
//! two-estimator disagreement, and policy modes. Zero network. See SPEC-v3 §6.

use crate::budget::dims::{contributions, N_DIMS};
use crate::budget::level::{bounds, Level};
use crate::model::Policy;

/// Output of the decision layer.
pub struct Decision {
    pub level: Level,
    pub confidence: f64,
    pub needs_tool: bool,
    pub tool_type: Option<String>,
    pub requires_verifier: bool,
    pub fallback_policy: String,
    pub reason_codes: Vec<String>,
}

const HIGH_RISK: &[&str] = &[
    "legal", "lawsuit", "contract", "法律", "合約",
    "medical", "diagnosis", "health", "醫療", "病歷",
    "invest", "stock", "financial", "金融", "投資", "股票",
    "security", "vulnerability", "exploit", "資安", "漏洞",
    "production", "deploy", "生產環境", "部署",
    "pii", "personal data", "privacy", "個資", "隱私",
];

const LATEST_INFO: &[&str] = &[
    "today", "latest", "current", "right now", "this week",
    "今天", "最新", "現在", "目前",
    "exchange rate", "匯率", "stock price", "股價",
    "news", "新聞", "ceo of", "who is the current",
];

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

/// Confidence = ½·boundary_margin + ½·estimator_agreement (SPEC-v3 §6.2).
pub fn confidence(score: f64, budget_level: Level, learned_level: Level) -> f64 {
    let (lo, hi) = bounds(budget_level);
    let half = ((hi - lo) / 2.0).max(1e-9);
    let margin = ((score - lo).min(hi - score) / half).clamp(0.0, 1.0);
    let agreement =
        1.0 - (budget_level.index() as f64 - learned_level.index() as f64).abs() / 4.0;
    (0.5 * margin + 0.5 * agreement).clamp(0.0, 1.0)
}

/// Top-2 dimensions by contribution → reason codes (canonical order; SPEC-v3 §6/§7).
fn dimension_reason_codes(dims: &[f64; N_DIMS]) -> Vec<String> {
    const CODE: [&str; N_DIMS] = [
        "multi_step_reasoning",
        "needs_validation",
        "constraint_dense",
        "context_heavy",
        "ambiguous",
        "high_error_cost",
    ];
    let contrib = contributions(dims);
    let mut idx: Vec<usize> = (0..N_DIMS).collect();
    idx.sort_by(|&a, &b| {
        contrib[b]
            .partial_cmp(&contrib[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b)) // stable tie-break by index
    });
    idx.into_iter()
        .filter(|&i| contrib[i] > 0.0)
        .take(2)
        .map(|i| CODE[i].to_string())
        .collect()
}

/// Apply the full decision layer. `query_lower` is the lowercased query.
pub fn decide(
    policy: Policy,
    query_lower: &str,
    dims: &[f64; N_DIMS],
    score: f64,
    base_level: Level,
    learned_level: Level,
) -> Decision {
    let mut level = base_level;
    let mut reason_codes = dimension_reason_codes(dims);
    let conf = confidence(score, base_level, learned_level);

    // (2) high-risk floors
    let high_risk = contains_any(query_lower, HIGH_RISK);
    if high_risk {
        level = level.max_with(Level::R3);
        reason_codes.push("high_risk_domain".into());
        if dims[0] >= 3.0 {
            // deep reasoning in a risk domain → expert tier
            level = level.max_with(Level::R4);
        }
    }

    // (3) latest info → tool, no level change
    let needs_tool = contains_any(query_lower, LATEST_INFO);
    let tool_type = if needs_tool {
        reason_codes.push("needs_latest_info".into());
        Some("web_search".to_string())
    } else {
        None
    };

    let dlevel = (base_level.index() as i32 - learned_level.index() as i32).abs();
    let mut requires_verifier = false;
    let mut fallback_policy = "none".to_string();

    // (4)+(5) policy-specific
    match policy {
        Policy::Balanced => {
            if conf < 0.7 {
                level = level.shift(1);
                reason_codes.push("low_confidence".into());
            }
        }
        Policy::Strict => {
            if conf < 0.85 {
                level = level.shift(1);
                reason_codes.push("low_confidence".into());
            }
            if dlevel >= 1 {
                level = level.shift(1);
                reason_codes.push("estimator_disagreement".into());
            }
            if high_risk {
                level = level.max_with(Level::R4);
            }
            if base_level.index() >= Level::R3.index() {
                requires_verifier = true;
            }
        }
        Policy::Cheap => {
            if !high_risk && dims[5] <= 1.0 {
                level = level.shift(-1);
            }
            requires_verifier = true;
            fallback_policy = "upgrade_if_verifier_fails".to_string();
        }
    }

    // (6) strong disagreement → max + verifier (all policies)
    if dlevel >= 2 {
        level = level.max_with(Level::from_index(
            base_level.index().max(learned_level.index()),
        ));
        requires_verifier = true;
        if !reason_codes.iter().any(|c| c == "estimator_disagreement") {
            reason_codes.push("estimator_disagreement".into());
        }
    }

    Decision {
        level,
        confidence: conf,
        needs_tool,
        tool_type,
        requires_verifier,
        fallback_policy,
        reason_codes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZERO: [f64; N_DIMS] = [0.0; N_DIMS];

    #[test]
    fn high_risk_floors_to_at_least_r3() {
        let d = decide(Policy::Balanced, "review this legal contract", &ZERO, 1.0, Level::R0, Level::R0);
        assert!(d.level.index() >= Level::R3.index());
        assert!(d.reason_codes.iter().any(|c| c == "high_risk_domain"));
    }

    #[test]
    fn latest_info_sets_tool_without_raising_level() {
        let d = decide(Policy::Balanced, "what is the latest exchange rate", &ZERO, 10.0, Level::R2, Level::R2);
        assert!(d.needs_tool);
        assert_eq!(d.tool_type.as_deref(), Some("web_search"));
        assert_eq!(d.level, Level::R2, "latest-info must not raise the level");
    }

    #[test]
    fn strong_disagreement_requires_verifier_and_takes_max() {
        let d = decide(Policy::Balanced, "x", &ZERO, 2.0, Level::R0, Level::R4);
        assert!(d.requires_verifier);
        assert_eq!(d.level, Level::R4);
        assert!(d.reason_codes.iter().any(|c| c == "estimator_disagreement"));
    }

    #[test]
    fn cheap_downgrades_low_risk_and_flags_fallback() {
        let d = decide(Policy::Cheap, "summarize this paragraph", &ZERO, 9.0, Level::R2, Level::R2);
        assert_eq!(d.level, Level::R1, "cheap downgrades a low-risk task");
        assert!(d.requires_verifier);
        assert_eq!(d.fallback_policy, "upgrade_if_verifier_fails");
    }

    #[test]
    fn strict_adds_verifier_for_hard_base() {
        let d = decide(Policy::Strict, "design a system", &ZERO, 13.0, Level::R3, Level::R3);
        assert!(d.requires_verifier);
    }

    #[test]
    fn reason_codes_pick_top_contributors() {
        let mut dims = ZERO;
        dims[0] = 4.0; // reasoning_depth
        dims[5] = 4.0; // error_cost
        let d = decide(Policy::Balanced, "x", &dims, 10.0, Level::R2, Level::R2);
        assert!(d.reason_codes.iter().any(|c| c == "multi_step_reasoning"));
        assert!(d.reason_codes.iter().any(|c| c == "high_error_cost"));
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p route-llm-core --release budget::escalation`
Expected: **6 tests pass**.

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/budget/mod.rs crates/core/src/budget/escalation.rs
git commit -m "$(cat <<'EOF'
feat(core): budget decision layer — risk/tool/confidence/disagreement/policy

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `budget/weights.rs` — shipped heads (zero placeholder)

**Files:**
- Create: `crates/core/src/budget/weights.rs`
- Modify: `crates/core/src/budget/mod.rs` (add `pub mod weights;`)

- [ ] **Step 1: Register the module**

In `crates/core/src/budget/mod.rs`, add:

```rust
pub mod weights;
```

- [ ] **Step 2: Write the placeholder + test**

Create `crates/core/src/budget/weights.rs`. This is a GENERATED file (Task 16 overwrites it via `trainer fit-budget`); until real labels exist it returns six zero heads (each query → 0.5·scale), which compiles and is deterministic:

```rust
//! Shipped budget dimension heads. GENERATED by `trainer fit-budget` — do not edit by hand.
//! Placeholder (zero heads) until real 6-dim labels are fit; see PLAN-v3 Task 16.
use crate::budget::dims::{BUDGET_SCHEMA_VERSION, N_DIMS};
use crate::learned::features::{feature_count, SCHEMA_VERSION};
use crate::learned::model::LinearModel;

pub fn shipped_dim_models() -> Vec<LinearModel> {
    assert_eq!(SCHEMA_VERSION, 1, "weights/features schema mismatch");
    assert_eq!(BUDGET_SCHEMA_VERSION, 1, "budget schema mismatch");
    let n = feature_count();
    (0..N_DIMS)
        .map(|_| LinearModel {
            schema_version: SCHEMA_VERSION,
            weights: vec![0.0; n],
            bias: 0.0,
            means: vec![0.0; n],
            stds: vec![1.0; n],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ships_six_well_formed_heads() {
        let models = shipped_dim_models();
        assert_eq!(models.len(), N_DIMS);
        for m in &models {
            assert_eq!(m.weights.len(), feature_count());
            assert_eq!(m.means.len(), feature_count());
            assert_eq!(m.stds.len(), feature_count());
        }
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p route-llm-core --release budget::weights`
Expected: **1 test passes**.

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/budget/mod.rs crates/core/src/budget/weights.rs
git commit -m "$(cat <<'EOF'
feat(core): budget weights placeholder (zero heads until fit-budget)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `budget/mod.rs` — `BudgetRouter` + lib re-exports

**Files:**
- Modify: `crates/core/src/budget/mod.rs` (add the router)
- Modify: `crates/core/src/lib.rs` (re-exports)

- [ ] **Step 1: Write the failing tests**

Replace the contents of `crates/core/src/budget/mod.rs` with (submodule decls kept, router added):

```rust
//! Budget router subsystem (v3). Isolated from v1 heuristic & v2 learned cores.
pub mod dims;
pub mod escalation;
pub mod level;
pub mod weights;

use crate::learned::model::LinearModel;
use crate::learned::weights::shipped_model as learned_shipped_model;
use crate::model::{BudgetBreakdown, Difficulty, ModelProfile, Policy, Recommendation, RoutingPreferences};
use crate::ranker;
use crate::router::Router;

/// v3 strategy: six learned budget dimensions → R0..R4 + decision layer → shared ranker.
pub struct BudgetRouter {
    dim_models: Vec<LinearModel>,
    policy: Policy,
}

impl Default for BudgetRouter {
    fn default() -> Self {
        Self {
            dim_models: weights::shipped_dim_models(),
            policy: Policy::Balanced,
        }
    }
}

impl BudgetRouter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Startup-configured policy (server reads `ROUTE_LLM_POLICY`).
    pub fn with_policy(policy: Policy) -> Self {
        Self {
            dim_models: weights::shipped_dim_models(),
            policy,
        }
    }

    /// For tests: inject dimension heads directly.
    pub fn with_models(dim_models: Vec<LinearModel>, policy: Policy) -> Self {
        Self { dim_models, policy }
    }

    /// Raw estimator difficulty (pre-escalation) — for offline gold eval (SPEC-v3 §5.3 / §8 axis A).
    pub fn raw_difficulty(&self, query: &str) -> f64 {
        let d = dims::score_dims(&self.dim_models, query);
        level::raw_difficulty(dims::budget_score(&d))
    }
}

impl Router for BudgetRouter {
    fn recommend(
        &self,
        query: &str,
        models: &[ModelProfile],
        prefs: &RoutingPreferences,
    ) -> Recommendation {
        let lower = query.to_lowercase();
        let dim_arr = dims::score_dims(&self.dim_models, query);
        let score = dims::budget_score(&dim_arr);
        let base_level = level::level_of(score);

        // Second, independent estimator: v2 learned scalar mapped onto the R-scale.
        let learned_diff = learned_shipped_model().difficulty(query).score;
        let learned_level = level::level_of(learned_diff * dims::MAX_BUDGET);

        let decision =
            escalation::decide(self.policy, &lower, &dim_arr, score, base_level, learned_level);

        // Runtime difficulty = max(raw estimator, escalated level floor) — SPEC-v3 §5.3.
        let difficulty_score =
            level::raw_difficulty(score).max(level::level_floor_difficulty(decision.level));

        let difficulty = Difficulty {
            score: difficulty_score,
            signals: decision.reason_codes.clone(),
        };
        let ranking = ranker::rank(&difficulty, models, prefs);

        let budget = BudgetBreakdown {
            level: decision.level.label().to_string(),
            budget_score: score,
            recommended_model_tier: decision.level.tier().to_string(),
            confidence: decision.confidence,
            dimensions: dims::to_scores(&dim_arr),
            reason_codes: decision.reason_codes,
            needs_tool: decision.needs_tool,
            tool_type: decision.tool_type,
            requires_verifier: decision.requires_verifier,
            fallback_policy: decision.fallback_policy,
        };

        Recommendation {
            difficulty,
            ranking,
            budget: Some(budget),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::features::{feature_count, SCHEMA_VERSION};

    fn head_with(weight_idx: usize, w: f64) -> LinearModel {
        let n = feature_count();
        let mut weights = vec![0.0; n];
        weights[weight_idx] = w;
        LinearModel {
            schema_version: SCHEMA_VERSION,
            weights,
            bias: 0.0,
            means: vec![0.0; n],
            stds: vec![1.0; n],
        }
    }

    fn models() -> Vec<ModelProfile> {
        vec![
            ModelProfile { id: "strong".into(), quality: 0.97, cost: 0.90 },
            ModelProfile { id: "cheap".into(), quality: 0.60, cost: 0.10 },
        ]
    }

    /// Six heads where reasoning_depth (dim 0) responds to the reasoning feature (idx 3).
    fn reasoning_router() -> BudgetRouter {
        let mut heads: Vec<LinearModel> = (0..dims::N_DIMS).map(|_| head_with(0, 0.0)).collect();
        heads[0] = head_with(3, 6.0);
        BudgetRouter::with_models(heads, Policy::Balanced)
    }

    #[test]
    fn produces_full_ranking_with_budget_block() {
        let rec = reasoning_router().recommend("hello", &models(), &RoutingPreferences::default());
        assert_eq!(rec.ranking.len(), 2);
        let b = rec.budget.expect("budget present");
        assert!(b.level.starts_with('R'));
        assert!(!b.recommended_model_tier.is_empty());
        assert!(b.confidence >= 0.0 && b.confidence <= 1.0);
    }

    #[test]
    fn harder_query_scores_higher_budget() {
        let r = reasoning_router();
        let easy = r.recommend("hi", &models(), &RoutingPreferences::default());
        let hard = r.recommend(
            "prove step by step and derive the invariant; analyze the partition",
            &models(),
            &RoutingPreferences::default(),
        );
        let eb = easy.budget.unwrap().budget_score;
        let hb = hard.budget.unwrap().budget_score;
        assert!(hb > eb, "hard {hb} vs easy {eb}");
    }
}
```

- [ ] **Step 2: Add re-exports**

In `crates/core/src/lib.rs`, update the re-export lines:

```rust
pub use budget::BudgetRouter;
pub use learned::LearnedRouter;
pub use model::{
    BudgetBreakdown, Difficulty, DimensionScores, ModelProfile, Policy, RankedModel,
    Recommendation, RoutingPreferences,
};
pub use registry::CandidateInput;
pub use router::{HeuristicRouter, Router};
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p route-llm-core --release`
Expected: all core tests pass, including the 2 new `budget::tests`.

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/budget/mod.rs crates/core/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(core): BudgetRouter strategy (dims → level → decision → ranker)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase B — server wiring

## Task 7: server — `ROUTE_LLM_ROUTER=budget` + `ROUTE_LLM_POLICY`

**Files:**
- Modify: `crates/server/src/main.rs`
- Modify: `crates/server/src/handlers.rs`
- Create: `crates/server/tests/budget.rs`
- Modify: `README.md`

- [ ] **Step 1: Write the failing unit tests (main.rs)**

In `crates/server/src/main.rs`, add to the `tests` module:

```rust
    #[test]
    fn explicit_budget_selects_budget() {
        assert_eq!(super::choose_router(Ok("budget")), Ok("budget"));
    }

    #[test]
    fn policy_parses_known_values_and_defaults() {
        use route_llm_core::Policy;
        assert_eq!(super::choose_policy(Ok("strict")), Ok(Policy::Strict));
        assert_eq!(super::choose_policy(Ok("cheap")), Ok(Policy::Cheap));
        assert_eq!(
            super::choose_policy(Err(&std::env::VarError::NotPresent)),
            Ok(Policy::Balanced)
        );
        assert!(super::choose_policy(Ok("bogus")).is_err());
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p route-llm-server --release explicit_budget_selects_budget`
Expected: compile error — `choose_router` doesn't accept `"budget"`, `choose_policy` undefined.

- [ ] **Step 3: Implement**

In `crates/server/src/main.rs`, add `Ok("budget") => Ok("budget"),` to `choose_router`:

```rust
    match var {
        Ok("heuristic") => Ok("heuristic"),
        Ok("learned") => Ok("learned"),
        Ok("budget") => Ok("budget"),
        // Genuinely unset → default to learned (SPEC-v2 §9)
        Err(std::env::VarError::NotPresent) => Ok("learned"),
        Err(_) => Err("ROUTE_LLM_ROUTER value is not valid UTF-8".to_string()),
        Ok(other) => Err(format!(
            "invalid ROUTE_LLM_ROUTER: {other:?} (expected 'learned', 'heuristic', or 'budget')"
        )),
    }
```

Add `choose_policy` directly below `choose_router`:

```rust
/// Resolves ROUTE_LLM_POLICY (for the budget router) to a `Policy`.
fn choose_policy(
    var: Result<&str, &std::env::VarError>,
) -> Result<route_llm_core::Policy, String> {
    use route_llm_core::Policy;
    match var {
        Ok("balanced") => Ok(Policy::Balanced),
        Ok("strict") => Ok(Policy::Strict),
        Ok("cheap") => Ok(Policy::Cheap),
        Err(std::env::VarError::NotPresent) => Ok(Policy::Balanced),
        Err(_) => Err("ROUTE_LLM_POLICY value is not valid UTF-8".to_string()),
        Ok(other) => Err(format!(
            "invalid ROUTE_LLM_POLICY: {other:?} (expected 'balanced', 'strict', or 'cheap')"
        )),
    }
}
```

In `main`, after resolving `router_name`, resolve the policy and add a `"budget"` arm to the builder match:

```rust
    let policy = match choose_policy(std::env::var("ROUTE_LLM_POLICY").as_deref()) {
        Ok(p) => p,
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(1);
        }
    };

    let (router, router_name): (route_llm_server::SharedRouter, &'static str) = match router_name {
        "heuristic" => (
            std::sync::Arc::new(route_llm_core::HeuristicRouter),
            "heuristic",
        ),
        "budget" => (
            std::sync::Arc::new(route_llm_core::BudgetRouter::with_policy(policy)),
            "budget",
        ),
        _ => (
            std::sync::Arc::new(route_llm_core::LearnedRouter::new()),
            "learned",
        ),
    };
```

- [ ] **Step 4: `summary_line` shows the level (handlers.rs)**

In `crates/server/src/handlers.rs`, change `summary_line` to append the budget level/tier when present:

```rust
pub(crate) fn summary_line(rec: &Recommendation) -> String {
    let order = rec
        .ranking
        .iter()
        .map(|r| r.id.as_str())
        .collect::<Vec<_>>()
        .join(" > ");
    let top = rec
        .ranking
        .first()
        .map(|r| r.id.as_str())
        .unwrap_or("(none)");
    let budget = match &rec.budget {
        Some(b) => format!(" [{} / {}]", b.level, b.recommended_model_tier),
        None => String::new(),
    };
    format!(
        "Recommended: {} (difficulty {:.2}){}. Order: {}.",
        top, rec.difficulty.score, budget, order
    )
}
```

- [ ] **Step 5: Write the server integration test**

Create `crates/server/tests/budget.rs` (mirrors `tests/learned.rs`):

```rust
use axum_test::TestServer;
use route_llm_core::BudgetRouter;
use serde_json::json;

fn server() -> TestServer {
    let router = std::sync::Arc::new(BudgetRouter::new());
    TestServer::new(route_llm_server::app_with_router(router)).unwrap()
}

#[tokio::test]
async fn recommend_includes_budget_block() {
    let res = server()
        .post("/v1/recommend")
        .json(&json!({
            "query": "Design a production-grade distributed lock and analyze deadlock risk.",
            "models": [{"id": "claude-opus-4-8"}, {"id": "claude-haiku-4-5"}]
        }))
        .await;
    res.assert_status_ok();
    let v: serde_json::Value = res.json();
    assert!(v["ranking"].as_array().unwrap().len() == 2);
    let b = &v["budget"];
    assert!(b["level"].as_str().unwrap().starts_with('R'));
    assert!(b["recommended_model_tier"].is_string());
    assert!(b["confidence"].is_number());
}
```

- [ ] **Step 6: Run server tests**

Run: `cargo test -p route-llm-server --release`
Expected: all pass, including `budget::recommend_includes_budget_block` and the existing `learned`/`heuristic` suites.

- [ ] **Step 7: Document the new env vars (README.md)**

In `README.md`, under the "v2: Learned router" select block, replace the two `ROUTE_LLM_ROUTER` lines with:

```bash
ROUTE_LLM_ROUTER=learned   cargo run --release -p route-llm-server   # default
ROUTE_LLM_ROUTER=heuristic cargo run --release -p route-llm-server   # v1 fallback
ROUTE_LLM_ROUTER=budget    cargo run --release -p route-llm-server   # v3 Reasoning Budget Router
# Budget policy (only for the budget router): balanced (default) | strict | cheap
ROUTE_LLM_ROUTER=budget ROUTE_LLM_POLICY=strict cargo run --release -p route-llm-server
```

- [ ] **Step 8: Commit**

```bash
git add crates/server/src/main.rs crates/server/src/handlers.rs crates/server/tests/budget.rs README.md
git commit -m "$(cat <<'EOF'
feat(server): select budget router via ROUTE_LLM_ROUTER + ROUTE_LLM_POLICY

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase C — trainer: offline 6-dim labeling, fit, eval

## Task 8: `dataset.rs` — `DimScores` / `DimsExample` + jsonl I/O

**Files:**
- Modify: `crates/trainer/src/dataset.rs`

- [ ] **Step 1: Write the failing test**

Append to the `tests` module in `crates/trainer/src/dataset.rs`:

```rust
    #[test]
    fn dims_example_round_trips() {
        let items = vec![DimsExample {
            query: "prove X".into(),
            category: "math".into(),
            dims: DimScores {
                reasoning_depth: 4,
                verification_difficulty: 3,
                constraint_density: 1,
                context_integration: 0,
                ambiguity: 2,
                error_cost: 1,
            },
        }];
        let s = to_dims_jsonl(&items);
        assert_eq!(parse_dims_jsonl(&s).unwrap(), items);
        assert!(s.contains("\"reasoning_depth\":4"));
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p route-llm-trainer --release dims_example_round_trips`
Expected: compile error — types/functions undefined.

- [ ] **Step 3: Implement**

Append to `crates/trainer/src/dataset.rs` (after `LabeledExample` helpers):

```rust
/// One query's six budget-dimension integer ratings (frontier-LLM labeled).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DimScores {
    pub reasoning_depth: u8,
    pub verification_difficulty: u8,
    pub constraint_density: u8,
    pub context_integration: u8,
    pub ambiguity: u8,
    pub error_cost: u8,
}

/// A 6-dim labeled example: `data/budget.<labeler>.jsonl` line shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DimsExample {
    pub query: String,
    #[serde(default)]
    pub category: String,
    pub dims: DimScores,
}

pub fn parse_dims_jsonl(text: &str) -> Result<Vec<DimsExample>, String> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let ex: DimsExample =
            serde_json::from_str(line).map_err(|e| format!("line {}: {e}", i + 1))?;
        out.push(ex);
    }
    Ok(out)
}

pub fn to_dims_jsonl(items: &[DimsExample]) -> String {
    let mut s = items
        .iter()
        .map(|x| serde_json::to_string(x).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    s.push('\n');
    s
}

pub fn load_dims(path: &str) -> Result<Vec<DimsExample>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    parse_dims_jsonl(&text)
}

pub fn save_dims(path: &str, items: &[DimsExample]) -> Result<(), String> {
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
    }
    std::fs::write(path, to_dims_jsonl(items)).map_err(|e| format!("write {path}: {e}"))
}

/// Dimension value by canonical index 0..6 (matches `budget::dims::DIM_NAMES`).
pub fn dim_value(d: &DimScores, i: usize) -> f64 {
    [
        d.reasoning_depth,
        d.verification_difficulty,
        d.constraint_density,
        d.context_integration,
        d.ambiguity,
        d.error_cost,
    ][i] as f64
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p route-llm-trainer --release dims_example_round_trips`
Expected: **pass**.

- [ ] **Step 5: Commit**

```bash
git add crates/trainer/src/dataset.rs
git commit -m "$(cat <<'EOF'
feat(trainer): DimsExample/DimScores + budget jsonl I/O

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `budget_label.rs` — 6-dim prompt, parser, `fit_dims`

**Files:**
- Modify: `crates/trainer/src/label.rs` (make two helpers `pub(crate)`)
- Create: `crates/trainer/src/budget_label.rs`
- Modify: `crates/trainer/src/main.rs` (add `mod budget_label;`)

- [ ] **Step 1: Register the module + relax helper visibility**

In `crates/trainer/src/main.rs`, add `mod budget_label;` to the module list:

```rust
mod budget_label;
mod corpus;
mod dataset;
mod emit;
mod eval;
mod gold;
mod label;
mod logreg;
```

In `crates/trainer/src/label.rs`, change `fn http_client(` to `pub(crate) fn http_client(` and `fn chat_complete(` to `pub(crate) fn chat_complete(` (reuse from `budget_label`, no logic change).

- [ ] **Step 2: Write the failing tests**

Create `crates/trainer/src/budget_label.rs`:

```rust
//! Frontier-LLM labeling of the six reasoning-budget dimensions (offline only).
//! Reuses `label`'s OpenAI-compatible client. Writes `data/budget.<model>.jsonl`.
//! Never used at inference. See SPEC-v3 §4.3.

use crate::dataset::{self, dim_value, DimScores, DimsExample};
use crate::label::{self, LabelConfig};
use route_llm_core::learned::model::LinearModel;

/// Per-dimension max integer (matches `budget::dims::DIM_SCALES`).
const SCALES: [u8; 6] = [4, 4, 4, 4, 3, 4];

/// Rubric prompt: asks for one `DIMS:` line with six integers in canonical order.
pub fn build_dims_prompt(query: &str) -> String {
    format!(
        "You rate how much reasoning a user query needs, on SIX independent axes.\n\
         Give an integer for each (ranges in brackets):\n\
         1. reasoning_depth [0-4]: layers of reasoning / problem decomposition\n\
         2. verification_difficulty [0-4]: how hard the answer is to check\n\
         3. constraint_density [0-4]: how many constraints must hold at once\n\
         4. context_integration [0-4]: how much context must be integrated\n\
         5. ambiguity [0-3]: how many reasonable interpretations\n\
         6. error_cost [0-4]: cost of being wrong\n\
         Reply with EXACTLY one line:\n\
         DIMS: <reasoning_depth> <verification_difficulty> <constraint_density> \
         <context_integration> <ambiguity> <error_cost>\n\
         then one short reason line.\n\n\
         Query: {query}\n"
    )
}

/// Parse six integers from a `DIMS:` line (preferred) or the first six ints found.
/// Each is clamped to its axis range. Returns None if fewer than six integers.
pub fn parse_dims(output: &str) -> Option<[u8; 6]> {
    let lower = output.to_lowercase();
    let slice = match lower.find("dims") {
        Some(idx) => &output[idx..],
        None => output,
    };
    let nums: Vec<u8> = slice
        .split(|c: char| !c.is_ascii_digit())
        .filter(|t| !t.is_empty())
        .filter_map(|t| t.parse::<u8>().ok())
        .take(6)
        .collect();
    if nums.len() < 6 {
        return None;
    }
    let mut out = [0u8; 6];
    for i in 0..6 {
        out[i] = nums[i].min(SCALES[i]);
    }
    Some(out)
}

/// Fit six per-dimension LinearModels by reusing the v2 logistic fitter: dimension
/// `i`'s target is `dim_i / scale_i ∈ [0,1]` (SPEC-v3 §4.2).
pub fn fit_dims(data: &[DimsExample]) -> Vec<LinearModel> {
    (0..6)
        .map(|i| {
            let examples: Vec<dataset::LabeledExample> = data
                .iter()
                .map(|d| dataset::LabeledExample {
                    query: d.query.clone(),
                    difficulty: (dim_value(&d.dims, i) / SCALES[i] as f64).clamp(0.0, 1.0),
                    category: d.category.clone(),
                })
                .collect();
            crate::logreg::fit(&examples, &crate::logreg::FitConfig::default())
        })
        .collect()
}

/// Sanitize a model id into a filename fragment (`/` and `:` → `-`).
fn sanitize(model: &str) -> String {
    model.replace(['/', ':', ' '], "-")
}

/// `label --dims`: read corpus, ask the LLM for six dims per query, write
/// `data/budget.<model>.jsonl`. Resumable: existing output rows are skipped.
/// The only networked step. NETWORK; not unit-tested.
pub fn run_dims() {
    let cfg = LabelConfig::from_env();
    let client = label::http_client().expect("build HTTP client");
    let corpus = dataset::load_corpus("data/corpus.jsonl")
        .expect("load data/corpus.jsonl (run `trainer synth` first)");
    let out_path = format!("data/budget.{}.jsonl", sanitize(&cfg.model));

    let mut done: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut labeled: Vec<DimsExample> = Vec::new();
    if let Ok(existing) = dataset::load_dims(&out_path) {
        for e in existing {
            done.insert(e.query.clone());
            labeled.push(e);
        }
    }

    let (mut ok, mut skip) = (0usize, 0usize);
    for (i, q) in corpus.iter().enumerate() {
        if done.contains(&q.query) {
            continue;
        }
        match label::chat_complete(&client, &cfg, &build_dims_prompt(&q.query)) {
            Ok(out) => match parse_dims(&out) {
                Some(d) => {
                    labeled.push(DimsExample {
                        query: q.query.clone(),
                        category: q.category.clone(),
                        dims: DimScores {
                            reasoning_depth: d[0],
                            verification_difficulty: d[1],
                            constraint_density: d[2],
                            context_integration: d[3],
                            ambiguity: d[4],
                            error_cost: d[5],
                        },
                    });
                    ok += 1;
                }
                None => {
                    eprintln!("skip (unparseable) [{i}]: {}", q.query);
                    skip += 1;
                }
            },
            Err(e) => {
                eprintln!("net-skip [{i}]: {}", e.lines().next().unwrap_or("network error"));
                skip += 1;
            }
        }
        if (i + 1) % 50 == 0 {
            let _ = dataset::save_dims(&out_path, &labeled); // checkpoint
            eprintln!("progress {}/{} (ok {ok}, skip {skip})", i + 1, corpus.len());
        }
    }
    dataset::save_dims(&out_path, &labeled).expect("save budget labels");
    eprintln!("label --dims: {ok} labeled, {skip} skipped -> {out_path}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_dims_reads_six_integers() {
        assert_eq!(
            parse_dims("DIMS: 3 2 2 1 1 2\nbecause ..."),
            Some([3, 2, 2, 1, 1, 2])
        );
    }

    #[test]
    fn parse_dims_clamps_to_axis_ranges() {
        // ambiguity (idx 4) max is 3; error_cost (idx 5) max is 4.
        assert_eq!(parse_dims("DIMS: 9 9 9 9 9 9"), Some([4, 4, 4, 4, 3, 4]));
    }

    #[test]
    fn parse_dims_rejects_too_few() {
        assert_eq!(parse_dims("DIMS: 3 2 1"), None);
        assert_eq!(parse_dims("no numbers here"), None);
    }

    #[test]
    fn prompt_lists_all_six_axes() {
        let p = build_dims_prompt("reverse a linked list");
        for axis in [
            "reasoning_depth",
            "verification_difficulty",
            "constraint_density",
            "context_integration",
            "ambiguity",
            "error_cost",
        ] {
            assert!(p.contains(axis), "missing {axis}");
        }
        assert!(p.contains("reverse a linked list"));
    }

    #[test]
    fn fit_dims_returns_six_models() {
        let data = vec![
            DimsExample {
                query: "hi".into(),
                category: "chat".into(),
                dims: DimScores {
                    reasoning_depth: 0,
                    verification_difficulty: 0,
                    constraint_density: 0,
                    context_integration: 0,
                    ambiguity: 0,
                    error_cost: 0,
                },
            },
            DimsExample {
                query: "prove and derive step by step; analyze".into(),
                category: "math".into(),
                dims: DimScores {
                    reasoning_depth: 4,
                    verification_difficulty: 4,
                    constraint_density: 3,
                    context_integration: 2,
                    ambiguity: 1,
                    error_cost: 3,
                },
            },
        ];
        let models = fit_dims(&data);
        assert_eq!(models.len(), 6);
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p route-llm-trainer --release budget_label`
Expected: **5 tests pass** (and `cargo build -p route-llm-trainer --release` is warning-clean).

- [ ] **Step 4: Commit**

```bash
git add crates/trainer/src/main.rs crates/trainer/src/label.rs crates/trainer/src/budget_label.rs
git commit -m "$(cat <<'EOF'
feat(trainer): 6-dim budget labeling (prompt/parse/run) + fit_dims

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: `emit.rs` — render `budget/weights.rs`

**Files:**
- Modify: `crates/trainer/src/emit.rs`

- [ ] **Step 1: Write the failing test**

Append to the `tests` module in `crates/trainer/src/emit.rs`:

```rust
    #[test]
    fn budget_weights_source_has_six_heads() {
        let n = feature_count();
        let head = LinearModel {
            schema_version: SCHEMA_VERSION,
            weights: vec![0.25; n],
            bias: -0.5,
            means: vec![0.0; n],
            stds: vec![1.0; n],
        };
        let models = vec![head; 6];
        let s = budget_weights_rs(&models);
        assert!(s.contains("pub fn shipped_dim_models()"));
        assert_eq!(s.matches("LinearModel {").count(), 6);
        assert!(s.contains("bias: -0.5"));
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p route-llm-trainer --release budget_weights_source_has_six_heads`
Expected: compile error — `budget_weights_rs` undefined.

- [ ] **Step 3: Implement**

Append to `crates/trainer/src/emit.rs`:

```rust
/// Render `crates/core/src/budget/weights.rs` from six fitted dimension heads.
pub fn budget_weights_rs(models: &[LinearModel]) -> String {
    let fmt = |v: &[f64]| {
        v.iter()
            .map(|x| format!("{x:?}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let heads = models
        .iter()
        .map(|m| {
            format!(
                "        LinearModel {{\n\
                 \x20           schema_version: {schema},\n\
                 \x20           weights: vec![{w}],\n\
                 \x20           bias: {b:?},\n\
                 \x20           means: vec![{me}],\n\
                 \x20           stds: vec![{st}],\n\
                 \x20       }},",
                schema = m.schema_version,
                w = fmt(&m.weights),
                b = m.bias,
                me = fmt(&m.means),
                st = fmt(&m.stds),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "//! Shipped budget dimension heads. GENERATED by `trainer fit-budget` — do not edit by hand.\n\
         use crate::budget::dims::BUDGET_SCHEMA_VERSION;\n\
         use crate::learned::features::SCHEMA_VERSION;\n\
         use crate::learned::model::LinearModel;\n\n\
         pub fn shipped_dim_models() -> Vec<LinearModel> {{\n\
         \x20   assert_eq!(SCHEMA_VERSION, 1, \"weights/features schema mismatch\");\n\
         \x20   assert_eq!(BUDGET_SCHEMA_VERSION, 1, \"budget schema mismatch\");\n\
         \x20   vec![\n{heads}\n    ]\n\
         }}\n"
    )
}

pub fn write_budget(models: &[LinearModel], path: &str) -> Result<(), String> {
    std::fs::write(path, budget_weights_rs(models)).map_err(|e| format!("write {path}: {e}"))
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p route-llm-trainer --release budget_weights_source_has_six_heads`
Expected: **pass**.

- [ ] **Step 5: Commit**

```bash
git add crates/trainer/src/emit.rs
git commit -m "$(cat <<'EOF'
feat(trainer): emit budget/weights.rs from six fitted dimension heads

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: `eval.rs` — `run_eval_budget` + `crosseval_dims`

**Files:**
- Modify: `crates/trainer/src/eval.rs`

- [ ] **Step 1: Write the failing tests**

Append to the `tests` module in `crates/trainer/src/eval.rs`:

```rust
    #[test]
    fn budget_router_difficulty_is_finite_on_gold_like_input() {
        // Uses the shipped (placeholder) heads; just checks the eval plumbing is finite.
        let router = route_llm_core::BudgetRouter::new();
        let d = router.raw_difficulty("prove and derive step by step");
        assert!(d.is_finite() && (0.0..=1.0).contains(&d));
    }

    #[test]
    fn crosseval_dims_matrix_is_square_per_dimension() {
        // 6 queries so the 80/20 holdout split is never empty.
        let queries = [
            "hi there",
            "summarize this paragraph",
            "prove the theorem step by step",
            "design a database schema",
            "what is two plus two",
            "analyze the trade-offs and justify",
        ];
        let a: Vec<_> = queries
            .iter()
            .enumerate()
            .map(|(i, q)| dims_ex(q, [(i % 5) as u8, 1, 2, 0, 1, 2]))
            .collect();
        let b: Vec<_> = queries
            .iter()
            .enumerate()
            .map(|(i, q)| dims_ex(q, [(i % 4) as u8, 2, 1, 1, 0, 3]))
            .collect();
        let mats = crosseval_dims_matrices(&[a, b]);
        assert_eq!(mats.len(), 6, "one matrix per dimension");
        for m in &mats {
            assert_eq!(m.len(), 2);
            assert_eq!(m[0].len(), 2);
        }
    }
```

Also add this test helper to the `tests` module:

```rust
    fn dims_ex(q: &str, d: [u8; 6]) -> crate::dataset::DimsExample {
        crate::dataset::DimsExample {
            query: q.into(),
            category: "x".into(),
            dims: crate::dataset::DimScores {
                reasoning_depth: d[0],
                verification_difficulty: d[1],
                constraint_density: d[2],
                context_integration: d[3],
                ambiguity: d[4],
                error_cost: d[5],
            },
        }
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p route-llm-trainer --release crosseval_dims_matrix_is_square_per_dimension`
Expected: compile error — `crosseval_dims_matrices` undefined.

- [ ] **Step 3: Implement**

Append to `crates/trainer/src/eval.rs` (reuses existing `spearman`, `ordinal_accuracy`, `crosseval_matrix`, and `dataset::{dim_value, DimsExample, LabeledExample}`):

```rust
use crate::dataset::{dim_value, DimsExample};

const DIM_NAMES: [&str; 6] = [
    "reasoning_depth",
    "verification_difficulty",
    "constraint_density",
    "context_integration",
    "ambiguity",
    "error_cost",
];
const DIM_SCALES: [f64; 6] = [4.0, 4.0, 4.0, 4.0, 3.0, 4.0];

/// Map a 6-dim labeled set to per-dimension `LabeledExample`s (target = dim/scale).
fn dim_as_labeled(set: &[DimsExample], i: usize) -> Vec<LabeledExample> {
    set.iter()
        .map(|d| LabeledExample {
            query: d.query.clone(),
            difficulty: (dim_value(&d.dims, i) / DIM_SCALES[i]).clamp(0.0, 1.0),
            category: d.category.clone(),
        })
        .collect()
}

/// Per-dimension cross-labeler matrices (fit on row, holdout-eval on col).
pub fn crosseval_dims_matrices(sets: &[Vec<DimsExample>]) -> Vec<Vec<Vec<f64>>> {
    (0..6)
        .map(|i| {
            let per_labeler: Vec<Vec<LabeledExample>> =
                sets.iter().map(|s| dim_as_labeled(s, i)).collect();
            crosseval_matrix(&per_labeler)
        })
        .collect()
}

/// `eval-budget`: score the shipped BudgetRouter (raw estimator difficulty) and the
/// learned model against the human gold's difficulty. SPEC-v3 §8 axis A.
pub fn run_eval_budget(gold_path: &str) {
    let gold = match crate::dataset::load(gold_path) {
        Ok(g) if !g.is_empty() => g,
        Ok(_) => {
            eprintln!("eval-budget: gold set {gold_path} is empty");
            std::process::exit(2);
        }
        Err(e) => {
            eprintln!("eval-budget: {e}");
            std::process::exit(2);
        }
    };
    let labels: Vec<f64> = gold.iter().map(|g| g.difficulty).collect();

    let budget = route_llm_core::BudgetRouter::new();
    let bpred: Vec<f64> = gold.iter().map(|g| budget.raw_difficulty(&g.query)).collect();

    let learned = route_llm_core::learned::weights::shipped_model();
    let lpred: Vec<f64> = gold.iter().map(|g| learned.difficulty(&g.query).score).collect();

    println!("eval-budget vs human gold ({} queries)", gold.len());
    println!("{:<10} {:>10} {:>10}", "router", "spearman", "ordinal");
    println!(
        "{:<10} {:>10.3} {:>10.3}",
        "budget",
        spearman(&bpred, &labels),
        ordinal_accuracy(&bpred, &labels)
    );
    println!(
        "{:<10} {:>10.3} {:>10.3}",
        "learned",
        spearman(&lpred, &labels),
        ordinal_accuracy(&lpred, &labels)
    );
    println!(
        "\nAxis A (SPEC-v3 §8): adopt budget as the difficulty backbone iff budget \
         Spearman >= learned AND budget ordinal >= learned."
    );
}

/// Print per-dimension cross-labeler matrices (diagnostic; SPEC-v3 §8 axis B).
pub fn run_crosseval_dims(paths: &[String]) {
    let sets: Vec<Vec<DimsExample>> = paths
        .iter()
        .filter_map(|p| match crate::dataset::load_dims(p) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("crosseval --dims: skip {p}: {e}");
                None
            }
        })
        .collect();
    if sets.len() < 2 {
        eprintln!("crosseval --dims needs >= 2 readable budget.*.jsonl files");
        std::process::exit(2);
    }
    let names: Vec<String> = paths.iter().map(|p| short_name(p)).collect();
    let mats = crosseval_dims_matrices(&sets);
    for (i, m) in mats.iter().enumerate() {
        println!("\n# dimension: {}", DIM_NAMES[i]);
        print!("{:<14}", "fit\\eval");
        for n in &names {
            print!("{n:>12}");
        }
        println!();
        for (r, row) in m.iter().enumerate() {
            print!("{:<14}", names.get(r).map(|s| s.as_str()).unwrap_or("?"));
            for v in row {
                print!("{v:>12.3}");
            }
            println!();
        }
    }
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p route-llm-trainer --release`
Expected: all trainer tests pass, including the two new eval tests.

- [ ] **Step 5: Commit**

```bash
git add crates/trainer/src/eval.rs
git commit -m "$(cat <<'EOF'
feat(trainer): eval-budget (vs human gold) + per-dimension crosseval

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: `main.rs` — dispatch `label --dims` / `fit-budget` / `eval-budget` / `crosseval --dims`

**Files:**
- Modify: `crates/trainer/src/main.rs`

- [ ] **Step 1: Wire the subcommands**

In `crates/trainer/src/main.rs`, replace the `"label"`, `"crosseval"`, and `other` arms, and add `"fit-budget"` / `"eval-budget"`:

```rust
        "label" => {
            let rest: Vec<String> = std::env::args().skip(2).collect();
            if rest.iter().any(|a| a == "--dims") {
                budget_label::run_dims();
            } else {
                label::run();
            }
        }
        "fit-budget" => {
            let data = dataset::load_dims("data/budget.jsonl").expect("load data/budget.jsonl");
            let models = budget_label::fit_dims(&data);
            emit::write_budget(&models, "crates/core/src/budget/weights.rs")
                .expect("write budget/weights.rs");
            eprintln!(
                "fit-budget: {} examples -> crates/core/src/budget/weights.rs (6 heads)",
                data.len()
            );
        }
        "eval-budget" => {
            let rest: Vec<String> = std::env::args().skip(2).collect();
            let gold = eval::parse_flag(&rest, "--gold").unwrap_or_else(|| "data/gold.jsonl".to_string());
            eval::run_eval_budget(&gold);
        }
        "crosseval" => {
            let rest: Vec<String> = std::env::args().skip(2).collect();
            if rest.iter().any(|a| a == "--dims") {
                let files: Vec<String> = rest.into_iter().filter(|a| a != "--dims").collect();
                eval::run_crosseval_dims(&files);
            } else {
                eval::crosseval(&rest);
            }
        }
```

Update the `other =>` usage string to advertise the new subcommands:

```rust
        other => {
            eprintln!("usage: trainer <synth|label [--dims]|fit|fit-budget|eval [--in <file>|--gold <file>]|eval-budget [--gold <file>]|compare [--gold <file>] <files...>|crosseval [--dims] [files...]|gold-pool>");
            if !other.is_empty() {
                eprintln!("unknown subcommand: {other:?}");
            }
            std::process::exit(2);
        }
```

- [ ] **Step 2: Build + run the whole suite**

Run: `cargo build -p route-llm-trainer --release`
Expected: warning-clean build.

Run: `cargo test --release`
Expected: **entire workspace green** (core + server + trainer).

- [ ] **Step 3: Smoke-test the CLI dispatch (no network, no labels yet)**

Run: `cargo run -p route-llm-trainer --release -- eval-budget --gold data/gold.jsonl`
Expected: prints the `budget` vs `learned` table against the existing human gold (budget row uses the placeholder heads — numbers are not the verdict yet; that comes in Task 16). Confirms the plumbing runs end-to-end.

- [ ] **Step 4: Commit**

```bash
git add crates/trainer/src/main.rs
git commit -m "$(cat <<'EOF'
feat(trainer): dispatch label --dims, fit-budget, eval-budget, crosseval --dims

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: `prompts/label.budget.prompt.md` — portable rubric

**Files:**
- Create: `prompts/label.budget.prompt.md`

- [ ] **Step 1: Write the rubric**

Create `prompts/label.budget.prompt.md` (portable copy of the in-code prompt, for external-tool labeling — mirrors `prompts/label.prompt.md`):

````markdown
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
````

- [ ] **Step 2: Commit**

```bash
git add prompts/label.budget.prompt.md
git commit -m "$(cat <<'EOF'
docs(prompts): portable 6-dim budget labeling rubric

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Full-workspace verification gate

**Files:** none (verification only)

- [ ] **Step 1: Format, build, test**

Run: `cargo fmt --all && cargo build --release && cargo test --release`
Expected: no diff from `fmt`; warning-clean build; **all tests green** across core/server/trainer.

- [ ] **Step 2: Lint the docs touched so far**

Run: `lineguard SPEC-v3.md PLAN-v3.md README.md prompts/label.budget.prompt.md`
Expected: all pass.

- [ ] **Step 3: Commit any fmt-only changes (if any)**

```bash
git status --short
# only if fmt changed files:
git add -A && git commit -m "$(cat <<'EOF'
style: rustfmt v3 budget modules

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase D — manual labeling + acceptance

## Task 15 (MANUAL): generate 6-dim labels with frontier LLMs

**Files:**
- Create: `data/budget.claude.jsonl`, `data/budget.codex.jsonl`, `data/budget.gemma.jsonl`

This is an **offline human-run step** (like v2.2's hand-labeling). It uses the network (frontier LLMs), never the inference path.

- [ ] **Step 1: Label with each frontier model**

For an OpenAI-compatible local/served endpoint, run once per labeler (the model id becomes the filename fragment):

```bash
ROUTE_LLM_LABEL_MODEL=claude  cargo run -p route-llm-trainer --release -- label --dims
ROUTE_LLM_LABEL_MODEL=codex   cargo run -p route-llm-trainer --release -- label --dims
ROUTE_LLM_LABEL_MODEL=gemma   cargo run -p route-llm-trainer --release -- label --dims
```

Or use `prompts/label.budget.prompt.md` with external tooling and hand-place the
`data/budget.<labeler>.jsonl` files. Each must align line-for-line with `data/corpus.jsonl`.

- [ ] **Step 2: Sanity-check the label files**

Run: `cargo run -p route-llm-trainer --release -- crosseval --dims data/budget.claude.jsonl data/budget.codex.jsonl data/budget.gemma.jsonl`
Expected: six per-dimension matrices print; diagonals high, off-diagonals show cross-labeler transfer (diagnostic for which axes are robust vs noisy).

- [ ] **Step 3: Commit the label data**

```bash
git add data/budget.claude.jsonl data/budget.codex.jsonl data/budget.gemma.jsonl
git commit -m "$(cat <<'EOF'
data(v3): 6-dim budget labels from three frontier labelers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: fit, gold-gated verdict, ship

**Files:**
- Modify: `crates/core/src/budget/weights.rs` (generated by `fit-budget`)
- Create: `data/budget.jsonl` (chosen-labeler copy)
- Modify: `SPEC-v3.md` (§16 verdict)
- Possibly modify: `crates/core/src/budget/level.rs` (calibrated thresholds), `crates/server/src/main.rs` (default router) — **only with human approval** (SPEC-v3 §8)

- [ ] **Step 1: Choose the shipping labeler and fit**

Per SPEC-v3 §8 axis B, pick the labeler whose budget router scores best against the
human gold. Start with the v2.2 incumbent (codex):

```bash
cp data/budget.codex.jsonl data/budget.jsonl
cargo run -p route-llm-trainer --release -- fit-budget   # regenerates crates/core/src/budget/weights.rs
cargo fmt --all
```

- [ ] **Step 2: Run the gold-gated evaluation (axis A)**

```bash
cargo run -p route-llm-trainer --release -- eval-budget --gold data/gold.jsonl
```

Record `budget` vs `learned` Spearman/ordinal. **Axis A decision (SPEC-v3 §8):** adopt
budget as the difficulty backbone **iff** `budget Spearman ≥ learned` AND `budget ordinal ≥ learned`
(learned was 0.932 / 0.874 in v2.2 §16). Otherwise keep learned as backbone — the
`budget` block still ships (decision/explainability layer); no router change is forced.

- [ ] **Step 3: Compare labelers (axis B) and calibrate thresholds**

If results are close, re-fit on `data/budget.claude.jsonl` / `data/budget.gemma.jsonl`
(repeat Step 1 with each) and compare `eval-budget`. If the `ordinal` is materially
improved by shifting the §5.1 thresholds, adjust `THRESHOLDS` in
`crates/core/src/budget/level.rs` and re-run `eval-budget`. **Threshold/labeler changes
require human approval** before commit.

- [ ] **Step 4: Run the whole suite**

Run: `cargo test --release`
Expected: green. (The fitted `weights.rs` changes the budget router's numbers but not
its structure; core/server budget tests use injected models and stay valid.)

- [ ] **Step 5: Write the §16 verdict**

In `SPEC-v3.md` §16, replace the pending note with the actual table and decisions:
(a) axis A — budget vs learned Spearman/ordinal and whether budget became the backbone;
(b) axis B — shipped labeler + per-dimension `crosseval` notes; (c) calibrated thresholds;
(d) default router (expected to stay `learned`; `budget` selectable via `ROUTE_LLM_ROUTER`).

- [ ] **Step 6: Lint + commit the verdict and shipped weights**

```bash
lineguard SPEC-v3.md
git add crates/core/src/budget/weights.rs data/budget.jsonl SPEC-v3.md
git commit -m "$(cat <<'EOF'
feat(v3): ship fitted budget heads; record gold-gated verdict (SPEC §16)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 7 (optional, human-approved): default-router change**

Only if the verdict and product owner decide `budget` should be the default, change the
fallthrough in `crates/server/src/main.rs` `choose_router` / builder and document it.
Otherwise leave default `learned` (conservative; SPEC-v3 §8).

---

## Self-Review Notes (plan author)

- **Spec coverage:** §3 module list ↔ Tasks 1–13; §4 dims/labeling ↔ Tasks 2, 9, 13; §5 level/difficulty + raw/runtime split ↔ Tasks 3, 6; §6 decision layer/confidence/policy ↔ Tasks 4, 7; §7 output contract ↔ Tasks 1, 6, 7; §8 acceptance/gold gate ↔ Tasks 11, 16; §11 TDD ↔ every task; §12 isolation ↔ Task 1 (`budget: None`, frozen trait/prefs).
- **Determinism / zero-network inference:** all `core` budget code is pure; the only networked code is `budget_label::run_dims` (offline, Task 9/15), reusing `label`'s client.
- **Type consistency:** canonical dimension order `[reasoning_depth, verification_difficulty, constraint_density, context_integration, ambiguity, error_cost]` and scales `[4,4,4,4,3,4]` are identical in `budget::dims` (Task 2), `dataset::dim_value` (Task 8), `budget_label`/`eval` (Tasks 9, 11). `BUDGET_SCHEMA_VERSION = 1` in `dims.rs` and both `weights.rs` variants.
- **Gold gate is honest:** `eval-budget` scores the **raw** estimator difficulty (pre-escalation), matching SPEC-v3 §5.3 / §8 axis A; runtime difficulty (`max(raw, level floor)`) is separate and only affects the ranker.
