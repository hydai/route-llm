# route-llm v2.2 — Trustworthy Verdict Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a label-independent human gold yardstick (the 143 queries where claude and codex disagree) and re-decide both shipping axes — learned-vs-heuristic default and which labeler's `weights.rs` ships — by scoring every router on the *human* labels.

**Architecture:** Add a `gold-pool` builder that extracts the claude≠codex disagreement set as a *blind* query list; a human hand-labels it into `data/gold.jsonl`; new `eval --gold` / `compare --gold` / `crosseval` subcommands score routers against the human gold labels reusing the existing fit + Spearman/ordinal/cost helpers. All offline, zero network, zero new dependencies. Inference core untouched unless the verdict flips something (then a `weights.rs` re-fit and/or a one-line `choose_router` default change).

**Tech Stack:** Rust (cargo workspace); `serde`/`serde_json`; existing `route-llm-core` learned model + ranker + heuristic; existing `crates/trainer` `dataset`/`logreg`/`eval`. No new crates.

**Branch:** `spec/v2.2-trustworthy-verdict` (already created; never commit to `master`). Release builds only.

**Spec:** `SPEC-v2.2.md` (approved).

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `crates/trainer/src/gold.rs` | Build the gold pool: compute `claude≠codex`, emit blind query list | **Create** |
| `crates/trainer/src/eval.rs` | All scoring: add `GoldReport` + `evaluate_gold`, `run_gold`, `compare_gold`, `crosseval(_matrix)`, generic `parse_flag`, `parse_compare_args` | **Modify** |
| `crates/trainer/src/main.rs` | Dispatch `gold-pool` / `crosseval`; route `eval`/`compare` through `--gold`; usage string | **Modify** |
| `prompts/README.md` | Document the gold blind-labeling workflow | **Modify** |
| `data/gold.unlabeled.jsonl` | Generated 143-query blind pool (committed artifact) | **Create (generated)** |
| `data/gold.jsonl` | Human-labeled 143-query gold set | **Create (manual)** |
| `SPEC-v2.2.md` | Fill §16 verdict table + decisions after the run | **Modify (post-run)** |

Tasks 1–7 are pure engineering (do now). Task 8 is a **manual human step** (hand-labeling). Tasks 9–10 are post-labeling (verdict + ship).

---

## Task 1: `gold.rs` — disagreement pool (pure function)

**Files:**
- Create: `crates/trainer/src/gold.rs`
- Modify: `crates/trainer/src/main.rs` (add `mod gold;`)

- [ ] **Step 1: Register the module**

In `crates/trainer/src/main.rs`, add `mod gold;` to the module list (after `mod eval;`):

```rust
mod corpus;
mod dataset;
mod emit;
mod eval;
mod gold;
mod label;
mod logreg;
```

- [ ] **Step 2: Write the failing tests**

Create `crates/trainer/src/gold.rs` with only the test module + a stub. Task 1
imports only what `disagreements` needs (the `dataset` *module* import is added in
Task 2 by `run_pool`), so each commit stays warning-clean:

```rust
use crate::dataset::{CorpusQuery, LabeledExample};
use std::collections::HashMap;

/// Queries where claude and codex assigned a different difficulty — the gold pool.
/// Output is BLIND (`CorpusQuery`: query + category only, no difficulty/rating) so a
/// human can re-judge without anchoring. Ordered by `codex`'s order (= corpus order);
/// `category` is taken from codex. Queries absent from `claude` are skipped.
pub fn disagreements(_claude: &[LabeledExample], _codex: &[LabeledExample]) -> Vec<CorpusQuery> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(q: &str, d: f64, c: &str) -> LabeledExample {
        LabeledExample {
            query: q.into(),
            difficulty: d,
            category: c.into(),
        }
    }

    #[test]
    fn disagreements_selects_only_differing_queries() {
        let claude = vec![ex("a", 0.25, "code"), ex("b", 0.5, "math"), ex("c", 1.0, "math")];
        let codex = vec![ex("a", 0.25, "code"), ex("b", 0.75, "math"), ex("c", 0.75, "math")];
        let pool = disagreements(&claude, &codex);
        let qs: Vec<&str> = pool.iter().map(|p| p.query.as_str()).collect();
        assert_eq!(qs, vec!["b", "c"], "only b and c differ");
    }

    #[test]
    fn disagreements_keeps_category_from_codex() {
        let claude = vec![ex("b", 0.5, "math")];
        let codex = vec![ex("b", 0.75, "math")];
        let pool = disagreements(&claude, &codex);
        assert_eq!(pool.len(), 1);
        assert_eq!(pool[0].category, "math");
        // CorpusQuery has no `difficulty` field — blindness is type-enforced.
    }

    #[test]
    fn disagreements_preserve_codex_order() {
        let claude = vec![ex("x", 0.0, "chat"), ex("y", 0.0, "chat"), ex("z", 0.0, "chat")];
        let codex = vec![ex("z", 0.25, "chat"), ex("y", 0.25, "chat"), ex("x", 0.25, "chat")];
        let pool = disagreements(&claude, &codex);
        let qs: Vec<&str> = pool.iter().map(|p| p.query.as_str()).collect();
        assert_eq!(qs, vec!["z", "y", "x"], "must follow codex order");
    }

    #[test]
    fn disagreements_skips_queries_missing_in_claude() {
        let claude = vec![ex("a", 0.25, "code")];
        let codex = vec![ex("a", 0.25, "code"), ex("orphan", 0.5, "math")];
        let pool = disagreements(&claude, &codex);
        assert!(pool.is_empty(), "a agrees; orphan not in claude → skipped");
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test -p route-llm-trainer --release disagreements`
Expected: compile OK, tests **panic** with `not yet implemented` (the `todo!()`).

- [ ] **Step 4: Implement `disagreements`**

Replace the `todo!()` body:

```rust
pub fn disagreements(claude: &[LabeledExample], codex: &[LabeledExample]) -> Vec<CorpusQuery> {
    let claude_by_q: HashMap<&str, f64> =
        claude.iter().map(|e| (e.query.as_str(), e.difficulty)).collect();
    codex
        .iter()
        .filter(|e| match claude_by_q.get(e.query.as_str()) {
            Some(&cd) => (cd - e.difficulty).abs() > 1e-9,
            None => false, // not in claude → can't compare → skip
        })
        .map(|e| CorpusQuery {
            query: e.query.clone(),
            category: e.category.clone(),
        })
        .collect()
}
```

Note: `disagreements` uses `HashMap`, `CorpusQuery`, and `LabeledExample` — all imported. The `dataset` module import is added in Task 2 (`run_pool`), keeping this commit warning-clean.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p route-llm-trainer --release disagreements`
Expected: 4 tests **pass**.

- [ ] **Step 6: Commit**

```bash
git add crates/trainer/src/gold.rs crates/trainer/src/main.rs
git commit -m "$(cat <<'EOF'
feat(trainer): add gold-pool disagreement extractor (claude≠codex)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `gold-pool` subcommand + generate the blind pool

**Files:**
- Modify: `crates/trainer/src/gold.rs` (add `run_pool`)
- Modify: `crates/trainer/src/main.rs` (dispatch + usage)
- Create (generated): `data/gold.unlabeled.jsonl`

- [ ] **Step 1: Add `run_pool` to `gold.rs`**

First, update the import line at the top of `gold.rs` to bring the `dataset` module into scope (`run_pool` calls `dataset::load`/`dataset::save_corpus`):

```rust
use crate::dataset::{self, CorpusQuery, LabeledExample};
```

Then append to `crates/trainer/src/gold.rs` (before the `#[cfg(test)]` module):

```rust
/// `gold-pool`: read the two strong-labeler sets, compute the claude≠codex
/// disagreement set, and write it BLIND to `data/gold.unlabeled.jsonl` for a
/// human to hand-label. Prints a total + per-category summary.
pub fn run_pool() {
    let claude = dataset::load("data/labeled.claude.jsonl")
        .unwrap_or_else(|e| panic!("load data/labeled.claude.jsonl: {e}"));
    let codex = dataset::load("data/labeled.codex.jsonl")
        .unwrap_or_else(|e| panic!("load data/labeled.codex.jsonl: {e}"));
    let pool = disagreements(&claude, &codex);
    dataset::save_corpus("data/gold.unlabeled.jsonl", &pool)
        .expect("write data/gold.unlabeled.jsonl");

    let mut by_cat: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for q in &pool {
        *by_cat.entry(q.category.as_str()).or_default() += 1;
    }
    eprintln!(
        "gold-pool: {} disagreements (claude≠codex) -> data/gold.unlabeled.jsonl",
        pool.len()
    );
    for (cat, n) in &by_cat {
        eprintln!("  {cat}: {n}");
    }
}
```

- [ ] **Step 2: Wire dispatch in `main.rs`**

In `crates/trainer/src/main.rs`, add a `gold-pool` arm (after the `"compare"` arm) and update the usage string:

```rust
        "gold-pool" => gold::run_pool(),
```

Update the usage line in the `other =>` arm to:

```rust
            eprintln!("usage: trainer <synth|label|fit|eval [--in <file>|--gold <file>]|compare [--gold <file>] <files...>|crosseval [files...]|gold-pool>");
```

- [ ] **Step 3: Build and run gold-pool**

Run:
```bash
cargo build --release -p route-llm-trainer
cargo run --release -p route-llm-trainer -- gold-pool
```
Expected stderr:
```
gold-pool: 143 disagreements (claude≠codex) -> data/gold.unlabeled.jsonl
  code: 15
  math: 80
  multilingual: 32
  reasoning: 16
```

- [ ] **Step 4: Sanity-check the artifact**

Run:
```bash
wc -l data/gold.unlabeled.jsonl                       # 143
head -1 data/gold.unlabeled.jsonl                     # {"query":...,"category":...} — NO difficulty
jq -c 'has("difficulty")' data/gold.unlabeled.jsonl | sort -u   # only "false"
```
Expected: 143 lines; no `difficulty` key on any line (blind).

- [ ] **Step 5: Commit (code + generated artifact)**

```bash
git add crates/trainer/src/gold.rs crates/trainer/src/main.rs data/gold.unlabeled.jsonl
git commit -m "$(cat <<'EOF'
feat(trainer): add `gold-pool` subcommand; generate 143-query blind pool

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `evaluate_gold` + `GoldReport` (scoring vs human gold)

**Files:**
- Modify: `crates/trainer/src/eval.rs` (add struct + function + test)

- [ ] **Step 1: Write the failing test**

In `crates/trainer/src/eval.rs`, inside the existing `#[cfg(test)] mod tests { ... }` block (it already has `sample_data()`), add:

```rust
    #[test]
    fn evaluate_gold_produces_in_range_metrics() {
        let train = sample_data();
        let gold = vec![
            LabeledExample { query: "hi".into(), difficulty: 0.0, category: "chat".into() },
            LabeledExample { query: "prove a tight lower bound".into(), difficulty: 1.0, category: "math".into() },
            LabeledExample { query: "implement a binary search in Rust".into(), difficulty: 0.5, category: "code".into() },
            LabeledExample { query: "define HTTP".into(), difficulty: 0.25, category: "extraction".into() },
        ];
        let r = evaluate_gold(&train, &gold);
        assert_eq!(r.n, 4);
        for s in [r.spearman_learned, r.spearman_heuristic] {
            assert!((-1.0..=1.0).contains(&s), "spearman out of range: {s}");
        }
        for o in [r.ordinal_learned, r.ordinal_heuristic] {
            assert!((0.0..=1.0).contains(&o), "ordinal out of range: {o}");
        }
        for c in [r.cost_learned, r.cost_heuristic] {
            assert!(c.is_finite() && c >= 0.0, "cost invalid: {c}");
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p route-llm-trainer --release evaluate_gold_produces`
Expected: **compile error** — `GoldReport` / `evaluate_gold` not found.

- [ ] **Step 3: Implement `GoldReport` + `evaluate_gold`**

In `crates/trainer/src/eval.rs`, after the `EvalReport` struct + `evaluate` function (before `print_report`), add:

```rust
/// Result of scoring routers against an EXTERNAL gold set (human labels = truth).
/// Holdout-free: the learned model is fit on ALL of `train`, then predicts the
/// gold queries. The `LinearModel` is too low-capacity to memorize individual
/// queries, so train/gold query overlap does not bias the comparison — and this
/// reflects the actually-shipped router's behavior. Cost is informational
/// (avg top-1 pick cost); the gold verdict's primary axes are spearman + ordinal.
#[derive(Debug, Clone)]
pub struct GoldReport {
    pub n: usize,
    pub spearman_learned: f64,
    pub spearman_heuristic: f64,
    pub ordinal_learned: f64,
    pub ordinal_heuristic: f64,
    pub cost_learned: f64,
    pub cost_heuristic: f64,
}

/// Fit learned on ALL of `train`; score learned + heuristic on `gold` vs gold's
/// human `difficulty`. Pure (no I/O).
pub fn evaluate_gold(train: &[LabeledExample], gold: &[LabeledExample]) -> GoldReport {
    let model = logreg::fit(train, &FitConfig::default());
    let labels: Vec<f64> = gold.iter().map(|e| e.difficulty).collect();
    let learned: Vec<f64> = gold
        .iter()
        .map(|e| model.difficulty(&e.query).score)
        .collect();
    let heuristic: Vec<f64> = gold
        .iter()
        .map(|e| difficulty::score(&e.query).score)
        .collect();
    let (lc, _) = cost_profile(&learned, &labels);
    let (hc, _) = cost_profile(&heuristic, &labels);
    GoldReport {
        n: gold.len(),
        spearman_learned: spearman(&learned, &labels),
        spearman_heuristic: spearman(&heuristic, &labels),
        ordinal_learned: ordinal_accuracy(&learned, &labels),
        ordinal_heuristic: ordinal_accuracy(&heuristic, &labels),
        cost_learned: lc,
        cost_heuristic: hc,
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p route-llm-trainer --release evaluate_gold_produces`
Expected: **PASS**.

- [ ] **Step 5: Commit**

```bash
git add crates/trainer/src/eval.rs
git commit -m "$(cat <<'EOF'
feat(trainer): add evaluate_gold — score routers vs external human gold

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: generic `parse_flag` + `run_gold` + `eval --gold` dispatch

**Files:**
- Modify: `crates/trainer/src/eval.rs` (add `parse_flag`, rewrite `parse_in_flag` to delegate, add `run_gold`, test)
- Modify: `crates/trainer/src/main.rs` (`eval` arm)

- [ ] **Step 1: Write the failing test**

In `eval.rs` tests module, add:

```rust
    #[test]
    fn parse_flag_finds_named_value() {
        let args = vec![
            "compare".to_string(),
            "--gold".to_string(),
            "g.jsonl".to_string(),
            "a.jsonl".to_string(),
        ];
        assert_eq!(parse_flag(&args, "--gold"), Some("g.jsonl".to_string()));
        assert_eq!(parse_flag(&args, "--in"), None);
        // flag present but no following value
        assert_eq!(parse_flag(&["--gold".to_string()], "--gold"), None);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p route-llm-trainer --release parse_flag_finds_named_value`
Expected: **compile error** — `parse_flag` not found.

- [ ] **Step 3: Add `parse_flag` and delegate `parse_in_flag`**

In `eval.rs`, replace the existing `parse_in_flag` with:

```rust
/// Parse an optional `--<name> <value>` flag from CLI args.
pub fn parse_flag(args: &[String], name: &str) -> Option<String> {
    let pos = args.iter().position(|a| a == name)?;
    args.get(pos + 1).cloned()
}

/// Parse an optional `--in <path>` flag (back-compat alias for `parse_flag`).
pub fn parse_in_flag(args: &[String]) -> Option<String> {
    parse_flag(args, "--in")
}
```

(The existing `parse_in_flag_finds_path` test still passes.)

- [ ] **Step 4: Add `run_gold`**

In `eval.rs`, after `run_path`, add:

```rust
/// `eval --gold <gold.jsonl>`: score the shipped router (fit on `data/labeled.jsonl`)
/// + heuristic against the human gold set. Deployment-faithful check on hard cases.
pub fn run_gold(gold_path: &str) {
    let gold = crate::dataset::load(gold_path).unwrap_or_else(|e| panic!("load {gold_path}: {e}"));
    let train = crate::dataset::load("data/labeled.jsonl")
        .unwrap_or_else(|e| panic!("load data/labeled.jsonl: {e}"));
    let r = evaluate_gold(&train, &gold);
    println!("gold eval — shipped router vs human gold ({gold_path}), n={}", r.n);
    println!(
        "  spearman  learned={:.3}  heuristic={:.3}",
        r.spearman_learned, r.spearman_heuristic
    );
    println!(
        "  ordinal   learned={:.3}  heuristic={:.3}",
        r.ordinal_learned, r.ordinal_heuristic
    );
    println!(
        "  avg cost  learned={:.3}  heuristic={:.3}  (informational)",
        r.cost_learned, r.cost_heuristic
    );
}
```

- [ ] **Step 5: Wire the `eval` arm in `main.rs`**

Replace the existing `"eval" => { ... }` arm with:

```rust
        "eval" => {
            let rest: Vec<String> = std::env::args().skip(2).collect();
            if let Some(gold) = eval::parse_flag(&rest, "--gold") {
                eval::run_gold(&gold);
            } else if let Some(path) = eval::parse_flag(&rest, "--in") {
                eval::run_path(&path);
            } else {
                eval::run();
            }
        }
```

- [ ] **Step 6: Run tests + smoke the wiring**

Run: `cargo test -p route-llm-trainer --release parse_flag`
Expected: `parse_flag_finds_named_value` and `parse_in_flag_finds_path` both **pass**.

Build check: `cargo build --release -p route-llm-trainer`
Expected: builds clean. (Running `eval --gold` is deferred to Task 9, after the human gold set exists.)

- [ ] **Step 7: Commit**

```bash
git add crates/trainer/src/eval.rs crates/trainer/src/main.rs
git commit -m "$(cat <<'EOF'
feat(trainer): add `eval --gold` (shipped router vs human gold)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `compare --gold` (cross-labeler table on one human yardstick)

**Files:**
- Modify: `crates/trainer/src/eval.rs` (add `parse_compare_args`, `compare_gold`, test)
- Modify: `crates/trainer/src/main.rs` (`compare` arm)

- [ ] **Step 1: Write the failing test**

In `eval.rs` tests module, add:

```rust
    #[test]
    fn parse_compare_args_splits_gold_and_files() {
        let rest = vec![
            "--gold".to_string(),
            "g.jsonl".to_string(),
            "a.jsonl".to_string(),
            "b.jsonl".to_string(),
        ];
        let (gold, files) = parse_compare_args(&rest);
        assert_eq!(gold, Some("g.jsonl".to_string()));
        assert_eq!(files, vec!["a.jsonl".to_string(), "b.jsonl".to_string()]);

        let (g2, f2) = parse_compare_args(&["a.jsonl".to_string()]);
        assert_eq!(g2, None);
        assert_eq!(f2, vec!["a.jsonl".to_string()]);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p route-llm-trainer --release parse_compare_args_splits`
Expected: **compile error** — `parse_compare_args` not found.

- [ ] **Step 3: Implement `parse_compare_args` + `compare_gold`**

In `eval.rs`, add `parse_compare_args` (near `parse_flag`):

```rust
/// Split `compare` args (everything after the subcommand) into an optional
/// `--gold <path>` and the positional labeled-file list.
pub fn parse_compare_args(args: &[String]) -> (Option<String>, Vec<String>) {
    let mut gold = None;
    let mut files = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--gold" {
            gold = args.get(i + 1).cloned();
            i += 2;
        } else {
            files.push(args[i].clone());
            i += 1;
        }
    }
    (gold, files)
}
```

Then add `compare_gold` (after the existing `compare` function):

```rust
/// `compare --gold <gold.jsonl> <labeled...>`: fit a learned router on each
/// labeled set (full), score it on the SAME human gold labels, and print one
/// table. Unlike `compare`, every row is measured against the same external
/// human yardstick → label-independent and cross-labeler comparable.
pub fn compare_gold(gold_path: &str, labeled_paths: &[String]) {
    if labeled_paths.is_empty() {
        eprintln!("usage: trainer compare --gold <gold.jsonl> <labeled1.jsonl> [...]");
        std::process::exit(2);
    }
    let gold = crate::dataset::load(gold_path).unwrap_or_else(|e| panic!("load {gold_path}: {e}"));
    let reports: Vec<(String, GoldReport)> = labeled_paths
        .iter()
        .map(|p| {
            let train = crate::dataset::load(p).unwrap_or_else(|e| panic!("load {p}: {e}"));
            (short_name(p), evaluate_gold(&train, &gold))
        })
        .collect();

    println!("gold comparison — routers vs human gold ({gold_path}), n={}", gold.len());
    println!(
        "{:<14} {:>5} {:>9} {:>9} {:>10}",
        "router", "n", "sp_gold", "ord_gold", "avg_cost"
    );
    // Heuristic is train-independent: print it once (from the first report).
    let h = &reports[0].1;
    println!(
        "{:<14} {:>5} {:>9.3} {:>9.3} {:>10.3}",
        "heuristic", h.n, h.spearman_heuristic, h.ordinal_heuristic, h.cost_heuristic
    );
    for (name, r) in &reports {
        println!(
            "{:<14} {:>5} {:>9.3} {:>9.3} {:>10.3}",
            name, r.n, r.spearman_learned, r.ordinal_learned, r.cost_learned
        );
    }
    println!("note: ALL rows scored vs the SAME human gold labels — label-independent, cross-labeler comparable.");
}
```

- [ ] **Step 4: Wire the `compare` arm in `main.rs`**

Replace the existing `"compare" => { ... }` arm with:

```rust
        "compare" => {
            let rest: Vec<String> = std::env::args().skip(2).collect();
            let (gold, files) = eval::parse_compare_args(&rest);
            match gold {
                Some(g) => eval::compare_gold(&g, &files),
                None => eval::compare(&files),
            }
        }
```

- [ ] **Step 5: Run tests + build**

Run: `cargo test -p route-llm-trainer --release parse_compare_args_splits`
Expected: **PASS**.
Run: `cargo build --release -p route-llm-trainer`
Expected: clean build. (Real run deferred to Task 9.)

- [ ] **Step 6: Commit**

```bash
git add crates/trainer/src/eval.rs crates/trainer/src/main.rs
git commit -m "$(cat <<'EOF'
feat(trainer): add `compare --gold` cross-labeler table on one human yardstick

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `crosseval` (label-transfer matrix)

**Files:**
- Modify: `crates/trainer/src/eval.rs` (add `crosseval_matrix`, `crosseval`, test)
- Modify: `crates/trainer/src/main.rs` (`crosseval` arm)

- [ ] **Step 1: Write the failing test**

In `eval.rs` tests module, add:

```rust
    #[test]
    fn crosseval_matrix_is_square_with_finite_diagonal() {
        let sets = vec![sample_data(), sample_data()];
        let m = crosseval_matrix(&sets);
        assert_eq!(m.len(), 2, "one row per train set");
        assert!(m.iter().all(|row| row.len() == 2), "one col per test set");
        for i in 0..2 {
            assert!(m[i][i].is_finite(), "diagonal must be finite");
            assert!((-1.0..=1.0).contains(&m[i][i]), "spearman in range");
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p route-llm-trainer --release crosseval_matrix_is_square`
Expected: **compile error** — `crosseval_matrix` not found.

- [ ] **Step 3: Implement `crosseval_matrix` + `crosseval`**

In `eval.rs`, add:

```rust
/// Spearman matrix for cross-labeler transfer: cell [i][j] = fit a learned model
/// on `sets[i]` (full), predict `sets[j]`'s queries, spearman vs `sets[j]`'s
/// labels. Diagonal is in-sample (optimistic); off-diagonal = transfer.
pub fn crosseval_matrix(sets: &[Vec<LabeledExample>]) -> Vec<Vec<f64>> {
    sets.iter()
        .map(|train| {
            let model = logreg::fit(train, &FitConfig::default());
            sets.iter()
                .map(|test| {
                    let labels: Vec<f64> = test.iter().map(|e| e.difficulty).collect();
                    let pred: Vec<f64> =
                        test.iter().map(|e| model.difficulty(&e.query).score).collect();
                    spearman(&pred, &labels)
                })
                .collect()
        })
        .collect()
}

/// `crosseval [files...]`: print the cross-labeler spearman matrix. Defaults to
/// the three committed labeler sets when no files are given.
pub fn crosseval(paths: &[String]) {
    let owned: Vec<String> = if paths.is_empty() {
        [
            "data/labeled.gemma.jsonl",
            "data/labeled.claude.jsonl",
            "data/labeled.codex.jsonl",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    } else {
        paths.to_vec()
    };
    let sets: Vec<Vec<LabeledExample>> = owned
        .iter()
        .map(|p| crate::dataset::load(p).unwrap_or_else(|e| panic!("load {p}: {e}")))
        .collect();
    let names: Vec<String> = owned.iter().map(|p| short_name(p)).collect();
    let m = crosseval_matrix(&sets);

    println!("crosseval — spearman of (fit on row) predicting (col)'s labels");
    print!("{:<14}", "train\\test");
    for n in &names {
        print!(" {n:>9}");
    }
    println!();
    for (i, row) in m.iter().enumerate() {
        print!("{:<14}", names[i]);
        for v in row {
            print!(" {v:>9.3}");
        }
        println!();
    }
    println!("note: diagonal is in-sample (optimistic); off-diagonal = cross-labeler transfer.");
}
```

- [ ] **Step 4: Wire the `crosseval` arm in `main.rs`**

Add (after the `gold-pool` arm):

```rust
        "crosseval" => {
            let files: Vec<String> = std::env::args().skip(2).collect();
            eval::crosseval(&files);
        }
```

- [ ] **Step 5: Run test + real run (uses committed labeler sets)**

Run: `cargo test -p route-llm-trainer --release crosseval_matrix_is_square`
Expected: **PASS**.

Run: `cargo run --release -p route-llm-trainer -- crosseval`
Expected: a 3×3 table (rows/cols `gemma`/`claude`/`codex`), diagonal near each set's self-fit spearman, all values in [-1, 1]. Record the matrix for the spec write-up in Task 9.

- [ ] **Step 6: Commit**

```bash
git add crates/trainer/src/eval.rs crates/trainer/src/main.rs
git commit -m "$(cat <<'EOF'
feat(trainer): add `crosseval` label-transfer matrix across labelers

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: document the gold workflow in `prompts/README.md`

**Files:**
- Modify: `prompts/README.md`

- [ ] **Step 1: Add a gold section**

Append this section to `prompts/README.md` (after the "fit / eval runbook" section, before "Comparing labelers — a fairness caveat"):

````markdown
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
````

- [ ] **Step 2: Lint**

Run: `lineguard prompts/README.md`
Expected: passes (fix any reported issues).

- [ ] **Step 3: Commit**

```bash
git add prompts/README.md
git commit -m "$(cat <<'EOF'
docs(prompts): document the v2.2 gold blind-labeling + verdict workflow

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8 (MANUAL — human judge): hand-label `data/gold.jsonl`

> This task has **no code**. It is the human-judgment step that makes the gold set
> independent. An agentic worker must STOP here and hand off to the repo owner; do
> NOT label it with a model.

**Files:**
- Create: `data/gold.jsonl`

- [ ] **Step 1: Read the blind pool**

Open `data/gold.unlabeled.jsonl` (143 lines, `{query, category}` only — no model labels shown).

- [ ] **Step 2: Rate each query 1–5, blind**

Apply the `prompts/label.prompt.md` rubric (1 = trivial chat … 5 = expert). Judge the
query only; do not look at any model's label. Map `difficulty = (rating − 1) / 4`.

- [ ] **Step 3: Write `data/gold.jsonl`**

One line per input line, **same order**, `query`/`category` copied byte-for-byte:

```jsonl
{"query":"...","difficulty":0.75,"category":"math","rating":4}
```

- [ ] **Step 4: Validate**

```bash
wc -l data/gold.unlabeled.jsonl data/gold.jsonl                  # counts must match (143)
jq -c 'select((.difficulty|IN(0,0.25,0.5,0.75,1)) | not)' data/gold.jsonl   # prints nothing
cargo run --release -p route-llm-trainer -- eval --gold data/gold.jsonl     # parses + runs
```
Expected: equal line counts; no out-of-rubric difficulties; `eval --gold` prints a report.

- [ ] **Step 5: Commit**

```bash
git add data/gold.jsonl
git commit -m "$(cat <<'EOF'
data(v2.2): add 143-query human gold set (blind-labeled, claude≠codex)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: run the gold verdict, fill SPEC §16, act on the decision

**Files:**
- Modify: `SPEC-v2.2.md` (§16)
- Possibly modify: `data/labeled.jsonl` + `crates/core/src/learned/weights.rs` (axis B) and/or `crates/server/src/main.rs` (axis A) — **only if the verdict requires it; human-approved.**

- [ ] **Step 1: Produce the verdict tables**

```bash
cargo run --release -p route-llm-trainer -- compare --gold data/gold.jsonl \
  data/labeled.codex.jsonl data/labeled.claude.jsonl data/labeled.gemma.jsonl
cargo run --release -p route-llm-trainer -- eval --gold data/gold.jsonl
cargo run --release -p route-llm-trainer -- crosseval
```
Record the `sp_gold` / `ord_gold` / `avg_cost` for `heuristic`, `codex`, `claude`, `gemma`, and the crosseval matrix.

- [ ] **Step 2: Apply the decision rules (SPEC §8)**

- **Axis A — default router:** learned wins iff, on gold, `sp_gold(best learned) ≥ sp_gold(heuristic)` AND `ord_gold(best learned) ≥ ord_gold(heuristic)`.
  - Win → keep `learned` (no `choose_router` change).
  - Lose → flip `crates/server/src/main.rs` `choose_router`: the `Err(VarError::NotPresent)` arm returns `"heuristic"` instead of `"learned"`.
- **Axis B — ship labeler:** the labeler with the best `sp_gold` (tie-break `ord_gold`).
  - Already `codex` → no change.
  - Otherwise → `cp data/labeled.<winner>.jsonl data/labeled.jsonl && cargo run --release -p route-llm-trainer -- fit`, then re-run Step 1's `eval --gold` to confirm.

- [ ] **Step 3: Fill `SPEC-v2.2.md` §16**

Replace the `_待填_` table cells with the measured numbers and write the two-axis decision sentences. Remove the `> 待跑` note.

- [ ] **Step 4: Verify the whole workspace is green**

```bash
cargo build --release
cargo test
```
Expected: builds clean; all tests pass.

- [ ] **Step 5: Lint + commit**

```bash
lineguard SPEC-v2.2.md
git add -A
git commit -m "$(cat <<'EOF'
docs(spec): fill v2.2 §16 verdict on the human gold set; act on decision

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```
(If axis A/B changed shipping, split into separate commits: one for the verdict doc, one for the `weights.rs` re-fit / `choose_router` flip.)

---

## Task 10: tick PLAN checkboxes, open PR

**Files:**
- Modify: `PLAN-v2.2.md` (status banner + checkboxes)

- [ ] **Step 1: Tick completed checkboxes + add status banner**

Mark `- [ ]` → `- [x]` for done tasks; add a status line at the top mirroring `PLAN-v2.1.md`.

- [ ] **Step 2: Commit + push**

```bash
git add PLAN-v2.2.md
git commit -m "$(cat <<'EOF'
docs: tick PLAN-v2.2 checkboxes (v2.2 implemented + verdict shipped)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push
```

- [ ] **Step 3: Open the PR**

```bash
gh pr create --base master --head spec/v2.2-trustworthy-verdict \
  --title "v2.2: trustworthy verdict via independent human gold set" \
  --body "$(cat <<'EOF'
Builds a label-independent human gold yardstick (143 claude≠codex queries),
adds `gold-pool` / `eval --gold` / `compare --gold` / `crosseval`, and re-decides
both shipping axes on the human labels. See SPEC-v2.2.md §16 for the verdict.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

**1. Spec coverage** (each SPEC-v2.2 section → task):
- §2.1 `gold-pool` → Tasks 1–2. §2.2 human gold → Task 8. §2.3 `eval --gold`/`compare --gold` → Tasks 4–5. §2.4 `crosseval` → Task 6. §2.5 two-axis re-ship → Task 9. §2.6 offline/no-deps → no `Cargo.toml` change in any task (verified). ✓
- §4 gold-pool (blind, deterministic, codex order, per-category summary) → Tasks 1–2. ✓
- §5 human blind labeling → Task 8. §6 committed artifacts → Tasks 2 + 8. ✓
- §7 eval/compare/crosseval + cost-as-informational (no `cost@ceil` gating; `GoldReport` has only `cost_*`) → Tasks 3–6. ✓
- §8 decision rules → Task 9 Step 2. §11 TDD tests → Tasks 1,3,4,5,6. §12 additive (existing `compare`/`eval` paths preserved; `parse_in_flag` delegates) → Tasks 4–5. ✓

**2. Placeholder scan:** No "TBD/TODO/handle appropriately" in code steps; every code step shows complete code. The only intentional placeholders are SPEC §16's `_待填_`, which Task 9 fills. ✓

**3. Type/name consistency:** `disagreements`, `run_pool`, `GoldReport{n,spearman_learned,spearman_heuristic,ordinal_learned,ordinal_heuristic,cost_learned,cost_heuristic}`, `evaluate_gold`, `parse_flag`, `parse_compare_args`, `compare_gold`, `crosseval_matrix`, `crosseval`, `run_gold` are used identically across tasks and dispatch. `cost_profile`/`spearman`/`ordinal_accuracy`/`short_name`/`logreg::fit`/`FitConfig`/`difficulty::score`/`model.difficulty(&q).score` match the current `eval.rs`/`logreg.rs`/`dataset.rs`. `dataset::save_corpus` takes `&[CorpusQuery]` — `disagreements` returns exactly that. ✓
