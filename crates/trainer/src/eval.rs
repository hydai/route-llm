use crate::dataset::LabeledExample;
use crate::logreg::{self, FitConfig};
use route_llm_core::{difficulty, ranker, registry, ModelProfile, RoutingPreferences};

/// Assign ranks using average rank for tied values (tie-corrected ranking for Spearman).
fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0; v.len()];
    let n = idx.len();
    let mut i = 0;
    while i < n {
        // Find the run of equal values starting at sorted position i.
        let mut j = i + 1;
        while j < n && v[idx[j]] == v[idx[i]] {
            j += 1;
        }
        // Average rank for all elements in [i, j).
        let avg = (i + j - 1) as f64 / 2.0;
        for k in i..j {
            r[idx[k]] = avg;
        }
        i = j;
    }
    r
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "pearson: slice length mismatch");
    let n = a.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let (x, y) = (a[i] - ma, b[i] - mb);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    if da == 0.0 || db == 0.0 {
        0.0
    } else {
        num / (da.sqrt() * db.sqrt())
    }
}

pub fn spearman(pred: &[f64], label: &[f64]) -> f64 {
    debug_assert_eq!(pred.len(), label.len(), "spearman: slice length mismatch");
    pearson(&ranks(pred), &ranks(label))
}

fn bucket(x: f64) -> u8 {
    if x < 0.4 {
        0
    } else if x < 0.7 {
        1
    } else {
        2
    }
}

pub fn ordinal_accuracy(pred: &[f64], label: &[f64]) -> f64 {
    debug_assert_eq!(
        pred.len(),
        label.len(),
        "ordinal_accuracy: slice length mismatch"
    );
    if pred.is_empty() {
        return 0.0;
    }
    let ok = (0..pred.len())
        .filter(|&i| bucket(pred[i]) == bucket(label[i]))
        .count();
    ok as f64 / pred.len() as f64
}

/// Baseline: always pick the highest-quality builtin model.
/// Returns (average cost, adequacy rate) over the given label set.
/// The "strongest" model is the one with the max `quality` in `registry::builtin()`.
fn always_strongest_baseline(labels: &[f64]) -> (f64, f64) {
    let models: Vec<ModelProfile> = registry::builtin();
    let strongest = models
        .iter()
        .max_by(|a, b| {
            a.quality
                .partial_cmp(&b.quality)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("builtin registry must be non-empty");
    let adequate = labels.iter().filter(|&&l| strongest.quality >= l).count();
    let n = labels.len().max(1) as f64;
    (strongest.cost, adequate as f64 / n)
}

/// Average cost of the top-1 pick over the builtin registry, plus the fraction
/// of picks that are "adequate" (chosen quality >= the query's true difficulty).
fn cost_profile(diffs: &[f64], labels: &[f64]) -> (f64, f64) {
    debug_assert_eq!(
        diffs.len(),
        labels.len(),
        "cost_profile: slice length mismatch"
    );
    let models: Vec<ModelProfile> = registry::builtin();
    let prefs = RoutingPreferences::default();
    let (mut cost_sum, mut adequate) = (0.0, 0.0);
    for (i, &d) in diffs.iter().enumerate() {
        let difficulty = route_llm_core::Difficulty {
            score: d,
            signals: vec![],
        };
        let ranking = ranker::rank(&difficulty, &models, &prefs);
        let top_id = &ranking[0].id;
        let top = models.iter().find(|m| &m.id == top_id).unwrap();
        cost_sum += top.cost;
        if top.quality >= labels[i] {
            adequate += 1.0;
        }
    }
    let n = diffs.len().max(1) as f64;
    (cost_sum / n, adequate / n)
}

/// SPEC §12: minimum average cost achievable while keeping adequacy_rate >= `target`.
///
/// Sweeps `cost_bias` over [0.0, 0.1, …, 1.0]; for each bias, routes every query
/// and measures (adequacy_rate, avg_cost). Returns the minimum avg_cost among grid
/// points that reach `target` adequacy, or `None` if no grid point does.
fn cost_at_adequacy(diffs: &[f64], labels: &[f64], target: f64) -> Option<f64> {
    debug_assert_eq!(
        diffs.len(),
        labels.len(),
        "cost_at_adequacy: slice length mismatch"
    );
    let models: Vec<ModelProfile> = registry::builtin();
    let n = diffs.len();
    if n == 0 {
        return None;
    }

    let mut best_cost: Option<f64> = None;

    // Sweep cost_bias in steps of 0.1 (11 grid points: 0.0 … 1.0).
    let steps = 10usize;
    for step in 0..=steps {
        let cost_bias = step as f64 / steps as f64;
        let prefs = RoutingPreferences { cost_bias };

        let mut cost_sum = 0.0;
        let mut adequate = 0usize;
        for (i, &d) in diffs.iter().enumerate() {
            let difficulty = route_llm_core::Difficulty {
                score: d,
                signals: vec![],
            };
            let ranking = ranker::rank(&difficulty, &models, &prefs);
            let top_id = &ranking[0].id;
            let top = models.iter().find(|m| &m.id == top_id).unwrap();
            cost_sum += top.cost;
            if top.quality >= labels[i] {
                adequate += 1;
            }
        }

        let adequacy_rate = adequate as f64 / n as f64;
        let avg_cost = cost_sum / n as f64;

        if adequacy_rate >= target {
            best_cost = Some(match best_cost {
                None => avg_cost,
                Some(prev) => prev.min(avg_cost),
            });
        }
    }

    best_cost
}

/// Structured holdout-eval result for one labeled dataset. Computed on the
/// 80/20-by-index holdout, so two datasets over the same (ordered) corpus are
/// evaluated on the same holdout queries.
#[derive(Debug, Clone)]
pub struct EvalReport {
    pub n_holdout: usize,
    pub spearman_learned: f64,
    pub spearman_heuristic: f64,
    pub ordinal_learned: f64,
    pub ordinal_heuristic: f64,
    pub cost_learned: f64,
    pub adeq_learned: f64,
    pub cost_heuristic: f64,
    pub adeq_heuristic: f64,
    pub cost_strongest: f64,
    pub adeq_strongest: f64,
    pub target_adequacy: f64,
    pub cost_at_adequacy_learned: Option<f64>,
    pub cost_at_adequacy_heuristic: Option<f64>,
    pub cost_at_adequacy_strongest: f64,
}

/// Fit on the 80% train split and compute every metric on the 20% holdout.
/// Pure (no I/O): the same labeled data always yields the same report.
pub fn evaluate(data: &[LabeledExample]) -> EvalReport {
    let (train, holdout): (Vec<_>, Vec<_>) = data
        .iter()
        .cloned()
        .enumerate()
        .partition(|(i, _)| i % 5 != 0); // 80/20 by index
    let train: Vec<LabeledExample> = train.into_iter().map(|(_, e)| e).collect();
    let holdout: Vec<LabeledExample> = holdout.into_iter().map(|(_, e)| e).collect();

    let model = logreg::fit(&train, &FitConfig::default());
    let labels: Vec<f64> = holdout.iter().map(|e| e.difficulty).collect();
    let learned: Vec<f64> = holdout
        .iter()
        .map(|e| model.difficulty(&e.query).score)
        .collect();
    let heuristic: Vec<f64> = holdout
        .iter()
        .map(|e| difficulty::score(&e.query).score)
        .collect();

    let (lc, la) = cost_profile(&learned, &labels);
    let (hc, ha) = cost_profile(&heuristic, &labels);
    let (sc, sa) = always_strongest_baseline(&labels);

    // SPEC §12: cheapest routing that still matches the achievable adequacy
    // ceiling. The ceiling is the always-strongest model's adequacy (sa): some
    // queries can be labeled harder than ANY model's quality, so a fixed target
    // like 0.90 may sit ABOVE the ceiling and be unreachable (n/a). Targeting sa
    // is always reachable (cost_bias=0 routes everything to the strongest model)
    // and adapts per labeled set.
    let target = sa;
    EvalReport {
        n_holdout: holdout.len(),
        spearman_learned: spearman(&learned, &labels),
        spearman_heuristic: spearman(&heuristic, &labels),
        ordinal_learned: ordinal_accuracy(&learned, &labels),
        ordinal_heuristic: ordinal_accuracy(&heuristic, &labels),
        cost_learned: lc,
        adeq_learned: la,
        cost_heuristic: hc,
        adeq_heuristic: ha,
        cost_strongest: sc,
        adeq_strongest: sa,
        target_adequacy: target,
        cost_at_adequacy_learned: cost_at_adequacy(&learned, &labels, target),
        cost_at_adequacy_heuristic: cost_at_adequacy(&heuristic, &labels, target),
        // always-strongest cost is the ceiling: always the max-quality model's cost.
        cost_at_adequacy_strongest: sc,
    }
}

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

/// Print one report in the original `eval` format, tagged with its source.
fn print_report(source: &str, r: &EvalReport) {
    eprintln!("eval {source} (holdout n={})", r.n_holdout);
    eprintln!(
        "  spearman   learned={:.3}  heuristic={:.3}",
        r.spearman_learned, r.spearman_heuristic
    );
    eprintln!(
        "  ordinal    learned={:.3}  heuristic={:.3}",
        r.ordinal_learned, r.ordinal_heuristic
    );
    eprintln!(
        "  avg cost   learned={:.3} (adeq {:.2})  heuristic={:.3} (adeq {:.2})  always-strongest={:.3} (adeq {:.2})",
        r.cost_learned, r.adeq_learned, r.cost_heuristic, r.adeq_heuristic, r.cost_strongest, r.adeq_strongest
    );
    eprintln!(
        "  cost @ ceiling-adeq({:.2})  learned={}  heuristic={}  always-strongest={:.3}",
        r.target_adequacy,
        r.cost_at_adequacy_learned
            .map_or("n/a (target unreachable)".into(), |c| format!("{c:.3}")),
        r.cost_at_adequacy_heuristic
            .map_or("n/a (target unreachable)".into(), |c| format!("{c:.3}")),
        r.cost_at_adequacy_strongest,
    );
}

/// `eval`: report metrics for the default labeled set (`data/labeled.jsonl`).
pub fn run() {
    run_path("data/labeled.jsonl");
}

/// `eval --in <path>`: report metrics for a specific labeled set.
pub fn run_path(path: &str) {
    let data = crate::dataset::load(path).unwrap_or_else(|e| panic!("load {path}: {e}"));
    print_report(path, &evaluate(&data));
}

/// `eval --gold <gold.jsonl>`: score the shipped router (fit on `data/labeled.jsonl`)
/// + heuristic against the human gold set. Deployment-faithful check on hard cases.
pub fn run_gold(gold_path: &str) {
    let gold = crate::dataset::load(gold_path).unwrap_or_else(|e| panic!("load {gold_path}: {e}"));
    let train = crate::dataset::load("data/labeled.jsonl")
        .unwrap_or_else(|e| panic!("load data/labeled.jsonl: {e}"));
    let r = evaluate_gold(&train, &gold);
    println!(
        "gold eval — shipped router vs human gold ({gold_path}), n={}",
        r.n
    );
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

/// Parse an optional `--<name> <value>` flag from CLI args.
pub fn parse_flag(args: &[String], name: &str) -> Option<String> {
    let pos = args.iter().position(|a| a == name)?;
    args.get(pos + 1).cloned()
}

/// Parse an optional `--in <path>` flag (back-compat alias for `parse_flag`).
pub fn parse_in_flag(args: &[String]) -> Option<String> {
    parse_flag(args, "--in")
}

/// Shorten a labeled-set path to a table label: `data/labeled.claude.jsonl` → `claude`.
fn short_name(path: &str) -> String {
    let base = std::path::Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path);
    let stem = base.strip_suffix(".jsonl").unwrap_or(base);
    stem.strip_prefix("labeled.")
        .filter(|s| !s.is_empty())
        .unwrap_or(stem)
        .to_string()
}

/// `compare <files...>`: eval each labeled set and print a side-by-side table of
/// the *learned* router's metrics (plus heuristic spearman for reference).
///
/// All sets should cover the same corpus so the holdout queries match. NOTE:
/// each set's spearman/ordinal/adequacy are measured against *that set's own
/// labels*, so they show how learnable each labeler's signal is — not which
/// labeler is objectively "correct". Compare `avg_cost` (same holdout queries)
/// for a label-independent view of how cheaply each router routes.
pub fn compare(paths: &[String]) {
    if paths.is_empty() {
        eprintln!("usage: trainer compare <labeled1.jsonl> <labeled2.jsonl> [...]");
        std::process::exit(2);
    }
    let reports: Vec<(String, EvalReport)> = paths
        .iter()
        .map(|p| {
            let data = crate::dataset::load(p).unwrap_or_else(|e| panic!("load {p}: {e}"));
            (short_name(p), evaluate(&data))
        })
        .collect();

    println!("labeler comparison — learned router on each set's holdout");
    println!(
        "{:<14} {:>5} {:>9} {:>9} {:>8} {:>10} {:>10}",
        "labeler", "n", "sp_learn", "sp_heur", "ordinal", "avg_cost", "cost@ceil"
    );
    for (name, r) in &reports {
        let ca = r
            .cost_at_adequacy_learned
            .map_or("n/a".to_string(), |c| format!("{c:.3}"));
        println!(
            "{:<14} {:>5} {:>9.3} {:>9.3} {:>8.3} {:>10.3} {:>10}",
            name,
            r.n_holdout,
            r.spearman_learned,
            r.spearman_heuristic,
            r.ordinal_learned,
            r.cost_learned,
            ca
        );
    }
    println!(
        "note: sp/ordinal/cost@ceil are vs each set's OWN labels; avg_cost shares holdout queries."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_assigns_average_rank_for_ties() {
        // [1.0, 2.0, 2.0, 3.0] → sorted positions: 1.0→0, 2.0→1, 2.0→2, 3.0→3
        // Ties at positions 1 and 2 get average rank (1+2)/2 = 1.5
        let v = [1.0, 2.0, 2.0, 3.0];
        let r = ranks(&v);
        assert!(
            (r[0] - 0.0).abs() < 1e-9,
            "rank of 1.0 should be 0.0, got {}",
            r[0]
        );
        assert!(
            (r[1] - 1.5).abs() < 1e-9,
            "rank of first 2.0 should be 1.5, got {}",
            r[1]
        );
        assert!(
            (r[2] - 1.5).abs() < 1e-9,
            "rank of second 2.0 should be 1.5, got {}",
            r[2]
        );
        assert!(
            (r[3] - 3.0).abs() < 1e-9,
            "rank of 3.0 should be 3.0, got {}",
            r[3]
        );
    }

    #[test]
    fn spearman_perfect_correlation_is_one() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [10.0, 20.0, 30.0, 40.0];
        assert!((spearman(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ordinal_accuracy_counts_matching_buckets() {
        let pred = [0.1, 0.5, 0.9];
        let label = [0.2, 0.8, 0.95]; // buckets: pred 0,1,2 ; label 0,2,2 -> 2/3
        assert!((ordinal_accuracy(&pred, &label) - (2.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn cost_at_adequacy_returns_finite_cost_when_target_reachable() {
        // Use easy labels (0.1) that every builtin model can handle adequately.
        // At target 0.9, the sweep should find at least one cost_bias where adequacy >= 0.9,
        // and the returned cost must be <= the always-strongest cost.
        let diffs = vec![0.1, 0.1, 0.1, 0.1];
        let labels = vec![0.1, 0.1, 0.1, 0.1];
        let result = cost_at_adequacy(&diffs, &labels, 0.9);
        assert!(result.is_some(), "expected Some cost for reachable target");
        let cost = result.unwrap();
        let (strongest_cost, _) = always_strongest_baseline(&labels);
        assert!(cost.is_finite(), "cost must be finite, got {cost}");
        assert!(
            cost <= strongest_cost + 1e-9,
            "cost {cost} should be <= always-strongest cost {strongest_cost}"
        );
    }

    #[test]
    fn cost_at_adequacy_returns_none_when_target_unreachable() {
        // Target 1.01 is impossible since adequacy_rate is in [0, 1].
        let diffs = vec![0.5, 0.5];
        let labels = vec![0.5, 0.5];
        let result = cost_at_adequacy(&diffs, &labels, 1.01);
        assert!(
            result.is_none(),
            "expected None for unreachable target 1.01"
        );
    }

    #[test]
    fn always_strongest_baseline_uses_max_quality_model() {
        // Derives expected values from registry::builtin() at runtime — no hard-coded
        // model name, quality, or cost here. With labels [0.5, 0.99]:
        //   row 0: strongest.quality >= 0.50 -> adequate
        //   row 1: strongest.quality >= 0.99 -> depends on registry
        // Expected adequacy and cost are computed from the live registry below.
        let labels = [0.5, 0.99];
        let (cost, adeq) = always_strongest_baseline(&labels);

        // Derive expected values from the actual builtin registry so this test
        // never hard-codes a model list that can diverge from the real one.
        let models = registry::builtin();
        let strongest = models
            .iter()
            .max_by(|a, b| a.quality.partial_cmp(&b.quality).unwrap())
            .unwrap();

        assert!(
            (cost - strongest.cost).abs() < 1e-12,
            "baseline cost must equal the strongest model's cost, got {cost}"
        );
        let expected_adeq =
            labels.iter().filter(|&&l| strongest.quality >= l).count() as f64 / labels.len() as f64;
        assert!(
            (adeq - expected_adeq).abs() < 1e-12,
            "baseline adequacy mismatch: got {adeq}, expected {expected_adeq}"
        );
    }

    fn sample_data() -> Vec<LabeledExample> {
        let rows: &[(&str, f64)] = &[
            ("hi", 0.0),
            ("hello there", 0.0),
            ("thanks", 0.0),
            ("good morning", 0.0),
            ("what is JSON", 0.25),
            ("define HTTP", 0.25),
            ("summarize DNA briefly", 0.25),
            ("list three facts about gravity", 0.25),
            ("implement a binary search in Rust", 0.5),
            ("write unit tests for a parser", 0.5),
            ("explain what this snippet does", 0.5),
            ("refactor a tangled module", 0.5),
            ("implement a lock-free concurrent queue", 0.75),
            ("analyze the convergence of a series", 0.75),
            ("optimize a hot loop and justify", 0.75),
            ("profile and tune an allocator", 0.75),
            ("prove by induction a statement about primes", 1.0),
            ("design Raft, prove correctness, analyze failures", 1.0),
            ("derive the closed form from first principles", 1.0),
            ("prove a tight lower bound", 1.0),
        ];
        rows.iter()
            .map(|(q, d)| LabeledExample {
                query: (*q).to_string(),
                difficulty: *d,
                category: String::new(),
            })
            .collect()
    }

    #[test]
    fn evaluate_produces_in_range_metrics() {
        let r = evaluate(&sample_data());
        assert!(r.n_holdout > 0, "holdout must be non-empty");
        for s in [r.spearman_learned, r.spearman_heuristic] {
            assert!((-1.0..=1.0).contains(&s), "spearman out of range: {s}");
        }
        for a in [
            r.ordinal_learned,
            r.ordinal_heuristic,
            r.adeq_learned,
            r.adeq_heuristic,
            r.adeq_strongest,
        ] {
            assert!((0.0..=1.0).contains(&a), "rate out of range: {a}");
        }
        for c in [r.cost_learned, r.cost_heuristic, r.cost_strongest] {
            assert!(c.is_finite() && c >= 0.0, "cost invalid: {c}");
        }
    }

    #[test]
    fn parse_in_flag_finds_path() {
        let with = vec![
            "eval".to_string(),
            "--in".to_string(),
            "data/x.jsonl".to_string(),
        ];
        assert_eq!(parse_in_flag(&with), Some("data/x.jsonl".to_string()));
        assert_eq!(parse_in_flag(&["eval".to_string()]), None);
        // --in present but no value
        assert_eq!(
            parse_in_flag(&["eval".to_string(), "--in".to_string()]),
            None
        );
    }

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
        assert_eq!(parse_flag(&["--gold".to_string()], "--gold"), None);
    }

    #[test]
    fn evaluate_gold_produces_in_range_metrics() {
        let train = sample_data();
        let gold = vec![
            LabeledExample {
                query: "hi".into(),
                difficulty: 0.0,
                category: "chat".into(),
            },
            LabeledExample {
                query: "prove a tight lower bound".into(),
                difficulty: 1.0,
                category: "math".into(),
            },
            LabeledExample {
                query: "implement a binary search in Rust".into(),
                difficulty: 0.5,
                category: "code".into(),
            },
            LabeledExample {
                query: "define HTTP".into(),
                difficulty: 0.25,
                category: "extraction".into(),
            },
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

    #[test]
    fn short_name_strips_path_and_prefix() {
        assert_eq!(short_name("data/labeled.claude.jsonl"), "claude");
        assert_eq!(short_name("data/labeled.codex.jsonl"), "codex");
        assert_eq!(short_name("data/labeled.jsonl"), "labeled");
        assert_eq!(short_name("/tmp/foo.jsonl"), "foo");
    }

    #[test]
    fn ceiling_target_reachable_with_some_impossible_labels() {
        // Holdout = indices 0,5,10,15. Mark a holdout entry impossible (1.0,
        // above every model's quality) so the achievable ceiling is < 1.0. The
        // old fixed 0.90 target could be unreachable here; targeting the ceiling
        // (sa) must stay reachable for both routers.
        let mut data = sample_data();
        data[15].difficulty = 1.0;
        let r = evaluate(&data);
        assert!(
            r.target_adequacy < 1.0,
            "ceiling should be < 1.0 with an impossible label, got {}",
            r.target_adequacy
        );
        assert!(
            r.cost_at_adequacy_learned.is_some(),
            "ceiling adequacy must be reachable (learned)"
        );
        assert!(
            r.cost_at_adequacy_heuristic.is_some(),
            "ceiling adequacy must be reachable (heuristic)"
        );
    }
}
