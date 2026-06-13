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
    debug_assert_eq!(pred.len(), label.len(), "ordinal_accuracy: slice length mismatch");
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
    debug_assert_eq!(diffs.len(), labels.len(), "cost_profile: slice length mismatch");
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
    debug_assert_eq!(diffs.len(), labels.len(), "cost_at_adequacy: slice length mismatch");
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

/// `eval`: fit on a train split, report metrics on holdout for learned vs heuristic.
pub fn run() {
    let data = crate::dataset::load("data/labeled.jsonl").expect("load labeled.jsonl");
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

    // SPEC §12: fixed-adequacy cost metric at 90% adequacy target.
    let target = 0.90;
    let learned_fa = cost_at_adequacy(&learned, &labels, target);
    let heuristic_fa = cost_at_adequacy(&heuristic, &labels, target);
    // always-strongest cost is the ceiling: it is always the max-quality model's cost.
    let strongest_fa_cost = sc;

    eprintln!("eval (holdout n={})", holdout.len());
    eprintln!(
        "  spearman   learned={:.3}  heuristic={:.3}",
        spearman(&learned, &labels),
        spearman(&heuristic, &labels)
    );
    eprintln!(
        "  ordinal    learned={:.3}  heuristic={:.3}",
        ordinal_accuracy(&learned, &labels),
        ordinal_accuracy(&heuristic, &labels)
    );
    eprintln!(
        "  avg cost   learned={:.3} (adeq {:.2})  heuristic={:.3} (adeq {:.2})  always-strongest={:.3} (adeq {:.2})",
        lc, la, hc, ha, sc, sa
    );
    eprintln!(
        "  cost @ adequacy>={:.2}  learned={}  heuristic={}  always-strongest={:.3}",
        target,
        learned_fa.map_or("n/a (target unreachable)".into(), |c| format!("{c:.3}")),
        heuristic_fa.map_or("n/a (target unreachable)".into(), |c| format!("{c:.3}")),
        strongest_fa_cost,
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
        assert!(
            cost.is_finite(),
            "cost must be finite, got {cost}"
        );
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
        assert!(result.is_none(), "expected None for unreachable target 1.01");
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
}
