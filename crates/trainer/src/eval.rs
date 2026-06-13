use crate::dataset::LabeledExample;
use crate::logreg::{self, FitConfig};
use route_llm_core::{difficulty, ranker, registry, ModelProfile, RoutingPreferences};

fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0; v.len()];
    for (rank, &i) in idx.iter().enumerate() {
        r[i] = rank as f64;
    }
    r
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
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
    if pred.is_empty() {
        return 0.0;
    }
    let ok = (0..pred.len())
        .filter(|&i| bucket(pred[i]) == bucket(label[i]))
        .count();
    ok as f64 / pred.len() as f64
}

/// Average cost of the top-1 pick over the builtin registry, plus the fraction
/// of picks that are "adequate" (chosen quality >= the query's true difficulty).
fn cost_profile(diffs: &[f64], labels: &[f64]) -> (f64, f64) {
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
        "  avg cost   learned={:.3} (adeq {:.2})  heuristic={:.3} (adeq {:.2})",
        lc, la, hc, ha
    );
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
