use crate::dataset::LabeledExample;
use route_llm_core::learned::features::{feature_count, features, SCHEMA_VERSION};
use route_llm_core::learned::model::LinearModel;

pub struct FitConfig {
    pub lr: f64,
    pub iters: usize,
    pub l2: f64,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            lr: 0.1,
            iters: 2000,
            l2: 1e-3,
        } // ★ tunable
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn standardization(rows: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let n = feature_count();
    let m = rows.len().max(1) as f64;
    let mut means = vec![0.0; n];
    for r in rows {
        for i in 0..n {
            means[i] += r[i];
        }
    }
    for v in means.iter_mut() {
        *v /= m;
    }
    let mut stds = vec![0.0; n];
    for r in rows {
        for i in 0..n {
            let d = r[i] - means[i];
            stds[i] += d * d;
        }
    }
    for v in stds.iter_mut() {
        *v = (*v / m).sqrt();
    }
    (means, stds)
}

/// Fit logistic regression on standardized features. Deterministic.
pub fn fit(examples: &[LabeledExample], cfg: &FitConfig) -> LinearModel {
    let n = feature_count();
    let raw: Vec<Vec<f64>> = examples.iter().map(|e| features(&e.query)).collect();
    let (means, stds) = standardization(&raw);
    let standardize = |r: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|i| {
                let s = if stds[i].abs() < 1e-9 { 1.0 } else { stds[i] };
                (r[i] - means[i]) / s
            })
            .collect()
    };
    let xs: Vec<Vec<f64>> = raw.iter().map(|r| standardize(r)).collect();
    let ys: Vec<f64> = examples
        .iter()
        .map(|e| e.difficulty.clamp(1e-6, 1.0 - 1e-6))
        .collect();

    let mut w = vec![0.0; n];
    let mut b = 0.0;
    let m = examples.len().max(1) as f64;
    for _ in 0..cfg.iters {
        let mut gw = vec![0.0; n];
        let mut gb = 0.0;
        for (x, &y) in xs.iter().zip(&ys) {
            let mut s = b;
            for i in 0..n {
                s += w[i] * x[i];
            }
            let err = sigmoid(s) - y; // cross-entropy gradient on the logit
            for i in 0..n {
                gw[i] += err * x[i];
            }
            gb += err;
        }
        for i in 0..n {
            w[i] -= cfg.lr * (gw[i] / m + cfg.l2 * w[i]);
        }
        b -= cfg.lr * (gb / m);
    }
    LinearModel {
        schema_version: SCHEMA_VERSION,
        weights: w,
        bias: b,
        means,
        stds,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> Vec<LabeledExample> {
        vec![
            LabeledExample {
                query: "hi".into(),
                difficulty: 0.05,
                category: "c".into(),
            },
            LabeledExample {
                query: "thanks".into(),
                difficulty: 0.05,
                category: "c".into(),
            },
            LabeledExample {
                query: "Prove step by step and derive the invariant; analyze.".into(),
                difficulty: 0.95,
                category: "r".into(),
            },
            LabeledExample {
                query: "Analyze, compare, and design; justify each step.".into(),
                difficulty: 0.95,
                category: "r".into(),
            },
        ]
    }

    #[test]
    fn fit_is_deterministic() {
        assert_eq!(
            fit(&data(), &FitConfig::default()),
            fit(&data(), &FitConfig::default())
        );
    }

    #[test]
    fn learns_easy_vs_hard_separation() {
        let m = fit(&data(), &FitConfig::default());
        let easy = m.difficulty("hi").score;
        let hard = m
            .difficulty("Prove step by step and derive; analyze.")
            .score;
        assert!(hard > easy, "hard {hard} vs easy {easy}");
    }
}
