use crate::difficulty::sigmoid; // reuse shared sigmoid (pub(crate) in v1)
use crate::learned::features::{feature_name, features};
use crate::model::Difficulty;

/// A standardized linear model producing a 0..1 difficulty via logistic link.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearModel {
    pub schema_version: u32,
    pub weights: Vec<f64>,
    pub bias: f64,
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
}

const STD_FLOOR: f64 = 1e-9;
const TOP_SIGNALS: usize = 4;

impl LinearModel {
    /// Standardize a raw feature value for index `i`.
    fn z(&self, i: usize, x: f64) -> f64 {
        let s = if self.stds[i].abs() < STD_FLOOR {
            1.0
        } else {
            self.stds[i]
        };
        (x - self.means[i]) / s
    }

    /// Compute difficulty for a query: sigmoid(w·z + bias). `signals` = top
    /// positive contributors (w_i · z_i), for explainability.
    pub fn difficulty(&self, query: &str) -> Difficulty {
        let x = features(query);
        debug_assert_eq!(self.weights.len(), x.len(), "weights length must match feature_count");
        debug_assert_eq!(self.means.len(), x.len(), "means length must match feature_count");
        debug_assert_eq!(self.stds.len(), x.len(), "stds length must match feature_count");
        let mut sum = self.bias;
        let mut contrib: Vec<(usize, f64)> = Vec::with_capacity(x.len());
        for (i, &xi) in x.iter().enumerate() {
            let c = self.weights[i] * self.z(i, xi);
            sum += c;
            contrib.push((i, c));
        }
        contrib.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let signals: Vec<String> = contrib
            .iter()
            .filter(|(_, c)| *c > 0.0)
            .take(TOP_SIGNALS)
            .map(|(i, _)| feature_name(*i))
            .collect();
        Difficulty {
            score: sigmoid(sum),
            signals,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::features::{feature_count, SCHEMA_VERSION};

    fn identity_model(weights: Vec<f64>, bias: f64) -> LinearModel {
        let n = feature_count();
        LinearModel {
            schema_version: SCHEMA_VERSION,
            weights,
            bias,
            means: vec![0.0; n],
            stds: vec![1.0; n], // std=1, mean=0 -> standardize is identity
        }
    }

    #[test]
    fn difficulty_is_in_unit_interval() {
        let m = identity_model(vec![0.0; feature_count()], 0.0);
        let d = m.difficulty("anything");
        assert!(d.score > 0.0 && d.score < 1.0);
        assert!((d.score - 0.5).abs() < 1e-9); // all-zero weights+bias -> 0.5
    }

    #[test]
    fn positive_weight_on_code_raises_difficulty() {
        let mut w = vec![0.0; feature_count()];
        w[1] = 5.0; // has_code
        let m = identity_model(w, -1.0);
        let hard = m.difficulty("```fn main(){}```").score;
        let easy = m.difficulty("hello").score;
        assert!(hard > easy, "hard {hard} should exceed easy {easy}");
    }

    #[test]
    fn signals_lists_top_contributors() {
        let mut w = vec![0.0; feature_count()];
        w[1] = 5.0; // has_code dominates
        let m = identity_model(w, -1.0);
        let d = m.difficulty("```code```");
        assert!(d.signals.contains(&"has_code".to_string()));
    }

    #[test]
    fn std_zero_is_protected() {
        let n = feature_count();
        let m = LinearModel {
            schema_version: SCHEMA_VERSION,
            weights: vec![0.0; n],
            bias: 0.0,
            means: vec![0.0; n],
            stds: vec![0.0; n], // zero std must not NaN
        };
        assert!(m.difficulty("x").score.is_finite());
    }
}
