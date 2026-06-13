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
            difficulty: Difficulty {
                score: 0.5,
                signals: vec!["code".into()],
            },
            ranking: vec![RankedModel {
                id: "m".into(),
                score: 0.4,
                reason: "r".into(),
            }],
        };
        let v = serde_json::to_value(&rec).unwrap();
        assert_eq!(v["difficulty"]["score"], 0.5);
        assert_eq!(v["ranking"][0]["id"], "m");
    }
}
