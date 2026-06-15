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

/// Set via `BudgetRouter` startup config (`ROUTE_LLM_POLICY`); not part of
/// `RoutingPreferences`. See SPEC-v3 §6.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Policy {
    #[default]
    Balanced,
    Strict,
    Cheap,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget: Option<BudgetBreakdown>,
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
            budget: None,
        };
        let v = serde_json::to_value(&rec).unwrap();
        assert_eq!(v["difficulty"]["score"], 0.5);
        assert_eq!(v["ranking"][0]["id"], "m");
    }

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
}
