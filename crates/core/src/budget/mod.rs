//! Budget router subsystem (v3). Isolated from v1 heuristic & v2 learned cores.
pub mod dims;
pub mod escalation;
pub mod level;
pub mod weights;

use crate::learned::model::LinearModel;
use crate::learned::weights::shipped_model as learned_shipped_model;
use crate::model::{
    BudgetBreakdown, Difficulty, ModelProfile, Policy, Recommendation, RoutingPreferences,
};
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

        let decision = escalation::decide(
            self.policy,
            &lower,
            &dim_arr,
            score,
            base_level,
            learned_level,
        );

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
            ModelProfile {
                id: "strong".into(),
                quality: 0.97,
                cost: 0.90,
            },
            ModelProfile {
                id: "cheap".into(),
                quality: 0.60,
                cost: 0.10,
            },
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
