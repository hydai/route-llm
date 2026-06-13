//! Learned router subsystem (v2). Isolated from v1's heuristic scorer.
pub mod features;
pub mod model;
pub mod weights;

use crate::model::{ModelProfile, Recommendation, RoutingPreferences};
use crate::ranker; // ★ shared v1 ranker, unchanged
use crate::router::Router;
use model::LinearModel;

/// v2 strategy: learned difficulty + the shared cost-quality ranker.
pub struct LearnedRouter {
    model: LinearModel,
}

impl Default for LearnedRouter {
    fn default() -> Self {
        Self {
            model: weights::shipped_model(),
        }
    }
}

impl LearnedRouter {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Router for LearnedRouter {
    fn recommend(
        &self,
        query: &str,
        models: &[ModelProfile],
        prefs: &RoutingPreferences,
    ) -> Recommendation {
        let difficulty = self.model.difficulty(query);
        let ranking = ranker::rank(&difficulty, models, prefs);
        Recommendation {
            difficulty,
            ranking,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn trivial_query_is_easier_than_hard_query() {
        let r = LearnedRouter::new();
        let easy = r.recommend("hi", &models(), &RoutingPreferences::default());
        let hard = r.recommend(
            "Prove step by step why Paxos is safe and derive its invariant; analyze a partition.",
            &models(),
            &RoutingPreferences::default(),
        );
        assert!(
            hard.difficulty.score > easy.difficulty.score,
            "hard {} vs easy {}",
            hard.difficulty.score,
            easy.difficulty.score
        );
    }

    #[test]
    fn produces_full_ranking() {
        let r = LearnedRouter::new();
        let rec = r.recommend("hello", &models(), &RoutingPreferences::default());
        assert_eq!(rec.ranking.len(), 2);
    }
}
