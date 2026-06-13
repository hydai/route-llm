use crate::model::{ModelProfile, Recommendation, RoutingPreferences};
use crate::{difficulty, ranker};

/// A routing strategy. v1 ships one implementation; future strategies plug in here.
pub trait Router {
    fn recommend(
        &self,
        query: &str,
        models: &[ModelProfile],
        prefs: &RoutingPreferences,
    ) -> Recommendation;
}

/// v1's first strategy: heuristic difficulty scoring + cost-quality ranking.
pub struct HeuristicRouter;

impl Router for HeuristicRouter {
    fn recommend(
        &self,
        query: &str,
        models: &[ModelProfile],
        prefs: &RoutingPreferences,
    ) -> Recommendation {
        let difficulty = difficulty::score(query);
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

    #[test]
    fn heuristic_router_orders_spec_example() {
        // SPEC §9 example: hard query, cost_bias 0.3 -> opus > haiku > gpt-4o-mini.
        let query = "Explain why the Paxos consensus algorithm guarantees safety, derive the invariant it maintains, and analyze step by step how it handles a network partition.";
        let models = [
            ModelProfile {
                id: "claude-opus-4-8".into(),
                quality: 0.97,
                cost: 0.90,
            },
            ModelProfile {
                id: "claude-haiku-4-5".into(),
                quality: 0.75,
                cost: 0.12,
            },
            ModelProfile {
                id: "gpt-4o-mini".into(),
                quality: 0.55,
                cost: 0.10,
            },
        ];
        let rec = HeuristicRouter.recommend(query, &models, &RoutingPreferences { cost_bias: 0.3 });

        let order: Vec<&str> = rec.ranking.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(
            order,
            ["claude-opus-4-8", "claude-haiku-4-5", "gpt-4o-mini"]
        );
        assert!(rec.difficulty.score > 0.6 && rec.difficulty.score < 0.85);
        assert!(rec.difficulty.signals.contains(&"reasoning".to_string()));
    }
}
