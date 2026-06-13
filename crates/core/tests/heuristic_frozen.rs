use route_llm_core::{HeuristicRouter, ModelProfile, Router, RoutingPreferences};

#[test]
fn heuristic_difficulty_matches_v1_on_spec_query() {
    let q = "Explain why the Paxos consensus algorithm guarantees safety, derive the invariant it maintains, and analyze step by step how it handles a network partition.";
    let models = [ModelProfile {
        id: "m".into(),
        quality: 0.9,
        cost: 0.5,
    }];
    let rec = HeuristicRouter.recommend(q, &models, &RoutingPreferences { cost_bias: 0.3 });
    // v1 SPEC §9: this query lands difficulty in (0.6, 0.85).
    assert!(
        rec.difficulty.score > 0.6 && rec.difficulty.score < 0.85,
        "v1 heuristic difficulty changed: {}",
        rec.difficulty.score
    );
    assert!(rec.difficulty.signals.contains(&"reasoning".to_string()));
}
