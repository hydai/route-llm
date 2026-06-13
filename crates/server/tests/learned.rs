use axum_test::TestServer;
use route_llm_server::app_with_router;
use serde_json::json;
use std::sync::Arc;

fn learned_server() -> TestServer {
    TestServer::new(app_with_router(Arc::new(
        route_llm_core::LearnedRouter::new(),
    )))
    .unwrap()
}

#[tokio::test]
async fn learned_router_serves_recommendations() {
    let res = learned_server()
        .post("/v1/recommend")
        .json(&json!({
            "query": "Prove step by step why Paxos is safe and derive the invariant.",
            "models": [{"id": "claude-opus-4-8"}, {"id": "gpt-4o-mini"}]
        }))
        .await;
    res.assert_status_ok();
    let body: serde_json::Value = res.json();
    let score = body["difficulty"]["score"].as_f64().unwrap();
    assert!(score > 0.0 && score < 1.0);
    assert_eq!(body["ranking"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn invalid_inputs_still_rejected_under_learned() {
    let res = learned_server()
        .post("/v1/recommend")
        .json(&json!({ "query": "  ", "models": [{"id": "gpt-4o-mini"}] }))
        .await;
    res.assert_status_bad_request();
}
