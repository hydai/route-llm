use axum_test::TestServer;
use route_llm_core::BudgetRouter;
use serde_json::json;

fn server() -> TestServer {
    let router = std::sync::Arc::new(BudgetRouter::new());
    TestServer::new(route_llm_server::app_with_router(router)).unwrap()
}

#[tokio::test]
async fn recommend_includes_budget_block() {
    let res = server()
        .post("/v1/recommend")
        .json(&json!({
            "query": "Design a production-grade distributed lock and analyze deadlock risk.",
            "models": [{"id": "claude-opus-4-8"}, {"id": "claude-haiku-4-5"}]
        }))
        .await;
    res.assert_status_ok();
    let v: serde_json::Value = res.json();
    assert!(v["ranking"].as_array().unwrap().len() == 2);
    let b = &v["budget"];
    assert!(b["level"].as_str().unwrap().starts_with('R'));
    assert!(b["recommended_model_tier"].is_string());
    assert!(b["confidence"].is_number());
}
