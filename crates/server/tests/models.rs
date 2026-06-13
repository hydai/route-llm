use axum_test::TestServer;
use route_llm_server::app;

#[tokio::test]
async fn lists_builtin_models() {
    let server = TestServer::new(app()).unwrap();
    let res = server.get("/v1/models").await;
    res.assert_status_ok();
    let body: serde_json::Value = res.json();
    let models = body["models"].as_array().unwrap();
    assert!(!models.is_empty());
    assert!(models.iter().any(|m| m["id"] == "claude-opus-4-8"));
    assert!(models[0]["quality"].is_number());
    assert!(models[0]["cost"].is_number());
}
