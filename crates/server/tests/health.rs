use axum_test::TestServer;
use route_llm_server::app;
use serde_json::json;

#[tokio::test]
async fn health_returns_ok() {
    let server = TestServer::new(app()).unwrap();
    let res = server.get("/health").await;
    res.assert_status_ok();
    res.assert_json(&json!({ "status": "ok" }));
}
