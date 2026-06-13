use axum_test::TestServer;
use route_llm_server::app;
use serde_json::json;

fn server() -> TestServer {
    TestServer::new(app()).unwrap()
}

#[tokio::test]
async fn recommend_happy_path_orders_models() {
    let res = server()
        .post("/v1/recommend")
        .json(&json!({
            "query": "Explain why the Paxos consensus algorithm guarantees safety, derive the invariant it maintains, and analyze step by step how it handles a network partition.",
            "models": [
                {"id": "claude-opus-4-8"},
                {"id": "claude-haiku-4-5"},
                {"id": "gpt-4o-mini", "quality": 0.55, "cost": 0.10}
            ],
            "preferences": {"cost_bias": 0.3}
        }))
        .await;
    res.assert_status_ok();
    let body: serde_json::Value = res.json();
    assert_eq!(body["ranking"][0]["id"], "claude-opus-4-8");
    let score = body["difficulty"]["score"].as_f64().unwrap();
    assert!(score > 0.6 && score < 0.85);
}

#[tokio::test]
async fn empty_query_is_rejected() {
    let res = server()
        .post("/v1/recommend")
        .json(&json!({ "query": "  ", "models": [{"id": "gpt-4o-mini"}] }))
        .await;
    res.assert_status_bad_request();
    assert_eq!(
        res.json::<serde_json::Value>()["error"]["code"],
        "empty_query"
    );
}

#[tokio::test]
async fn empty_candidates_is_rejected() {
    let res = server()
        .post("/v1/recommend")
        .json(&json!({ "query": "hi", "models": [] }))
        .await;
    res.assert_status_bad_request();
    assert_eq!(
        res.json::<serde_json::Value>()["error"]["code"],
        "empty_candidates"
    );
}

#[tokio::test]
async fn unknown_model_is_rejected_with_details() {
    let res = server()
        .post("/v1/recommend")
        .json(&json!({ "query": "hi", "models": [{"id": "does-not-exist"}] }))
        .await;
    res.assert_status_bad_request();
    let body: serde_json::Value = res.json();
    assert_eq!(body["error"]["code"], "unknown_models");
    assert_eq!(body["error"]["details"]["unknown"][0], "does-not-exist");
}

#[tokio::test]
async fn invalid_cost_bias_is_rejected() {
    let res = server()
        .post("/v1/recommend")
        .json(&json!({
            "query": "hi",
            "models": [{"id": "gpt-4o-mini"}],
            "preferences": {"cost_bias": 1.5}
        }))
        .await;
    res.assert_status_bad_request();
    assert_eq!(
        res.json::<serde_json::Value>()["error"]["code"],
        "invalid_preferences"
    );
}

#[tokio::test]
async fn malformed_json_is_rejected() {
    let res = server()
        .post("/v1/recommend")
        .text("{ not json")
        .content_type("application/json")
        .await;
    res.assert_status_bad_request();
    assert_eq!(
        res.json::<serde_json::Value>()["error"]["code"],
        "invalid_json"
    );
}

#[tokio::test]
async fn out_of_range_override_is_rejected() {
    let res = server()
        .post("/v1/recommend")
        .json(&json!({
            "query": "hi",
            "models": [{"id": "x", "quality": 1.5, "cost": 0.1}]
        }))
        .await;
    res.assert_status_bad_request();
    assert_eq!(
        res.json::<serde_json::Value>()["error"]["code"],
        "invalid_model"
    );
}
