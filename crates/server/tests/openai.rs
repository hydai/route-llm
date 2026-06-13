use axum_test::TestServer;
use route_llm_server::app;
use serde_json::json;

#[tokio::test]
async fn chat_completions_returns_completion_envelope() {
    let server = TestServer::new(app()).unwrap();
    let res = server
        .post("/v1/chat/completions")
        .json(&json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Explain why the Paxos consensus algorithm guarantees safety, derive the invariant it maintains, and analyze step by step how it handles a network partition."}],
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
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "claude-opus-4-8");
    assert_eq!(body["route_llm"]["ranking"][0]["id"], "claude-opus-4-8");
    assert_eq!(body["usage"]["total_tokens"], 0);
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(content.starts_with("Recommended:"));
}

#[tokio::test]
async fn model_field_alone_is_used_as_candidate() {
    let server = TestServer::new(app()).unwrap();
    let res = server
        .post("/v1/chat/completions")
        .json(&json!({
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .await;
    res.assert_status_ok();
    let body: serde_json::Value = res.json();
    assert_eq!(body["route_llm"]["ranking"].as_array().unwrap().len(), 1);
    assert_eq!(body["model"], "claude-haiku-4-5");
}

#[tokio::test]
async fn whitespace_model_field_is_not_a_candidate() {
    let server = TestServer::new(app()).unwrap();
    let res = server
        .post("/v1/chat/completions")
        .json(&json!({ "model": "  ", "messages": [{"role": "user", "content": "hi"}] }))
        .await;
    res.assert_status_bad_request();
    assert_eq!(
        res.json::<serde_json::Value>()["error"]["code"],
        "empty_candidates"
    );
}
