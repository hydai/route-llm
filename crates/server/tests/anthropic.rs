use axum_test::TestServer;
use route_llm_server::app;
use serde_json::json;

#[tokio::test]
async fn messages_returns_message_envelope() {
    let server = TestServer::new(app()).unwrap();
    let res = server
        .post("/v1/messages")
        .json(&json!({
            "model": "claude-haiku-4-5",
            "system": "You are concise.",
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
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert_eq!(body["model"], "claude-opus-4-8");
    assert_eq!(body["content"][0]["type"], "text");
    assert_eq!(body["route_llm"]["ranking"][0]["id"], "claude-opus-4-8");
    assert_eq!(body["usage"]["output_tokens"], 0);
}
