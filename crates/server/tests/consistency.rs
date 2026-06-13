use axum_test::TestServer;
use route_llm_server::app;
use serde_json::{json, Value};

const QUERY: &str = "Explain why the Paxos consensus algorithm guarantees safety, derive the invariant it maintains, and analyze step by step how it handles a network partition.";

fn models() -> Value {
    json!([
        {"id": "claude-opus-4-8"},
        {"id": "claude-haiku-4-5"},
        {"id": "gpt-4o-mini", "quality": 0.55, "cost": 0.10}
    ])
}

fn order(ranking: &Value) -> Vec<String> {
    ranking
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap().to_string())
        .collect()
}

#[tokio::test]
async fn all_three_dialects_agree_on_ranking() {
    let server = TestServer::new(app()).unwrap();

    let native = server
        .post("/v1/recommend")
        .json(&json!({ "query": QUERY, "models": models(), "preferences": {"cost_bias": 0.3} }))
        .await
        .json::<Value>();

    let openai = server
        .post("/v1/chat/completions")
        .json(&json!({
            "messages": [{"role": "user", "content": QUERY}],
            "models": models(),
            "preferences": {"cost_bias": 0.3}
        }))
        .await
        .json::<Value>();

    let anthropic = server
        .post("/v1/messages")
        .json(&json!({
            "messages": [{"role": "user", "content": QUERY}],
            "models": models(),
            "preferences": {"cost_bias": 0.3}
        }))
        .await
        .json::<Value>();

    let native_order = order(&native["ranking"]);
    assert_eq!(native_order, order(&openai["route_llm"]["ranking"]));
    assert_eq!(native_order, order(&anthropic["route_llm"]["ranking"]));
    assert_eq!(native_order[0], "claude-opus-4-8");
}
