use std::sync::atomic::{AtomicU64, Ordering};

use axum::extract::rejection::JsonRejection;
use axum::Json;
use serde_json::{json, Value};

use route_llm_core::{
    registry, CandidateInput, HeuristicRouter, Recommendation, Router, RoutingPreferences,
};

use crate::dto::{
    AnthropicContent, AnthropicUsage, ChatChoice, ChatCompletionRequest, ChatCompletionResponse,
    ChatRespMessage, MessagesRequest, MessagesResponse, ModelInput, OpenAiUsage, PrefsInput,
    RecommendRequest,
};
use crate::error::ApiError;

static ID_COUNTER: AtomicU64 = AtomicU64::new(1);

pub(crate) fn next_id() -> String {
    format!("rec-{:016x}", ID_COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// Human-readable one-liner describing the recommendation.
pub(crate) fn summary_line(rec: &Recommendation) -> String {
    let order = rec
        .ranking
        .iter()
        .map(|r| r.id.as_str())
        .collect::<Vec<_>>()
        .join(" > ");
    let top = rec
        .ranking
        .first()
        .map(|r| r.id.as_str())
        .unwrap_or("(none)");
    format!(
        "Recommended: {} (difficulty {:.2}). Order: {}.",
        top, rec.difficulty.score, order
    )
}

pub async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

pub async fn list_models() -> Json<Value> {
    Json(json!({ "models": registry::builtin() }))
}

/// Merge a candidate list with an optional standard `model` field (hint).
pub(crate) fn collect_candidates(
    model: Option<String>,
    models: Vec<ModelInput>,
) -> Vec<CandidateInput> {
    let mut out: Vec<CandidateInput> = models
        .into_iter()
        .filter_map(|m| {
            let id = m.id.trim();
            if id.is_empty() {
                None
            } else {
                Some(CandidateInput {
                    id: id.to_string(),
                    quality: m.quality,
                    cost: m.cost,
                })
            }
        })
        .collect();
    if let Some(id) = model {
        let id = id.trim();
        if !id.is_empty() && !out.iter().any(|c| c.id == id) {
            out.push(CandidateInput {
                id: id.to_string(),
                quality: None,
                cost: None,
            });
        }
    }
    out
}

pub(crate) fn prefs_or_default(p: Option<PrefsInput>) -> RoutingPreferences {
    p.map(Into::into).unwrap_or_default()
}

/// Shared across all three dialects: validate, resolve, route.
pub(crate) fn process(
    query: &str,
    candidates: Vec<CandidateInput>,
    prefs: RoutingPreferences,
) -> Result<Recommendation, ApiError> {
    if query.trim().is_empty() {
        return Err(ApiError::EmptyQuery);
    }
    if candidates.is_empty() {
        return Err(ApiError::EmptyCandidates);
    }
    if !(0.0..=1.0).contains(&prefs.cost_bias) {
        return Err(ApiError::InvalidPreferences(
            "cost_bias must be in 0.0..=1.0".into(),
        ));
    }
    let profiles = registry::resolve(&candidates).map_err(ApiError::UnknownModels)?;
    let in_range = |v: f64| v.is_finite() && (0.0..=1.0).contains(&v);
    for p in &profiles {
        if !in_range(p.quality) || !in_range(p.cost) {
            return Err(ApiError::InvalidModel(p.id.clone()));
        }
    }
    Ok(HeuristicRouter.recommend(query, &profiles, &prefs))
}

pub async fn recommend(
    payload: Result<Json<RecommendRequest>, JsonRejection>,
) -> Result<Json<Recommendation>, ApiError> {
    let Json(req) = payload.map_err(|e| ApiError::InvalidJson(e.body_text()))?;
    let candidates = collect_candidates(None, req.models);
    let rec = process(&req.query, candidates, prefs_or_default(req.preferences))?;
    Ok(Json(rec))
}

pub async fn chat_completions(
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Json<ChatCompletionResponse>, ApiError> {
    let Json(req) = payload.map_err(|e| ApiError::InvalidJson(e.body_text()))?;
    let query = req
        .messages
        .iter()
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let candidates = collect_candidates(req.model, req.models);
    let rec = process(&query, candidates, prefs_or_default(req.preferences))?;
    let top = rec
        .ranking
        .first()
        .map(|r| r.id.clone())
        .unwrap_or_default();

    let resp = ChatCompletionResponse {
        id: next_id(),
        object: "chat.completion",
        model: top,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatRespMessage {
                role: "assistant",
                content: summary_line(&rec),
            },
            finish_reason: "stop",
        }],
        usage: OpenAiUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
        route_llm: rec,
    };
    Ok(Json(resp))
}

pub async fn messages(
    payload: Result<Json<MessagesRequest>, JsonRejection>,
) -> Result<Json<MessagesResponse>, ApiError> {
    let Json(req) = payload.map_err(|e| ApiError::InvalidJson(e.body_text()))?;
    let mut parts: Vec<String> = Vec::new();
    if let Some(sys) = req.system {
        parts.push(sys);
    }
    parts.extend(req.messages.into_iter().map(|m| m.content));
    let query = parts.join("\n");

    let candidates = collect_candidates(req.model, req.models);
    let rec = process(&query, candidates, prefs_or_default(req.preferences))?;
    let top = rec
        .ranking
        .first()
        .map(|r| r.id.clone())
        .unwrap_or_default();

    let resp = MessagesResponse {
        id: next_id(),
        kind: "message",
        role: "assistant",
        model: top,
        content: vec![AnthropicContent {
            kind: "text",
            text: summary_line(&rec),
        }],
        stop_reason: "end_turn",
        usage: AnthropicUsage {
            input_tokens: 0,
            output_tokens: 0,
        },
        route_llm: rec,
    };
    Ok(Json(resp))
}
