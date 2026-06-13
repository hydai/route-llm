use axum::extract::rejection::JsonRejection;
use axum::Json;
use serde_json::{json, Value};

use route_llm_core::{
    registry, CandidateInput, HeuristicRouter, Recommendation, Router, RoutingPreferences,
};

use crate::dto::{ModelInput, PrefsInput, RecommendRequest};
use crate::error::ApiError;

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
    let mut out: Vec<CandidateInput> = models.into_iter().map(Into::into).collect();
    if let Some(id) = model {
        if !id.is_empty() && !out.iter().any(|c| c.id == id) {
            out.push(CandidateInput {
                id,
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
