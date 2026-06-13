use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("invalid JSON body: {0}")]
    InvalidJson(String),
    #[error("query text is empty")]
    EmptyQuery,
    #[error("no candidate models provided")]
    EmptyCandidates,
    #[error("unknown models: {0:?}")]
    UnknownModels(Vec<String>),
    #[error("{0}")]
    InvalidPreferences(String),
    #[error("model '{0}' has a non-finite or out-of-range quality/cost (must be finite and within 0.0..=1.0)")]
    InvalidModel(String),
}

impl ApiError {
    fn code(&self) -> &'static str {
        match self {
            ApiError::InvalidJson(_) => "invalid_json",
            ApiError::EmptyQuery => "empty_query",
            ApiError::EmptyCandidates => "empty_candidates",
            ApiError::UnknownModels(_) => "unknown_models",
            ApiError::InvalidPreferences(_) => "invalid_preferences",
            ApiError::InvalidModel(_) => "invalid_model",
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let details = match &self {
            ApiError::UnknownModels(ids) => json!({ "unknown": ids }),
            ApiError::InvalidModel(id) => json!({ "model": id }),
            _ => json!({}),
        };
        let body = json!({
            "error": { "code": self.code(), "message": self.to_string(), "details": details }
        });
        (StatusCode::BAD_REQUEST, Json(body)).into_response()
    }
}
