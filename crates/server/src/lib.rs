//! route-llm HTTP server.

pub mod dto;
pub mod error;
pub mod handlers;

use axum::routing::{get, post};
use axum::Router;

/// Build the axum application (used by both `main` and integration tests).
pub fn app() -> Router {
    Router::new()
        .route("/health", get(handlers::health))
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/recommend", post(handlers::recommend))
}
