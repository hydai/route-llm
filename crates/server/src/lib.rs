//! route-llm HTTP server.

pub mod dto;
pub mod error;
pub mod handlers;

use axum::routing::{get, post};
use std::sync::Arc;

/// A boxed routing strategy shared across handlers (axum state).
pub type SharedRouter = Arc<dyn route_llm_core::Router + Send + Sync>;

/// Build the axum application with an explicit routing strategy.
pub fn app_with_router(router: SharedRouter) -> axum::Router {
    axum::Router::new()
        .route("/health", get(handlers::health))
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/recommend", post(handlers::recommend))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/messages", post(handlers::messages))
        .with_state(router)
}

/// Back-compat constructor used by v1 integration tests: heuristic strategy.
pub fn app() -> axum::Router {
    app_with_router(Arc::new(route_llm_core::HeuristicRouter))
}
