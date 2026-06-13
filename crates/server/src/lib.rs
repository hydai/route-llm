//! route-llm HTTP server.

pub mod handlers;

use axum::routing::get;
use axum::Router;

/// Build the axum application (used by both `main` and integration tests).
pub fn app() -> Router {
    Router::new().route("/health", get(handlers::health))
}
