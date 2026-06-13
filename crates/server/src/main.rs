use std::net::SocketAddr;

use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let host = std::env::var("ROUTE_LLM_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port: u16 = match std::env::var("ROUTE_LLM_PORT") {
        Ok(v) => v
            .parse()
            .expect("ROUTE_LLM_PORT must be a valid port number (0-65535)"),
        Err(_) => 8080,
    };
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .expect("invalid ROUTE_LLM_HOST/PORT");

    let app = route_llm_server::app();
    tracing::info!("route-llm listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind");
    axum::serve(listener, app).await.expect("server error");
}
