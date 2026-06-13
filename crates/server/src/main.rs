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

    let (router, router_name): (route_llm_server::SharedRouter, &'static str) =
        match std::env::var("ROUTE_LLM_ROUTER").as_deref() {
            Ok("heuristic") => (
                std::sync::Arc::new(route_llm_core::HeuristicRouter),
                "heuristic",
            ),
            // default (unset) is the learned strategy per SPEC-v2 §9
            Ok("learned") | Err(_) => (
                std::sync::Arc::new(route_llm_core::LearnedRouter::new()),
                "learned",
            ),
            Ok(other) => {
                eprintln!(
                    "invalid ROUTE_LLM_ROUTER: {other:?} (expected 'learned' or 'heuristic')"
                );
                std::process::exit(1);
            }
        };
    tracing::info!("route-llm using router strategy: {}", router_name);
    let app = route_llm_server::app_with_router(router);

    let listener = tokio::net::TcpListener::bind((host.as_str(), port))
        .await
        .expect("failed to bind ROUTE_LLM_HOST/PORT");
    let addr = listener.local_addr().expect("failed to read local address");
    tracing::info!("route-llm listening on http://{addr}");

    axum::serve(listener, app).await.expect("server error");
}
