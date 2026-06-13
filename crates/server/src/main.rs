use tracing_subscriber::EnvFilter;

/// Resolves the ROUTE_LLM_ROUTER env-var result to a router name.
///
/// Returns `Ok(name)` on a valid or absent value, or `Err(message)` when the
/// value is unrecognised or not valid UTF-8.  Callers should `eprintln!` the
/// message and `exit(1)` on `Err`.
fn choose_router(var: Result<&str, &std::env::VarError>) -> Result<&'static str, String> {
    match var {
        Ok("heuristic") => Ok("heuristic"),
        Ok("learned") => Ok("learned"),
        // Genuinely unset → default to learned (SPEC-v2 §9)
        Err(std::env::VarError::NotPresent) => Ok("learned"),
        // Non-UTF-8 value: fail fast instead of silently defaulting
        Err(_) => Err("ROUTE_LLM_ROUTER value is not valid UTF-8".to_string()),
        Ok(other) => Err(format!(
            "invalid ROUTE_LLM_ROUTER: {other:?} (expected 'learned' or 'heuristic')"
        )),
    }
}

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

    let router_name = match choose_router(std::env::var("ROUTE_LLM_ROUTER").as_deref()) {
        Ok(name) => name,
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(1);
        }
    };

    let (router, router_name): (route_llm_server::SharedRouter, &'static str) =
        match router_name {
            "heuristic" => (
                std::sync::Arc::new(route_llm_core::HeuristicRouter),
                "heuristic",
            ),
            _ => (
                std::sync::Arc::new(route_llm_core::LearnedRouter::new()),
                "learned",
            ),
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

#[cfg(test)]
mod tests {
    use super::choose_router;

    #[test]
    fn unset_defaults_to_learned() {
        assert_eq!(
            choose_router(Err(&std::env::VarError::NotPresent)),
            Ok("learned")
        );
    }

    #[test]
    fn explicit_learned_selects_learned() {
        assert_eq!(choose_router(Ok("learned")), Ok("learned"));
    }

    #[test]
    fn explicit_heuristic_selects_heuristic() {
        assert_eq!(choose_router(Ok("heuristic")), Ok("heuristic"));
    }

    #[test]
    fn unknown_value_is_error() {
        let result = choose_router(Ok("unknown"));
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("invalid ROUTE_LLM_ROUTER"),
            "unexpected error: {msg}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn not_unicode_is_error() {
        // Simulate VarError::NotUnicode with a synthetic OsString.
        // Unix-only: OsStringExt::from_vec is the portable-enough way to forge
        // non-UTF-8 bytes; the production code path is platform-agnostic.
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;
        let bad = OsString::from_vec(vec![0xFF, 0xFE]);
        let err = std::env::VarError::NotUnicode(bad);
        let result = choose_router(Err(&err));
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("not valid UTF-8"),
            "unexpected error: {msg}"
        );
    }
}
