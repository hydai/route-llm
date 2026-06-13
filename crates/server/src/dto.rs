use route_llm_core::{CandidateInput, RoutingPreferences};
use serde::Deserialize;

/// A candidate model entry in a request (id + optional overrides).
#[derive(Debug, Deserialize)]
pub struct ModelInput {
    pub id: String,
    #[serde(default)]
    pub quality: Option<f64>,
    #[serde(default)]
    pub cost: Option<f64>,
}

impl From<ModelInput> for CandidateInput {
    fn from(m: ModelInput) -> Self {
        CandidateInput {
            id: m.id,
            quality: m.quality,
            cost: m.cost,
        }
    }
}

/// Optional routing preferences in a request body.
#[derive(Debug, Deserialize)]
pub struct PrefsInput {
    pub cost_bias: f64,
}

impl From<PrefsInput> for RoutingPreferences {
    fn from(p: PrefsInput) -> Self {
        RoutingPreferences {
            cost_bias: p.cost_bias,
        }
    }
}

/// Native `/v1/recommend` request.
#[derive(Debug, Deserialize)]
pub struct RecommendRequest {
    pub query: String,
    #[serde(default)]
    pub models: Vec<ModelInput>,
    #[serde(default)]
    pub preferences: Option<PrefsInput>,
}
