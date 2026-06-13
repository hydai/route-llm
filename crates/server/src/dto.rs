use route_llm_core::{CandidateInput, Recommendation, RoutingPreferences};
use serde::{Deserialize, Serialize};

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

/// A chat message (request side). We only support string `content` in v1.
#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub content: String,
}

/// OpenAI-shaped `/v1/chat/completions` request.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub models: Vec<ModelInput>,
    #[serde(default)]
    pub preferences: Option<PrefsInput>,
}

#[derive(Debug, Serialize)]
pub struct ChatRespMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatRespMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct OpenAiUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: OpenAiUsage,
    pub route_llm: Recommendation,
}

/// Anthropic-shaped `/v1/messages` request.
#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub models: Vec<ModelInput>,
    #[serde(default)]
    pub preferences: Option<PrefsInput>,
}

#[derive(Debug, Serialize)]
pub struct AnthropicContent {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<AnthropicContent>,
    pub stop_reason: &'static str,
    pub usage: AnthropicUsage,
    pub route_llm: Recommendation,
}
