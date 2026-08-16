//! ZAI / Codin provider — `api.zai.ai` / other ZAI-branded endpoints, fully
//! OpenAI chat completions compatible.

use super::openai_compat::{build_compat_provider, CompatProviderSpec, GenericCompatProvider};
use crate::types::{Model, ModelCostRates, ModelThinkingLevel};

pub type ZaiProvider = GenericCompatProvider;

pub fn zai_provider() -> ZaiProvider {
    build_compat_provider(CompatProviderSpec {
        id: "zai",
        display_name: "ZAI (Codin)",
        base_url: "https://api.zai.ai/v1",
        env_var: "ZAI_API_KEY",
        key_placeholder: "zai-...",
        api_label: "openai-chat-completions",
        default_models_fn: default_models,
        extra_headers: None,
    })
}

fn default_models() -> Vec<Model> {
    vec![Model {
        id: "zai-codin".into(),
        provider_id: "zai".into(),
        name: "ZAI Codin".into(),
        api: "openai-chat-completions".into(),
        max_input_tokens: Some(128_000),
        max_output_tokens: Some(32_768),
        supports_images: false,
        supports_audio: false,
        supports_video: false,
        supports_pdf: false,
        supports_tool_calling: true,
        supports_structured_output: true,
        supports_system_prompt: true,
        thinking: ModelThinkingLevel::None,
        reasoning: false,
        cost_rates: ModelCostRates::default(),
        context_window: Some(128_000),
        default_temperature: Some(1.0),
        thinking_level_map: None,
    }]
}
