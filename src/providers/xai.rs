//! xAI provider (Grok) — `api.x.ai/v1`, OpenAI chat completions wire protocol.

use super::openai_compat::{build_compat_provider, CompatProviderSpec, GenericCompatProvider};
use crate::types::{Model, ModelCostRates, ModelThinkingLevel};

pub type XaiProvider = GenericCompatProvider;

pub fn xai_provider() -> XaiProvider {
    build_compat_provider(CompatProviderSpec {
        id: "xai",
        display_name: "xAI",
        base_url: "https://api.x.ai/v1",
        env_var: "XAI_API_KEY",
        key_placeholder: "xai-...",
        api_label: "openai-chat-completions",
        default_models_fn: default_models,
        extra_headers: None,
    })
}

fn default_models() -> Vec<Model> {
    vec![
        Model {
            id: "grok-4".into(),
            provider_id: "xai".into(),
            name: "Grok 4".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(1_048_576),
            max_output_tokens: Some(131_072),
            supports_images: false,
            supports_audio: false,
            supports_video: false,
            supports_pdf: false,
            supports_tool_calling: true,
            supports_structured_output: true,
            supports_system_prompt: true,
            thinking: ModelThinkingLevel::Medium,
            reasoning: true,
            cost_rates: ModelCostRates {
                input_per_million: Some(3.0),
                output_per_million: Some(15.0),
                ..Default::default()
            },
            context_window: Some(1_048_576),
            default_temperature: Some(1.0),
            thinking_level_map: None,
        },
        Model {
            id: "grok-3-mini-fast".into(),
            provider_id: "xai".into(),
            name: "Grok 3 Mini Fast".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(128_000),
            max_output_tokens: Some(8_192),
            supports_images: false,
            supports_audio: false,
            supports_video: false,
            supports_pdf: false,
            supports_tool_calling: true,
            supports_structured_output: true,
            supports_system_prompt: true,
            thinking: ModelThinkingLevel::None,
            reasoning: false,
            cost_rates: ModelCostRates {
                input_per_million: Some(0.1),
                output_per_million: Some(0.3),
                ..Default::default()
            },
            context_window: Some(128_000),
            default_temperature: Some(1.0),
            thinking_level_map: None,
        },
    ]
}
