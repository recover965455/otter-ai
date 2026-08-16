//! Fireworks AI — high-speed inference platform, OpenAI chat-completions
//! compatible.  Authenticated via `FIREWORKS_API_KEY`.

use super::openai_compat::{build_compat_provider, CompatProviderSpec, GenericCompatProvider};
use crate::types::{Model, ModelCostRates, ModelThinkingLevel};

pub type FireworksProvider = GenericCompatProvider;

pub fn fireworks_provider() -> FireworksProvider {
    build_compat_provider(CompatProviderSpec {
        id: "fireworks",
        display_name: "Fireworks AI",
        base_url: "https://api.fireworks.ai/inference/v1",
        env_var: "FIREWORKS_API_KEY",
        key_placeholder: "fw-...",
        api_label: "openai-chat-completions",
        default_models_fn: default_models,
        extra_headers: None,
    })
}

fn default_models() -> Vec<Model> {
    vec![
        Model {
            id: "accounts/fireworks/models/llama-v3p1-405b-instruct".into(),
            provider_id: "fireworks".into(),
            name: "Llama 3.1 405B Instruct (Fireworks)".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(128_000),
            max_output_tokens: Some(16_384),
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
                input_per_million: Some(3.0),
                output_per_million: Some(3.0),
                ..Default::default()
            },
            context_window: Some(128_000),
            default_temperature: Some(1.0),
        },
        Model {
            id: "accounts/fireworks/models/llama-v3p1-70b-instruct".into(),
            provider_id: "fireworks".into(),
            name: "Llama 3.1 70B Instruct (Fireworks)".into(),
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
                input_per_million: Some(0.9),
                output_per_million: Some(0.9),
                ..Default::default()
            },
            context_window: Some(128_000),
            default_temperature: Some(1.0),
        },
        Model {
            id: "accounts/fireworks/models/deepseek-v3".into(),
            provider_id: "fireworks".into(),
            name: "DeepSeek V3 (Fireworks)".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(64_000),
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
            cost_rates: ModelCostRates::default(),
            context_window: Some(64_000),
            default_temperature: Some(1.0),
        },
    ]
}
