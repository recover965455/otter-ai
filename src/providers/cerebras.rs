//! Cerebras provider — fastest inference for small-to-medium parameter models.
//! Wire-compatible with OpenAI chat completions.

use super::openai_compat::{build_compat_provider, CompatProviderSpec, GenericCompatProvider};
use crate::types::{Model, ModelCostRates, ModelThinkingLevel};

pub type CerebrasProvider = GenericCompatProvider;

pub fn cerebras_provider() -> CerebrasProvider {
    build_compat_provider(CompatProviderSpec {
        id: "cerebras",
        display_name: "Cerebras",
        base_url: "https://api.cerebras.ai/v1",
        env_var: "CEREBRAS_API_KEY",
        key_placeholder: "sk-cerebras-...",
        api_label: "openai-chat-completions",
        default_models_fn: default_models,
        extra_headers: None,
    })
}

fn default_models() -> Vec<Model> {
    vec![
        Model {
            id: "llama3.1-8b".into(),
            provider_id: "cerebras".into(),
            name: "Llama 3.1 8B (Cerebras)".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(8_000),
            max_output_tokens: Some(8_000),
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
            context_window: Some(8_000),
            default_temperature: Some(1.0),
            thinking_level_map: None,
        },
        Model {
            id: "llama3.1-70b".into(),
            provider_id: "cerebras".into(),
            name: "Llama 3.1 70B (Cerebras)".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(8_000),
            max_output_tokens: Some(8_000),
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
            context_window: Some(8_000),
            default_temperature: Some(1.0),
            thinking_level_map: None,
        },
    ]
}
