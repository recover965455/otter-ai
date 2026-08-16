//! Baseten — managed model inference platform.  OpenAI chat-completions
//! compatible, authenticated via `BASETEN_API_KEY`.

use super::openai_compat::{build_compat_provider, CompatProviderSpec, GenericCompatProvider};
use crate::types::{Model, ModelCostRates, ModelThinkingLevel};

pub type BasetenProvider = GenericCompatProvider;

pub fn baseten_provider() -> BasetenProvider {
    build_compat_provider(CompatProviderSpec {
        id: "baseten",
        display_name: "Baseten",
        base_url: "https://api.baseten.co/v1",
        env_var: "BASETEN_API_KEY",
        key_placeholder: "b-...",
        api_label: "openai-chat-completions",
        default_models_fn: default_models,
        extra_headers: None,
    })
}

fn default_models() -> Vec<Model> {
    vec![
        Model {
            id: "deepseek-v3".into(),
            provider_id: "baseten".into(),
            name: "DeepSeek V3 (Baseten)".into(),
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
        Model {
            id: "llama-3.1-405b-instruct".into(),
            provider_id: "baseten".into(),
            name: "Llama 3.1 405B Instruct (Baseten)".into(),
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
            cost_rates: ModelCostRates::default(),
            context_window: Some(128_000),
            default_temperature: Some(1.0),
        },
    ]
}
