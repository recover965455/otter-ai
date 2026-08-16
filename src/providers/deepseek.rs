//! DeepSeek provider — chat completions compatible, `api.deepseek.com`.

use super::openai_compat::{build_compat_provider, CompatProviderSpec, GenericCompatProvider};
use crate::types::{Model, ModelCostRates, ModelThinkingLevel};

pub type DeepSeekProvider = GenericCompatProvider;

pub fn deepseek_provider() -> DeepSeekProvider {
    build_compat_provider(CompatProviderSpec {
        id: "deepseek",
        display_name: "DeepSeek",
        base_url: "https://api.deepseek.com/v1",
        env_var: "DEEPSEEK_API_KEY",
        key_placeholder: "sk-...",
        api_label: "openai-chat-completions",
        default_models_fn: default_models,
        extra_headers: None,
    })
}

fn default_models() -> Vec<Model> {
    vec![
        Model {
            id: "deepseek-chat".into(),
            provider_id: "deepseek".into(),
            name: "DeepSeek V3 Chat".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(64_000),
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
            cost_rates: ModelCostRates {
                input_per_million: Some(0.14),
                output_per_million: Some(0.28),
                ..Default::default()
            },
            context_window: Some(64_000),
            default_temperature: Some(1.0),
            thinking_level_map: None,
        },
        Model {
            id: "deepseek-reasoner".into(),
            provider_id: "deepseek".into(),
            name: "DeepSeek R1 Reasoner".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(64_000),
            max_output_tokens: Some(8_000),
            supports_images: false,
            supports_audio: false,
            supports_video: false,
            supports_pdf: false,
            supports_tool_calling: true,
            supports_structured_output: false,
            supports_system_prompt: true,
            thinking: ModelThinkingLevel::High,
            reasoning: true,
            cost_rates: ModelCostRates {
                input_per_million: Some(0.55),
                output_per_million: Some(2.19),
                ..Default::default()
            },
            context_window: Some(64_000),
            default_temperature: Some(1.0),
            thinking_level_map: None,
        },
    ]
}
