//! Groq provider — ultra-fast inference, full OpenAI chat completions compatibility.

use super::openai_compat::{build_compat_provider, CompatProviderSpec, GenericCompatProvider};
use crate::types::{Model, ModelCostRates, ModelThinkingLevel};

pub type GroqProvider = GenericCompatProvider;

pub fn groq_provider() -> GroqProvider {
    build_compat_provider(CompatProviderSpec {
        id: "groq",
        display_name: "Groq",
        base_url: "https://api.groq.com/openai/v1",
        env_var: "GROQ_API_KEY",
        key_placeholder: "gsk_...",
        api_label: "openai-chat-completions",
        default_models_fn: default_models,
        extra_headers: None,
    })
}

fn default_models() -> Vec<Model> {
    vec![
        Model {
            id: "llama-3.3-70b-versatile".into(),
            provider_id: "groq".into(),
            name: "Llama 3.3 70B (Groq)".into(),
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
            cost_rates: ModelCostRates {
                input_per_million: Some(0.59),
                output_per_million: Some(0.79),
                ..Default::default()
            },
            context_window: Some(128_000),
            default_temperature: Some(1.0),
        },
        Model {
            id: "llama-3.1-8b-instant".into(),
            provider_id: "groq".into(),
            name: "Llama 3.1 8B Instant (Groq)".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(128_000),
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
                input_per_million: Some(0.04),
                output_per_million: Some(0.06),
                ..Default::default()
            },
            context_window: Some(128_000),
            default_temperature: Some(1.0),
        },
        Model {
            id: "mixtral-8x7b-32768".into(),
            provider_id: "groq".into(),
            name: "Mixtral 8x7B 32K (Groq)".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(32_768),
            max_output_tokens: Some(4_096),
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
                input_per_million: Some(0.24),
                output_per_million: Some(0.24),
                ..Default::default()
            },
            context_window: Some(32_768),
            default_temperature: Some(1.0),
        },
    ]
}
