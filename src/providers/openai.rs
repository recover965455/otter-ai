//! OpenAI provider (official `api.openai.com`) built on top of
//! [`crate::providers::openai_compat`] shared primitives.

use super::openai_compat::{build_compat_provider, CompatProviderSpec, GenericCompatProvider};
use crate::types::{Model, ModelCostRates, ModelThinkingLevel};

pub use super::openai_compat::{
    GenericCompatApiKeyAuth as OpenAIApiKeyAuth, GenericCompatConfig as OpenAIProviderConfig,
};

pub type OpenAIProvider = GenericCompatProvider;

pub fn openai_provider() -> OpenAIProvider {
    build_compat_provider(CompatProviderSpec {
        id: "openai",
        display_name: "OpenAI",
        base_url: "https://api.openai.com/v1",
        env_var: "OPENAI_API_KEY",
        key_placeholder: "sk-...",
        api_label: "openai-chat-completions",
        default_models_fn: default_models,
        extra_headers: None,
    })
}

fn default_models() -> Vec<Model> {
    vec![
        Model {
            id: "gpt-4o-mini".into(),
            provider_id: "openai".into(),
            name: "GPT-4o Mini".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(128_000),
            max_output_tokens: Some(16_384),
            supports_images: true,
            supports_audio: true,
            supports_video: false,
            supports_pdf: false,
            supports_tool_calling: true,
            supports_structured_output: true,
            supports_system_prompt: true,
            thinking: ModelThinkingLevel::None,
            reasoning: false,
            cost_rates: ModelCostRates {
                input_per_million: Some(0.15),
                output_per_million: Some(0.60),
                ..Default::default()
            },
            context_window: Some(128_000),
            default_temperature: Some(1.0),
        },
        Model {
            id: "gpt-4o".into(),
            provider_id: "openai".into(),
            name: "GPT-4o".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(128_000),
            max_output_tokens: Some(16_384),
            supports_images: true,
            supports_audio: true,
            supports_video: true,
            supports_pdf: false,
            supports_tool_calling: true,
            supports_structured_output: true,
            supports_system_prompt: true,
            thinking: ModelThinkingLevel::None,
            reasoning: false,
            cost_rates: ModelCostRates {
                input_per_million: Some(2.50),
                output_per_million: Some(10.0),
                ..Default::default()
            },
            context_window: Some(128_000),
            default_temperature: Some(1.0),
        },
        Model {
            id: "o3-mini".into(),
            provider_id: "openai".into(),
            name: "o3 Mini".into(),
            api: "openai-responses".into(),
            max_input_tokens: Some(200_000),
            max_output_tokens: Some(100_000),
            supports_images: false,
            supports_audio: false,
            supports_video: false,
            supports_pdf: false,
            supports_tool_calling: true,
            supports_structured_output: true,
            supports_system_prompt: false,
            thinking: ModelThinkingLevel::Medium,
            reasoning: true,
            cost_rates: ModelCostRates {
                input_per_million: Some(1.10),
                output_per_million: Some(4.40),
                ..Default::default()
            },
            context_window: Some(200_000),
            default_temperature: Some(1.0),
        },
        Model {
            id: "o1".into(),
            provider_id: "openai".into(),
            name: "o1".into(),
            api: "openai-responses".into(),
            max_input_tokens: Some(200_000),
            max_output_tokens: Some(100_000),
            supports_images: true,
            supports_audio: false,
            supports_video: false,
            supports_pdf: false,
            supports_tool_calling: true,
            supports_structured_output: true,
            supports_system_prompt: false,
            thinking: ModelThinkingLevel::High,
            reasoning: true,
            cost_rates: ModelCostRates {
                input_per_million: Some(15.0),
                output_per_million: Some(60.0),
                ..Default::default()
            },
            context_window: Some(200_000),
            default_temperature: Some(1.0),
        },
    ]
}
