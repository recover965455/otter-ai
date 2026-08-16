//! Mistral provider — `api.mistral.ai`; also compatible with local
//! Mistral-compatible servers pointed at via a custom base URL.

use super::openai_compat::{build_compat_provider, CompatProviderSpec, GenericCompatProvider};
use crate::types::{Model, ModelCostRates, ModelThinkingLevel};

pub type MistralProvider = GenericCompatProvider;

pub fn mistral_provider() -> MistralProvider {
    build_compat_provider(CompatProviderSpec {
        id: "mistral",
        display_name: "Mistral",
        base_url: "https://api.mistral.ai/v1",
        env_var: "MISTRAL_API_KEY",
        key_placeholder: "trMZ...",
        api_label: "mistral-conversations",
        default_models_fn: default_models,
        extra_headers: None,
    })
}

fn default_models() -> Vec<Model> {
    vec![
        Model {
            id: "mistral-large-latest".into(),
            provider_id: "mistral".into(),
            name: "Mistral Large".into(),
            api: "mistral-conversations".into(),
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
                input_per_million: Some(2.0),
                output_per_million: Some(6.0),
                ..Default::default()
            },
            context_window: Some(128_000),
            default_temperature: Some(0.7),
        },
        Model {
            id: "mistral-small-latest".into(),
            provider_id: "mistral".into(),
            name: "Mistral Small".into(),
            api: "mistral-conversations".into(),
            max_input_tokens: Some(32_000),
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
                input_per_million: Some(0.2),
                output_per_million: Some(0.6),
                ..Default::default()
            },
            context_window: Some(32_000),
            default_temperature: Some(0.7),
        },
        Model {
            id: "codestral-latest".into(),
            provider_id: "mistral".into(),
            name: "Codestral".into(),
            api: "mistral-conversations".into(),
            max_input_tokens: Some(80_000),
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
                input_per_million: Some(0.2),
                output_per_million: Some(0.6),
                ..Default::default()
            },
            context_window: Some(80_000),
            default_temperature: Some(0.7),
        },
    ]
}
