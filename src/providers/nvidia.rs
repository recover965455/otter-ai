//! NVIDIA NIM provider — fully OpenAI chat-completions compatible, with
//! `integrate.api.nvidia.com/v1` as the canonical base URL.

use super::openai_compat::{build_compat_provider, CompatProviderSpec, GenericCompatProvider};
use crate::types::{Model, ModelCostRates, ModelThinkingLevel};

pub type NvidiaProvider = GenericCompatProvider;

pub fn nvidia_provider() -> NvidiaProvider {
    build_compat_provider(CompatProviderSpec {
        id: "nvidia",
        display_name: "NVIDIA NIM",
        base_url: "https://integrate.api.nvidia.com/v1",
        env_var: "NVIDIA_API_KEY",
        key_placeholder: "nvapi-...",
        api_label: "openai-chat-completions",
        default_models_fn: default_models,
        extra_headers: None,
    })
}

fn default_models() -> Vec<Model> {
    vec![
        Model {
            id: "meta/llama-3.1-405b-instruct".into(),
            provider_id: "nvidia".into(),
            name: "Llama 3.1 405B Instruct (NVIDIA NIM)".into(),
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
            cost_rates: ModelCostRates::default(),
            context_window: Some(128_000),
            default_temperature: Some(1.0),
        },
        Model {
            id: "mistralai/mixtral-8x22b-instruct-v0.1".into(),
            provider_id: "nvidia".into(),
            name: "Mixtral 8x22B v0.1 (NVIDIA NIM)".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(64_000),
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
            cost_rates: ModelCostRates::default(),
            context_window: Some(64_000),
            default_temperature: Some(1.0),
        },
    ]
}
