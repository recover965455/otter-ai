//! Moonshot AI / Kimi Coding — Chinese LLM provider, OpenAI chat-completions
//! compatible.  Authenticated via `MOONSHOT_API_KEY`.
//!
//! For Moonshot AI China (mainland endpoint), set `MOONSHOT_BASE_URL` to
//! the domestic URL or override `base_url` at construction time.

use super::openai_compat::{build_compat_provider, CompatProviderSpec, GenericCompatProvider};
use crate::types::{Model, ModelCostRates, ModelThinkingLevel};

pub type MoonshotProvider = GenericCompatProvider;

pub fn moonshot_provider() -> MoonshotProvider {
    build_compat_provider(CompatProviderSpec {
        id: "moonshot",
        display_name: "Moonshot AI (Kimi)",
        base_url: "https://api.moonshot.cn/v1",
        env_var: "MOONSHOT_API_KEY",
        key_placeholder: "sk-...",
        api_label: "openai-chat-completions",
        default_models_fn: default_models,
        extra_headers: None,
    })
}

fn default_models() -> Vec<Model> {
    vec![
        Model {
            id: "kimi-k3".into(),
            provider_id: "moonshot".into(),
            name: "Kimi K3".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(256_000),
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
            context_window: Some(256_000),
            default_temperature: Some(1.0),
        },
        Model {
            id: "moonshot-v1-128k".into(),
            provider_id: "moonshot".into(),
            name: "Moonshot V1 128K".into(),
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
            cost_rates: ModelCostRates::default(),
            context_window: Some(128_000),
            default_temperature: Some(1.0),
        },
        Model {
            id: "moonshot-v1-32k".into(),
            provider_id: "moonshot".into(),
            name: "Moonshot V1 32K".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(32_000),
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
            context_window: Some(32_000),
            default_temperature: Some(1.0),
        },
    ]
}
