//! OpenRouter provider — unified API aggregator, fully OpenAI
//! Chat-Completions compatible.  Supports passing the requesting app name
//! via `HTTP-Referer` and `X-Title` headers for OpenRouter leaderboards,
//! which is why this provider has the extra_headers hook wired in.

use std::collections::HashMap;

use super::openai_compat::{build_compat_provider, CompatProviderSpec, GenericCompatProvider};
use crate::types::{Model, ModelCostRates, ModelThinkingLevel};

pub type OpenRouterProvider = GenericCompatProvider;

pub fn openrouter_provider() -> OpenRouterProvider {
    let mut extra = HashMap::new();
    extra.insert(
        "HTTP-Referer".into(),
        "https://github.com/recover965455/otter-ai".into(),
    );
    extra.insert("X-Title".into(), "otter-ai".into());

    build_compat_provider(CompatProviderSpec {
        id: "openrouter",
        display_name: "OpenRouter",
        base_url: "https://openrouter.ai/api/v1",
        env_var: "OPENROUTER_API_KEY",
        key_placeholder: "sk-or-v1-...",
        api_label: "openai-chat-completions",
        default_models_fn: default_models,
        extra_headers: Some(extra),
    })
}

fn default_models() -> Vec<Model> {
    vec![
        Model {
            id: "anthropic/claude-3.7-sonnet".into(),
            provider_id: "openrouter".into(),
            name: "Claude 3.7 Sonnet (OpenRouter)".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(200_000),
            max_output_tokens: Some(64_000),
            supports_images: true,
            supports_audio: false,
            supports_video: false,
            supports_pdf: true,
            supports_tool_calling: true,
            supports_structured_output: true,
            supports_system_prompt: true,
            thinking: ModelThinkingLevel::High,
            reasoning: false,
            cost_rates: ModelCostRates {
                input_per_million: Some(3.0),
                output_per_million: Some(15.0),
                ..Default::default()
            },
            context_window: Some(200_000),
            default_temperature: Some(1.0),
            thinking_level_map: None,
        },
        Model {
            id: "openai/gpt-4o-mini".into(),
            provider_id: "openrouter".into(),
            name: "GPT-4o Mini (OpenRouter)".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(128_000),
            max_output_tokens: Some(16_384),
            supports_images: true,
            supports_audio: false,
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
            thinking_level_map: None,
        },
        Model {
            id: "google/gemini-2.5-pro-exp-03-25:free".into(),
            provider_id: "openrouter".into(),
            name: "Gemini 2.5 Pro Exp (Free, OpenRouter)".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(1_000_000),
            max_output_tokens: Some(65_536),
            supports_images: true,
            supports_audio: true,
            supports_video: true,
            supports_pdf: true,
            supports_tool_calling: true,
            supports_structured_output: true,
            supports_system_prompt: true,
            thinking: ModelThinkingLevel::None,
            reasoning: false,
            cost_rates: ModelCostRates::default(),
            context_window: Some(1_000_000),
            default_temperature: Some(1.0),
            thinking_level_map: None,
        },
    ]
}
