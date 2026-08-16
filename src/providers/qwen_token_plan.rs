//! Qwen Token Plan Individual — Alibaba's Qwen international subscription,
//! authenticated via `QWEN_TOKEN_PLAN_API_KEY`.  OpenAI chat-completions
//! compatible.

use super::openai_compat::{build_compat_provider, CompatProviderSpec, GenericCompatProvider};
use crate::types::{Model, ModelCostRates, ModelThinkingLevel};

pub type QwenTokenPlanProvider = GenericCompatProvider;

pub fn qwen_token_plan_provider() -> QwenTokenPlanProvider {
    build_compat_provider(CompatProviderSpec {
        id: "qwen-token-plan",
        display_name: "Qwen Token Plan",
        base_url: "https://api.qwen.ai/v1",
        env_var: "QWEN_TOKEN_PLAN_API_KEY",
        key_placeholder: "qwen-...",
        api_label: "openai-chat-completions",
        default_models_fn: default_models,
        extra_headers: None,
    })
}

fn default_models() -> Vec<Model> {
    vec![
        Model {
            id: "qwen-coder-plus".into(),
            provider_id: "qwen-token-plan".into(),
            name: "Qwen Coder Plus".into(),
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
            thinking_level_map: None,
        },
        Model {
            id: "qwen-max".into(),
            provider_id: "qwen-token-plan".into(),
            name: "Qwen Max".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(128_000),
            max_output_tokens: Some(8_192),
            supports_images: true,
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
            thinking_level_map: None,
        },
        Model {
            id: "qwen-turbo".into(),
            provider_id: "qwen-token-plan".into(),
            name: "Qwen Turbo".into(),
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
            thinking_level_map: None,
        },
    ]
}
