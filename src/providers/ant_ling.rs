//! Ant Ling — Ant Group's AI coding assistant.  OpenAI chat-completions
//! compatible, authenticated via `ANT_LING_API_KEY`.

use super::openai_compat::{build_compat_provider, CompatProviderSpec, GenericCompatProvider};
use crate::types::{Model, ModelCostRates, ModelThinkingLevel};

pub type AntLingProvider = GenericCompatProvider;

pub fn ant_ling_provider() -> AntLingProvider {
    build_compat_provider(CompatProviderSpec {
        id: "ant-ling",
        display_name: "Ant Ling",
        base_url: "https://api.antling.ai/v1",
        env_var: "ANT_LING_API_KEY",
        key_placeholder: "al-...",
        api_label: "openai-chat-completions",
        default_models_fn: default_models,
        extra_headers: None,
    })
}

fn default_models() -> Vec<Model> {
    vec![Model {
        id: "ant-ling-coding".into(),
        provider_id: "ant-ling".into(),
        name: "Ant Ling Coding".into(),
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
        cost_rates: ModelCostRates::default(),
        context_window: Some(128_000),
        default_temperature: Some(1.0),
        thinking_level_map: None,
    }]
}
