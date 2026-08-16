//! Cloudflare Workers AI — serverless models run directly on Cloudflare's
//! edge network.  Uses the same API-style as OpenAI chat completions with
//! the account id embedded in the base URL.

use super::openai_compat::{build_compat_provider, CompatProviderSpec, GenericCompatProvider};
use crate::types::{Model, ModelCostRates, ModelThinkingLevel};

pub type CloudflareWorkersAiProvider = GenericCompatProvider;

pub fn cloudflare_workers_ai_provider() -> CloudflareWorkersAiProvider {
    build_compat_provider(CompatProviderSpec {
        id: "cloudflare-workers-ai",
        display_name: "Cloudflare Workers AI",
        base_url: "https://api.cloudflare.com/client/v4/accounts/ACCOUNT_ID/ai/v1",
        env_var: "CLOUDFLARE_API_KEY",
        key_placeholder: "...",
        api_label: "openai-chat-completions",
        default_models_fn: default_models,
        extra_headers: None,
    })
}

fn default_models() -> Vec<Model> {
    vec![
        Model {
            id: "@cf/meta/llama-3.1-8b-instruct".into(),
            provider_id: "cloudflare-workers-ai".into(),
            name: "Llama 3.1 8B Instruct (Workers AI)".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(32_000),
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
            context_window: Some(32_000),
            default_temperature: Some(1.0),
            thinking_level_map: None,
        },
        Model {
            id: "@cf/qwen/qwen1.5-14b-chat-awq".into(),
            provider_id: "cloudflare-workers-ai".into(),
            name: "Qwen 1.5 14B Chat AWQ (Workers AI)".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(32_000),
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
            context_window: Some(32_000),
            default_temperature: Some(1.0),
            thinking_level_map: None,
        },
    ]
}
