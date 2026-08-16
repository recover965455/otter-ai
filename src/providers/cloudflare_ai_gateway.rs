//! Cloudflare AI Gateway — sits in front of your chosen upstream providers
//! and injects Cloudflare billing/observability headers.
//!
//! Auth model (aligned with pi docs `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID` +
//! `CLOUDFLARE_GATEWAY_ID`): the API key itself is read from `CLOUDFLARE_API_KEY`
//! (via the standard compat env-var resolver) and the account/gateway ids are
//! injected as static extra headers here, **or** the caller can override by
//! passing their own Config instead of using the default constructor.

use std::collections::HashMap;

use super::openai_compat::{
    build_compat_provider, CompatProviderSpec, GenericCompatConfig, GenericCompatProvider,
};
use crate::types::{Model, ModelCostRates, ModelThinkingLevel};

pub type CloudflareAiGatewayProvider = GenericCompatProvider;
pub use super::openai_compat::GenericCompatConfig as CloudflareAiGatewayConfig;

pub fn cloudflare_ai_gateway_provider() -> CloudflareAiGatewayProvider {
    build_compat_provider(CompatProviderSpec {
        id: "cloudflare-ai-gateway",
        display_name: "Cloudflare AI Gateway",
        base_url: "https://gateway.ai.cloudflare.com/v1",
        env_var: "CLOUDFLARE_API_KEY",
        key_placeholder: "...",
        api_label: "openai-chat-completions",
        default_models_fn: default_models,
        extra_headers: None, // Account/Gateway IDs go in the URL path in practice; callers override base_url.
    })
}

fn default_models() -> Vec<Model> {
    vec![Model {
        id: "@cf/meta/llama-3.1-8b-instruct".into(),
        provider_id: "cloudflare-ai-gateway".into(),
        name: "Llama 3.1 8B Instruct (Cloudflare Workers AI via Gateway)".into(),
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
    }]
}

// Silence unused-warning for re-export.
#[allow(dead_code)]
fn _reexport(_: CloudflareAiGatewayConfig) {}
