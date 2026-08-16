//! Vercel AI Gateway — OpenAI-compatible reverse proxy with analytics.
//! Requires `AI_GATEWAY_API_KEY` for authorization; the backend provider
//! is selected via Vercel-side config or the standard provider-specific
//! `base_url` override pattern.

use std::collections::HashMap;

use super::openai_compat::{build_compat_provider, CompatProviderSpec, GenericCompatProvider};
use crate::types::{Model, ModelCostRates, ModelThinkingLevel};

pub type VercelAiGatewayProvider = GenericCompatProvider;

pub fn vercel_ai_gateway_provider() -> VercelAiGatewayProvider {
    let mut extra = HashMap::new();
    // Vercel AI Gateway expects the consumer key as a bearer AND additionally
    // advertises `x-vercel-id`; injecting the header here is harmless.
    extra.insert("x-vercel-ai-gateway".into(), "1".into());

    build_compat_provider(CompatProviderSpec {
        id: "vercel-ai-gateway",
        display_name: "Vercel AI Gateway",
        base_url: "https://ai-gateway.vercel.dev/api/v1",
        env_var: "AI_GATEWAY_API_KEY",
        key_placeholder: "vai_...",
        api_label: "openai-chat-completions",
        default_models_fn: default_models,
        extra_headers: Some(extra),
    })
}

fn default_models() -> Vec<Model> {
    vec![
        // Vercel AI Gateway transparently routes to upstream providers, so the
        // "model" list here is really just the most commonly pinned upstreams.
        Model {
            id: "gpt-4o-mini".into(),
            provider_id: "vercel-ai-gateway".into(),
            name: "GPT-4o Mini (via Vercel AI Gateway)".into(),
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
            cost_rates: ModelCostRates::default(),
            context_window: Some(128_000),
            default_temperature: Some(1.0),
        },
    ]
}
