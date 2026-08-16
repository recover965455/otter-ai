//! OpenRouter OAuth (PKCE) — creates a user-controlled API key billed
//! from your OpenRouter credits.  On remote/headless machines the browser
//! cannot reach the loopback callback; paste the final redirect URL (or
//! the authorization code) into the login prompt instead.
//!
//! `OPENROUTER_API_KEY` remains available through the API-key OpenRouter
//! provider for non-OAuth usage.

use super::oauth_compat::{build_oauth_provider, GenericOAuthProvider, OAuthProviderSpec};

pub type OpenRouterOauthProvider = GenericOAuthProvider;

pub fn openrouter_oauth_provider() -> OpenRouterOauthProvider {
    build_oauth_provider(OAuthProviderSpec {
        id: "openrouter-oauth",
        display_name: "OpenRouter (OAuth)",
        base_url: "https://openrouter.ai/api/v1",
        // TODO: verify client_id from pi-ai source — OpenRouter PKCE.
        client_id: "app-openrouter-oauth",
        scopes: &["openid", "profile"],
        auth_url: Some("https://openrouter.ai/auth"),
        token_url: Some("https://openrouter.ai/api/v1/auth/keys"),
        device_auth_url: None,
        redirect_uri: Some("http://localhost:1455/callback"),
        is_subscription: false, // mints a user-controlled API key, not a subscription
        login_label: Some("Sign in with OpenRouter"),
        api_label: "openai-chat-completions",
        default_models_fn: default_models,
        extra_headers: None,
    })
}

fn default_models() -> Vec<crate::types::Model> {
    use crate::types::{Model, ModelCostRates, ModelThinkingLevel};
    vec![
        Model {
            id: "anthropic/claude-3.7-sonnet".into(),
            provider_id: "openrouter-oauth".into(),
            name: "Claude 3.7 Sonnet (OpenRouter OAuth)".into(),
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
            cost_rates: ModelCostRates::default(),
            context_window: Some(200_000),
            default_temperature: Some(1.0),
        },
        Model {
            id: "openai/gpt-4o".into(),
            provider_id: "openrouter-oauth".into(),
            name: "GPT-4o (OpenRouter OAuth)".into(),
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
