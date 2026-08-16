//! Claude Pro/Max — subscription-based OAuth provider.
//! Third-party harness usage draws from [extra usage](https://claude.ai/settings/usage)
//! and is billed per token, not against Claude plan limits.

use super::oauth_compat::{build_oauth_provider, GenericOAuthProvider, OAuthProviderSpec};

pub type ClaudeProMaxProvider = GenericOAuthProvider;

pub fn claude_pro_max_provider() -> ClaudeProMaxProvider {
    build_oauth_provider(OAuthProviderSpec {
        id: "claude-pro-max",
        display_name: "Claude Pro/Max",
        base_url: "https://api.anthropic.com/v1",
        // TODO: verify client_id from pi-ai source — Anthropic subscription OAuth.
        client_id: "9d1c250a-e61b-44a4-8edc-1e6cb45e0f56",
        scopes: &["org:create_api_key"],
        auth_url: Some("https://claude.ai/oauth/authorize"),
        token_url: Some("https://claude.ai/oauth/token"),
        device_auth_url: None,
        redirect_uri: Some("https://console.anthropic.com/oauth/code/callback"),
        is_subscription: true,
        login_label: Some("Claude Pro/Max"),
        api_label: "anthropic-messages",
        default_models_fn: default_models,
        extra_headers: None,
    })
}

fn default_models() -> Vec<crate::types::Model> {
    use crate::types::{Model, ModelCostRates, ModelThinkingLevel};
    vec![
        Model {
            id: "claude-sonnet-4-20250514".into(),
            provider_id: "claude-pro-max".into(),
            name: "Claude Sonnet 4 (Pro/Max)".into(),
            api: "anthropic-messages".into(),
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
            thinking_level_map: None,
        },
        Model {
            id: "claude-opus-4-20250514".into(),
            provider_id: "claude-pro-max".into(),
            name: "Claude Opus 4 (Pro/Max)".into(),
            api: "anthropic-messages".into(),
            max_input_tokens: Some(200_000),
            max_output_tokens: Some(32_000),
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
            thinking_level_map: None,
        },
        Model {
            id: "claude-3-7-sonnet-20250219".into(),
            provider_id: "claude-pro-max".into(),
            name: "Claude 3.7 Sonnet (Pro/Max)".into(),
            api: "anthropic-messages".into(),
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
            thinking_level_map: None,
        },
    ]
}
