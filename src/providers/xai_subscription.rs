//! xAI subscription (Grok/X subscription) — OAuth device-code flow.
//! `XAI_API_KEY` remains available through the API-key xAI provider for
//! non-subscription usage.

use super::oauth_compat::{build_oauth_provider, GenericOAuthProvider, OAuthProviderSpec};

pub type XaiSubscriptionProvider = GenericOAuthProvider;

pub fn xai_subscription_provider() -> XaiSubscriptionProvider {
    build_oauth_provider(OAuthProviderSpec {
        id: "xai-subscription",
        display_name: "xAI (Grok/X subscription)",
        base_url: "https://api.x.ai/v1",
        // TODO: verify client_id from pi-ai source — xAI device OAuth.
        client_id: "xai-cli",
        scopes: &["openid", "profile", "offline_access"],
        auth_url: None, // device-code flow
        token_url: Some("https://xai.com/oauth/token"),
        device_auth_url: Some("https://xai.com/oauth/device/code"),
        redirect_uri: None,
        is_subscription: true,
        login_label: Some("xAI subscription"),
        api_label: "openai-chat-completions",
        default_models_fn: default_models,
        extra_headers: None,
    })
}

fn default_models() -> Vec<crate::types::Model> {
    use crate::types::{Model, ModelCostRates, ModelThinkingLevel};
    vec![
        Model {
            id: "grok-4".into(),
            provider_id: "xai-subscription".into(),
            name: "Grok 4 (subscription)".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(1_048_576),
            max_output_tokens: Some(131_072),
            supports_images: false,
            supports_audio: false,
            supports_video: false,
            supports_pdf: false,
            supports_tool_calling: true,
            supports_structured_output: true,
            supports_system_prompt: true,
            thinking: ModelThinkingLevel::Medium,
            reasoning: true,
            cost_rates: ModelCostRates::default(),
            context_window: Some(1_048_576),
            default_temperature: Some(1.0),
            thinking_level_map: None,
        },
        Model {
            id: "grok-3".into(),
            provider_id: "xai-subscription".into(),
            name: "Grok 3 (subscription)".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(131_072),
            max_output_tokens: Some(16_384),
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
            context_window: Some(131_072),
            default_temperature: Some(1.0),
            thinking_level_map: None,
        },
    ]
}
