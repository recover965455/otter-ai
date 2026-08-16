//! ChatGPT Plus/Pro (Codex) — subscription-based OAuth provider.
//! Requires a ChatGPT Plus or Pro subscription. Officially endorsed by
//! OpenAI under the [Codex for OSS](https://developers.openai.com/community/codex-for-oss) program.

use super::oauth_compat::{build_oauth_provider, GenericOAuthProvider, OAuthProviderSpec};

pub type ChatGptPlusProvider = GenericOAuthProvider;

pub fn chatgpt_plus_provider() -> ChatGptPlusProvider {
    build_oauth_provider(OAuthProviderSpec {
        id: "chatgpt-plus",
        display_name: "ChatGPT Plus/Pro (Codex)",
        base_url: "https://api.openai.com/v1",
        // TODO: verify client_id from pi-ai source — OpenAI Codex OAuth.
        client_id: "app_EMoamEEZ73f0CkXaXp7hrann",
        scopes: &["openid", "profile", "email", "offline_access"],
        auth_url: Some("https://auth.openai.com/authorize"),
        token_url: Some("https://auth.openai.com/oauth/token"),
        device_auth_url: None,
        redirect_uri: Some("http://localhost:1455/callback"),
        is_subscription: true,
        login_label: Some("ChatGPT Plus/Pro"),
        api_label: "openai-codex-responses",
        default_models_fn: default_models,
        extra_headers: None,
    })
}

fn default_models() -> Vec<crate::types::Model> {
    use crate::types::{Model, ModelCostRates, ModelThinkingLevel};
    vec![
        Model {
            id: "gpt-4.1".into(),
            provider_id: "chatgpt-plus".into(),
            name: "GPT-4.1 (Codex)".into(),
            api: "openai-codex-responses".into(),
            max_input_tokens: Some(1_000_000),
            max_output_tokens: Some(32_768),
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
            context_window: Some(1_000_000),
            default_temperature: Some(1.0),
        },
        Model {
            id: "o3".into(),
            provider_id: "chatgpt-plus".into(),
            name: "o3 (Codex)".into(),
            api: "openai-codex-responses".into(),
            max_input_tokens: Some(200_000),
            max_output_tokens: Some(100_000),
            supports_images: true,
            supports_audio: false,
            supports_video: false,
            supports_pdf: false,
            supports_tool_calling: true,
            supports_structured_output: true,
            supports_system_prompt: false,
            thinking: ModelThinkingLevel::High,
            reasoning: true,
            cost_rates: ModelCostRates::default(),
            context_window: Some(200_000),
            default_temperature: Some(1.0),
        },
        Model {
            id: "o4-mini".into(),
            provider_id: "chatgpt-plus".into(),
            name: "o4-mini (Codex)".into(),
            api: "openai-codex-responses".into(),
            max_input_tokens: Some(200_000),
            max_output_tokens: Some(100_000),
            supports_images: true,
            supports_audio: false,
            supports_video: false,
            supports_pdf: false,
            supports_tool_calling: true,
            supports_structured_output: true,
            supports_system_prompt: false,
            thinking: ModelThinkingLevel::Medium,
            reasoning: true,
            cost_rates: ModelCostRates::default(),
            context_window: Some(200_000),
            default_temperature: Some(1.0),
        },
    ]
}
