//! GitHub Copilot — subscription-based OAuth provider using GitHub's
//! device-code flow.  Press Enter for github.com, or enter your GitHub
//! Enterprise Server domain during login.

use super::oauth_compat::{build_oauth_provider, GenericOAuthProvider, OAuthProviderSpec};

pub type GitHubCopilotProvider = GenericOAuthProvider;

pub fn github_copilot_provider() -> GitHubCopilotProvider {
    build_oauth_provider(OAuthProviderSpec {
        id: "github-copilot",
        display_name: "GitHub Copilot",
        base_url: "https://api.githubcopilot.com",
        // GitHub Copilot VS Code extension client_id (public, same as pi-ai uses).
        client_id: "Iv1.b507a3c57e6377c4",
        scopes: &["read:user"],
        auth_url: None, // device-code flow, not browser PKCE.
        token_url: Some("https://github.com/login/oauth/access_token"),
        device_auth_url: Some("https://github.com/login/device/code"),
        redirect_uri: None,
        is_subscription: true,
        login_label: Some("GitHub Copilot"),
        api_label: "openai-chat-completions",
        default_models_fn: default_models,
        extra_headers: None,
    })
}

fn default_models() -> Vec<crate::types::Model> {
    use crate::types::{Model, ModelCostRates, ModelThinkingLevel};
    vec![
        Model {
            id: "gpt-4o".into(),
            provider_id: "github-copilot".into(),
            name: "GPT-4o (Copilot)".into(),
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
        Model {
            id: "claude-3.7-sonnet".into(),
            provider_id: "github-copilot".into(),
            name: "Claude 3.7 Sonnet (Copilot)".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(200_000),
            max_output_tokens: Some(64_000),
            supports_images: true,
            supports_audio: false,
            supports_video: false,
            supports_pdf: false,
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
            id: "gemini-2.0-flash".into(),
            provider_id: "github-copilot".into(),
            name: "Gemini 2.0 Flash (Copilot)".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(1_000_000),
            max_output_tokens: Some(65_536),
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
            id: "o3-mini".into(),
            provider_id: "github-copilot".into(),
            name: "o3-mini (Copilot)".into(),
            api: "openai-chat-completions".into(),
            max_input_tokens: Some(200_000),
            max_output_tokens: Some(100_000),
            supports_images: false,
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
