//! ChatGPT Plus/Pro (Codex) — subscription-based OAuth provider.
//! Requires a ChatGPT Plus or Pro subscription. Officially endorsed by
//! OpenAI under the [Codex for OSS](https://developers.openai.com/community/codex-for-oss) program.
//!
//! Wire protocol: `openai-codex-responses` against
//! `https://chatgpt.com/backend-api/codex/responses` (see
//! [`crate::providers::openai_responses`]). The `client_id` below matches
//! pi-ai's `loadOpenAICodexOAuth` (`auth/oauth/openai-codex.ts`).

use super::oauth_compat::{build_oauth_provider, GenericOAuthProvider, OAuthProviderSpec};

pub type ChatGptPlusProvider = GenericOAuthProvider;

pub fn chatgpt_plus_provider() -> ChatGptPlusProvider {
    build_oauth_provider(OAuthProviderSpec {
        id: "chatgpt-plus",
        display_name: "ChatGPT Plus/Pro (Codex)",
        base_url: "https://chatgpt.com/backend-api",
        // Verified against pi-ai `auth/oauth/openai-codex.ts` (CLIENT_ID).
        client_id: "app_EMoamEEZ73f0CkXaXp7hrann",
        scopes: &["openid", "profile", "email", "offline_access"],
        auth_url: Some("https://auth.openai.com/oauth/authorize"),
        token_url: Some("https://auth.openai.com/oauth/token"),
        device_auth_url: None,
        redirect_uri: Some("http://localhost:1455/auth/callback"),
        is_subscription: true,
        login_label: Some("ChatGPT Plus/Pro"),
        api_label: "openai-codex-responses",
        default_models_fn: default_models,
        extra_headers: None,
    })
}

fn map(entries: &[(&str, &str)]) -> Option<std::collections::HashMap<String, String>> {
    Some(
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
    )
}

/// Model catalogue aligned with pi-ai `providers/data/openai-codex.json`
/// (generated from the same upstream source).
fn default_models() -> Vec<crate::types::Model> {
    use crate::types::{CostTier, Model, ModelCostRates, ModelThinkingLevel};

    let codex_map = map(&[("xhigh", "xhigh"), ("minimal", "low")]);
    let codex_map_max = map(&[("xhigh", "xhigh"), ("max", "max"), ("minimal", "low")]);

    let spark = Model {
        id: "gpt-5.3-codex-spark".into(),
        provider_id: "chatgpt-plus".into(),
        name: "GPT-5.3 Codex Spark".into(),
        api: "openai-codex-responses".into(),
        max_input_tokens: Some(128_000),
        max_output_tokens: Some(128_000),
        supports_images: false,
        supports_audio: false,
        supports_video: false,
        supports_pdf: false,
        supports_tool_calling: true,
        supports_structured_output: true,
        supports_system_prompt: true,
        thinking: ModelThinkingLevel::High,
        reasoning: true,
        cost_rates: ModelCostRates {
            input_per_million: Some(1.75),
            output_per_million: Some(14.0),
            input_cache_read_per_million: Some(0.175),
            input_cache_write_per_million: Some(0.0),
            tiers: vec![],
        },
        context_window: Some(128_000),
        default_temperature: Some(1.0),
        thinking_level_map: codex_map.clone(),
    };

    let gpt54 = Model {
        id: "gpt-5.4".into(),
        provider_id: "chatgpt-plus".into(),
        name: "GPT-5.4".into(),
        api: "openai-codex-responses".into(),
        max_input_tokens: Some(272_000),
        max_output_tokens: Some(128_000),
        supports_images: true,
        supports_audio: false,
        supports_video: false,
        supports_pdf: false,
        supports_tool_calling: true,
        supports_structured_output: true,
        supports_system_prompt: true,
        thinking: ModelThinkingLevel::High,
        reasoning: true,
        cost_rates: ModelCostRates {
            input_per_million: Some(2.5),
            output_per_million: Some(15.0),
            input_cache_read_per_million: Some(0.25),
            input_cache_write_per_million: Some(0.0),
            tiers: vec![CostTier {
                input_tokens_above: 272_000,
                input_per_million: 5.0,
                output_per_million: 22.5,
                cache_read_per_million: 0.5,
                cache_write_per_million: 0.0,
            }],
        },
        context_window: Some(272_000),
        default_temperature: Some(1.0),
        thinking_level_map: codex_map.clone(),
    };

    let gpt54_mini = Model {
        id: "gpt-5.4-mini".into(),
        provider_id: "chatgpt-plus".into(),
        name: "GPT-5.4 mini".into(),
        api: "openai-codex-responses".into(),
        max_input_tokens: Some(272_000),
        max_output_tokens: Some(128_000),
        supports_images: true,
        supports_audio: false,
        supports_video: false,
        supports_pdf: false,
        supports_tool_calling: true,
        supports_structured_output: true,
        supports_system_prompt: true,
        thinking: ModelThinkingLevel::High,
        reasoning: true,
        cost_rates: ModelCostRates {
            input_per_million: Some(0.75),
            output_per_million: Some(4.5),
            input_cache_read_per_million: Some(0.075),
            input_cache_write_per_million: Some(0.0),
            tiers: vec![],
        },
        context_window: Some(272_000),
        default_temperature: Some(1.0),
        thinking_level_map: codex_map.clone(),
    };

    let gpt55 = Model {
        id: "gpt-5.5".into(),
        provider_id: "chatgpt-plus".into(),
        name: "GPT-5.5".into(),
        api: "openai-codex-responses".into(),
        max_input_tokens: Some(272_000),
        max_output_tokens: Some(128_000),
        supports_images: true,
        supports_audio: false,
        supports_video: false,
        supports_pdf: false,
        supports_tool_calling: true,
        supports_structured_output: true,
        supports_system_prompt: true,
        thinking: ModelThinkingLevel::High,
        reasoning: true,
        cost_rates: ModelCostRates {
            input_per_million: Some(5.0),
            output_per_million: Some(30.0),
            input_cache_read_per_million: Some(0.5),
            input_cache_write_per_million: Some(0.0),
            tiers: vec![CostTier {
                input_tokens_above: 272_000,
                input_per_million: 10.0,
                output_per_million: 45.0,
                cache_read_per_million: 1.0,
                cache_write_per_million: 0.0,
            }],
        },
        context_window: Some(272_000),
        default_temperature: Some(1.0),
        thinking_level_map: codex_map.clone(),
    };

    let luna = Model {
        id: "gpt-5.6-luna".into(),
        provider_id: "chatgpt-plus".into(),
        name: "GPT-5.6 Luna".into(),
        api: "openai-codex-responses".into(),
        max_input_tokens: Some(272_000),
        max_output_tokens: Some(128_000),
        supports_images: true,
        supports_audio: false,
        supports_video: false,
        supports_pdf: false,
        supports_tool_calling: true,
        supports_structured_output: true,
        supports_system_prompt: true,
        thinking: ModelThinkingLevel::High,
        reasoning: true,
        cost_rates: ModelCostRates {
            input_per_million: Some(0.2),
            output_per_million: Some(1.2),
            input_cache_read_per_million: Some(0.02),
            input_cache_write_per_million: Some(0.25),
            tiers: vec![CostTier {
                input_tokens_above: 272_000,
                input_per_million: 0.4,
                output_per_million: 1.8,
                cache_read_per_million: 0.04,
                cache_write_per_million: 0.5,
            }],
        },
        context_window: Some(272_000),
        default_temperature: Some(1.0),
        thinking_level_map: codex_map_max.clone(),
    };

    let sol = Model {
        id: "gpt-5.6-sol".into(),
        provider_id: "chatgpt-plus".into(),
        name: "GPT-5.6 Sol".into(),
        api: "openai-codex-responses".into(),
        max_input_tokens: Some(272_000),
        max_output_tokens: Some(128_000),
        supports_images: true,
        supports_audio: false,
        supports_video: false,
        supports_pdf: false,
        supports_tool_calling: true,
        supports_structured_output: true,
        supports_system_prompt: true,
        thinking: ModelThinkingLevel::High,
        reasoning: true,
        cost_rates: ModelCostRates {
            input_per_million: Some(5.0),
            output_per_million: Some(30.0),
            input_cache_read_per_million: Some(0.5),
            input_cache_write_per_million: Some(6.25),
            tiers: vec![CostTier {
                input_tokens_above: 272_000,
                input_per_million: 10.0,
                output_per_million: 45.0,
                cache_read_per_million: 1.0,
                cache_write_per_million: 12.5,
            }],
        },
        context_window: Some(272_000),
        default_temperature: Some(1.0),
        thinking_level_map: codex_map_max.clone(),
    };

    let terra = Model {
        id: "gpt-5.6-terra".into(),
        provider_id: "chatgpt-plus".into(),
        name: "GPT-5.6 Terra".into(),
        api: "openai-codex-responses".into(),
        max_input_tokens: Some(272_000),
        max_output_tokens: Some(128_000),
        supports_images: true,
        supports_audio: false,
        supports_video: false,
        supports_pdf: false,
        supports_tool_calling: true,
        supports_structured_output: true,
        supports_system_prompt: true,
        thinking: ModelThinkingLevel::High,
        reasoning: true,
        cost_rates: ModelCostRates {
            input_per_million: Some(2.0),
            output_per_million: Some(12.0),
            input_cache_read_per_million: Some(0.2),
            input_cache_write_per_million: Some(2.5),
            tiers: vec![CostTier {
                input_tokens_above: 272_000,
                input_per_million: 4.0,
                output_per_million: 18.0,
                cache_read_per_million: 0.4,
                cache_write_per_million: 5.0,
            }],
        },
        context_window: Some(272_000),
        default_temperature: Some(1.0),
        thinking_level_map: codex_map_max,
    };

    vec![spark, gpt54, gpt54_mini, gpt55, luna, sol, terra]
}
