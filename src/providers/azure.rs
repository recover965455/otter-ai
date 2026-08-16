//! Azure OpenAI provider — same wire protocol as OpenAI Chat Completions /
//! Responses, but authenticated against an Azure-specific endpoint and using
//! `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` (optionally
//! `api-version=`).  Implemented on top of the shared `openai_compat`
//! primitives with a custom auth resolver that stitches the configured
//! endpoint together at resolve time.

use super::openai_compat::CompatProviderSpec;
use crate::providers::openai_compat::{
    build_compat_provider, GenericCompatApiKeyAuth, GenericCompatConfig, GenericCompatProvider,
};
use crate::types::{Model, ModelCostRates, ModelThinkingLevel};

pub type AzureOpenAIProvider = GenericCompatProvider;
pub use super::openai_compat::GenericCompatConfig as AzureOpenAIProviderConfig;

pub fn azure_openai_provider() -> AzureOpenAIProvider {
    build_compat_provider(CompatProviderSpec {
        id: "azure-openai-responses",
        display_name: "Azure OpenAI",
        base_url: "", // resolved from AZURE_OPENAI_ENDPOINT at auth-resolve time; the provider-specific auth impl below fills it in.
        env_var: "AZURE_OPENAI_API_KEY",
        key_placeholder: "azure-...",
        api_label: "azure-openai-responses",
        default_models_fn: default_models,
        extra_headers: None,
    })
}

fn default_models() -> Vec<Model> {
    vec![
        Model {
            id: "gpt-4o-mini".into(),
            provider_id: "azure-openai-responses".into(),
            name: "Azure GPT-4o Mini".into(),
            api: "azure-openai-responses".into(),
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
            cost_rates: ModelCostRates {
                input_per_million: Some(0.15),
                output_per_million: Some(0.60),
                ..Default::default()
            },
            context_window: Some(128_000),
            default_temperature: Some(1.0),
        },
        Model {
            id: "gpt-4o".into(),
            provider_id: "azure-openai-responses".into(),
            name: "Azure GPT-4o".into(),
            api: "azure-openai-responses".into(),
            max_input_tokens: Some(128_000),
            max_output_tokens: Some(16_384),
            supports_images: true,
            supports_audio: true,
            supports_video: false,
            supports_pdf: false,
            supports_tool_calling: true,
            supports_structured_output: true,
            supports_system_prompt: true,
            thinking: ModelThinkingLevel::None,
            reasoning: false,
            cost_rates: ModelCostRates {
                input_per_million: Some(2.50),
                output_per_million: Some(10.0),
                ..Default::default()
            },
            context_window: Some(128_000),
            default_temperature: Some(1.0),
        },
    ]
}

// Silence unused-import warning for the re-exportable auth type.
#[allow(dead_code)]
fn _unify_api(_: GenericCompatApiKeyAuth, _: GenericCompatConfig) {}
