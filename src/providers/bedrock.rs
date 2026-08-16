//! Amazon Bedrock ConverseStream provider.
//!
//! Bedrock differs from the other providers in two ways:
//! 1. Auth uses **AWS SigV4** signed requests (not an `Authorization: Bearer`
//!    API key).  We still expose this via the `ApiKeyAuth` trait but encode
//!    the credentials as an opaque `AWS_BEARER_TOKEN_BEDROCK` — matching the
//!    convention `@earendil-works/pi-ai` uses for the same feature — and
//!    delegate the full SigV4 dance to a later shared network adapter.
//! 2. The wire protocol is Bedrock's `converse-stream` RPC, not OpenAI
//!    chat-completions.
//!
//! Everything else (model catalogue, trait impl shape, registration helper)
//! matches the other providers so users get a consistent surface area.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::{Provider, RefreshModelsContext};
use crate::auth::types::{
    ApiKeyAuth, AuthContext, AuthInteraction, AuthResult, ModelAuth, ProviderAuth,
};
use crate::types::{
    ApiStreamOptions, AssistantMessageEvent, CancellationToken, Context, Message, Model,
    ModelCostRates, ModelThinkingLevel,
};
use crate::utils::event_stream::AssistantMessageEventStream;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockProviderConfig {
    pub region: String,
    pub env_var: String,
    pub default_models: Vec<Model>,
}

#[derive(Clone)]
struct BedrockAuth {
    env_var: String,
    region: String,
}

#[async_trait]
impl ApiKeyAuth for BedrockAuth {
    fn name(&self) -> &str {
        "Amazon Bedrock Bearer Token"
    }

    async fn login(
        &self,
        interaction: &(dyn AuthInteraction + Send + Sync),
    ) -> anyhow::Result<crate::auth::types::ApiKeyCredential> {
        use crate::auth::types::AuthPrompt;
        let key = interaction
            .prompt(AuthPrompt::Secret {
                message: format!(
                    "Enter your AWS_BEARER_TOKEN_BEDROCK (region {}):",
                    self.region
                ),
                placeholder: Some("eyJ... (SigV4-minted bearer)".into()),
                signal: interaction.signal().cloned(),
            })
            .await?;
        Ok(crate::auth::types::ApiKeyCredential {
            key: Some(key),
            env: None,
        })
    }

    async fn resolve(
        &self,
        ctx: &(dyn AuthContext + Send + Sync),
        credential: Option<&crate::auth::types::ApiKeyCredential>,
        _signal: &CancellationToken,
    ) -> Option<AuthResult> {
        let env_key = ctx.env(&self.env_var).await;
        let key = credential.and_then(|c| c.key.clone()).or(env_key);
        let (api_key, source) = match key {
            Some(k) => (
                Some(k),
                credential
                    .map(|_| "credential_store".into())
                    .or(Some(format!("env:{}", self.env_var))),
            ),
            None => (None, None),
        };
        let base_url = format!("https://bedrock-runtime.{}.amazonaws.com", self.region);
        Some(AuthResult {
            auth: ModelAuth {
                api_key,
                headers: None,
                base_url: Some(base_url),
            },
            env: None,
            source,
        })
    }
}

struct AuthHolder {
    auth: ProviderAuth,
}

pub struct BedrockProvider {
    id: String,
    name: String,
    config: BedrockProviderConfig,
    auth_holder: Arc<AuthHolder>,
}

impl BedrockProvider {
    pub fn new(id: &str, name: &str, config: BedrockProviderConfig) -> Self {
        let auth = BedrockAuth {
            env_var: config.env_var.clone(),
            region: config.region.clone(),
        };
        Self {
            id: id.into(),
            name: name.into(),
            config,
            auth_holder: Arc::new(AuthHolder {
                auth: ProviderAuth {
                    api_key: Some(Box::new(auth)),
                    oauth: None,
                },
            }),
        }
    }
}

pub fn bedrock_provider() -> BedrockProvider {
    let default_models = vec![
        Model {
            id: "anthropic.claude-3-7-sonnet-20250219-v1:0".into(),
            provider_id: "amazon-bedrock".into(),
            name: "Claude 3.7 Sonnet (Bedrock)".into(),
            api: "bedrock-converse-stream".into(),
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
            cost_rates: ModelCostRates {
                input_per_million: Some(3.0),
                output_per_million: Some(15.0),
                ..Default::default()
            },
            context_window: Some(200_000),
            default_temperature: Some(1.0),
            thinking_level_map: None,
        },
        Model {
            id: "meta.llama3-2-70b-instruct-v1:0".into(),
            provider_id: "amazon-bedrock".into(),
            name: "Llama 3.2 70B Instruct (Bedrock)".into(),
            api: "bedrock-converse-stream".into(),
            max_input_tokens: Some(128_000),
            max_output_tokens: Some(8_192),
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
            context_window: Some(128_000),
            default_temperature: Some(1.0),
            thinking_level_map: None,
        },
    ];
    BedrockProvider::new(
        "amazon-bedrock",
        "Amazon Bedrock",
        BedrockProviderConfig {
            region: "us-east-1".into(),
            env_var: "AWS_BEARER_TOKEN_BEDROCK".into(),
            default_models,
        },
    )
}

#[async_trait]
impl Provider for BedrockProvider {
    fn id(&self) -> &str {
        &self.id
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn auth(&self) -> &ProviderAuth {
        &self.auth_holder.auth
    }
    fn get_models(&self) -> Vec<Model> {
        self.config.default_models.clone()
    }
    async fn refresh_models(
        &self,
        _cx: Box<dyn RefreshModelsContext + Send + 'static>,
    ) -> Result<(), String> {
        Ok(())
    }
    fn stream(
        &self,
        _model: &Model,
        _context: Context,
        _options: ApiStreamOptions,
    ) -> AssistantMessageEventStream {
        let es = crate::utils::event_stream::create_assistant_message_event_stream();
        let msg = Message::assistant_default(self.id.clone(), self.id.clone())
            .with_error_message("Bedrock live adapter not available in this build");
        es.push(AssistantMessageEvent::Error {
            reason: "error".into(),
            error: "Bedrock live adapter not available in this build".into(),
        });
        es.end(Some(msg));
        es
    }
}
