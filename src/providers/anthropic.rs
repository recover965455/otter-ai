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
pub struct AnthropicProviderConfig {
    pub base_url: String,
    pub env_var: String,
    pub default_models: Vec<Model>,
}

#[derive(Clone)]
struct AnthropicApiKeyAuth {
    env_var: String,
    base_url: String,
    name: String,
}

#[async_trait]
impl ApiKeyAuth for AnthropicApiKeyAuth {
    fn name(&self) -> &str {
        &self.name
    }

    async fn login(
        &self,
        interaction: &(dyn AuthInteraction + Send + Sync),
    ) -> anyhow::Result<crate::auth::types::ApiKeyCredential> {
        use crate::auth::types::AuthPrompt;
        let key = interaction
            .prompt(AuthPrompt::Secret {
                message: format!("Enter your {} API key:", self.name),
                placeholder: Some("sk-ant-...".to_string()),
                signal: interaction.signal().cloned(),
            })
            .await?;
        Ok(crate::auth::types::ApiKeyCredential {
            r#type: "api_key".to_string(),
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
                    .map(|_| "credential_store".to_string())
                    .or(Some(format!("env:{}", self.env_var))),
            ),
            None => (None, None),
        };
        Some(AuthResult {
            auth: ModelAuth {
                api_key,
                headers: Some(
                    vec![("anthropic-version".to_string(), "2023-06-01".to_string())]
                        .into_iter()
                        .collect(),
                ),
                base_url: Some(self.base_url.clone()),
            },
            env: None,
            source,
        })
    }
}

struct AnthropicAuthHolder {
    auth: ProviderAuth,
}

pub struct AnthropicProvider {
    id: String,
    name: String,
    config: AnthropicProviderConfig,
    auth_holder: Arc<AnthropicAuthHolder>,
}

impl AnthropicProvider {
    pub fn new(id: &str, name: &str, config: AnthropicProviderConfig) -> Self {
        let auth_impl = AnthropicApiKeyAuth {
            env_var: config.env_var.clone(),
            base_url: config.base_url.clone(),
            name: format!("{} API key", name),
        };
        Self {
            id: id.to_string(),
            name: name.to_string(),
            config,
            auth_holder: Arc::new(AnthropicAuthHolder {
                auth: ProviderAuth {
                    api_key: Some(Box::new(auth_impl)),
                    oauth: None,
                },
            }),
        }
    }
}

pub fn anthropic_provider() -> AnthropicProvider {
    let default_models = vec![
        Model {
            id: "claude-3-5-sonnet-latest".to_string(),
            provider_id: "anthropic".to_string(),
            name: "Claude 3.5 Sonnet".to_string(),
            api: "anthropic-messages".to_string(),
            max_input_tokens: Some(200000),
            max_output_tokens: Some(8192),
            supports_images: true,
            supports_audio: false,
            supports_video: false,
            supports_pdf: true,
            supports_tool_calling: true,
            supports_structured_output: true,
            supports_system_prompt: true,
            thinking: ModelThinkingLevel::None,
            reasoning: false,
            cost_rates: ModelCostRates {
                input_per_million: Some(3.0),
                output_per_million: Some(15.0),
                input_cache_read_per_million: Some(0.30),
                input_cache_write_per_million: Some(3.75),
                tiers: vec![],
            },
            context_window: Some(200000),
            default_temperature: Some(1.0),
        },
        Model {
            id: "claude-3-opus-latest".to_string(),
            provider_id: "anthropic".to_string(),
            name: "Claude 3 Opus".to_string(),
            api: "anthropic-messages".to_string(),
            max_input_tokens: Some(200000),
            max_output_tokens: Some(8192),
            supports_images: true,
            supports_audio: false,
            supports_video: false,
            supports_pdf: true,
            supports_tool_calling: true,
            supports_structured_output: true,
            supports_system_prompt: true,
            thinking: ModelThinkingLevel::None,
            reasoning: false,
            cost_rates: ModelCostRates {
                input_per_million: Some(15.0),
                output_per_million: Some(75.0),
                input_cache_read_per_million: Some(1.50),
                input_cache_write_per_million: Some(18.75),
                tiers: vec![],
            },
            context_window: Some(200000),
            default_temperature: Some(1.0),
        },
        Model {
            id: "claude-3-7-sonnet-latest".to_string(),
            provider_id: "anthropic".to_string(),
            name: "Claude 3.7 Sonnet".to_string(),
            api: "anthropic-messages".to_string(),
            max_input_tokens: Some(200000),
            max_output_tokens: Some(64000),
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
                input_cache_read_per_million: Some(0.30),
                input_cache_write_per_million: Some(3.75),
                tiers: vec![],
            },
            context_window: Some(200000),
            default_temperature: Some(1.0),
        },
    ];
    AnthropicProvider::new(
        "anthropic",
        "Anthropic",
        AnthropicProviderConfig {
            base_url: "https://api.anthropic.com".to_string(),
            env_var: "ANTHROPIC_API_KEY".to_string(),
            default_models,
        },
    )
}

#[async_trait]
impl Provider for AnthropicProvider {
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
        let msg = Message::assistant_default("anthropic".into(), "anthropic".into())
            .with_error_message("Anthropic live adapter not available in this build");
        es.push(AssistantMessageEvent::Error {
            reason: "error".into(),
            error: "Anthropic live adapter not available in this build".into(),
        });
        es.end(Some(msg));
        es
    }
}
