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
pub struct OpenAIProviderConfig {
    pub base_url: String,
    pub env_var: String,
    pub default_models: Vec<Model>,
}

#[derive(Clone)]
struct OpenAIApiKeyAuth {
    env_var: String,
    base_url: String,
    name: String,
}

#[async_trait]
impl ApiKeyAuth for OpenAIApiKeyAuth {
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
                placeholder: Some("sk-...".to_string()),
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
                headers: None,
                base_url: Some(self.base_url.clone()),
            },
            env: None,
            source,
        })
    }
}

struct OpenAIAuthHolder {
    auth: ProviderAuth,
}

pub struct OpenAIProvider {
    id: String,
    name: String,
    config: OpenAIProviderConfig,
    auth_holder: Arc<OpenAIAuthHolder>,
}

impl OpenAIProvider {
    pub fn new(id: &str, name: &str, config: OpenAIProviderConfig) -> Self {
        let auth_impl = OpenAIApiKeyAuth {
            env_var: config.env_var.clone(),
            base_url: config.base_url.clone(),
            name: format!("{} API key", name),
        };
        Self {
            id: id.to_string(),
            name: name.to_string(),
            config,
            auth_holder: Arc::new(OpenAIAuthHolder {
                auth: ProviderAuth {
                    api_key: Some(Box::new(auth_impl)),
                    oauth: None,
                },
            }),
        }
    }
}

pub fn openai_provider() -> OpenAIProvider {
    let default_models = vec![
        Model {
            id: "gpt-4o-mini".to_string(),
            provider_id: "openai".to_string(),
            name: "GPT-4o Mini".to_string(),
            api: "openai-chat-completions".to_string(),
            max_input_tokens: Some(128000),
            max_output_tokens: Some(16384),
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
                input_per_million: Some(0.15),
                output_per_million: Some(0.60),
                ..Default::default()
            },
            context_window: Some(128000),
            default_temperature: Some(1.0),
        },
        Model {
            id: "gpt-4o".to_string(),
            provider_id: "openai".to_string(),
            name: "GPT-4o".to_string(),
            api: "openai-chat-completions".to_string(),
            max_input_tokens: Some(128000),
            max_output_tokens: Some(16384),
            supports_images: true,
            supports_audio: true,
            supports_video: true,
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
            context_window: Some(128000),
            default_temperature: Some(1.0),
        },
    ];
    OpenAIProvider::new(
        "openai",
        "OpenAI",
        OpenAIProviderConfig {
            base_url: "https://api.openai.com/v1".to_string(),
            env_var: "OPENAI_API_KEY".to_string(),
            default_models,
        },
    )
}

#[async_trait]
impl Provider for OpenAIProvider {
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
        let msg = Message::assistant_default("openai".into(), "openai".into())
            .with_error_message("OpenAI live adapter not available in this build");
        es.push(AssistantMessageEvent::Error {
            reason: "error".into(),
            error: "OpenAI live adapter not available in this build".into(),
        });
        es.end(Some(msg));
        es
    }
}
