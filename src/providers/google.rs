//! Google Generative AI provider (Gemini family).
//!
//! The underlying transport is Google's own `google-generative-ai` REST API
//! (not OpenAI-compatible), but from the SDK consumer's perspective this
//! provider is indistinguishable from the others: it plugs into the same
//! [`Provider`] trait, ships a default model catalogue, and reads the API
//! key from `GEMINI_API_KEY`.
//!
//! Live streaming is stubbed (matching the other providers in this build);
//! only the auth + metadata skeleton is populated here so callers can
//! register providers, resolve auth, and drive the rest of the
//! Models registry correctly.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::{Provider, RefreshModelsContext};
use crate::auth::types::{
    ApiKeyAuth, AuthContext, AuthInteraction, AuthResult, ModelAuth, ProviderAuth,
};
use crate::types::{
    ApiStreamOptions, AssistantMessageEvent, CancellationToken, Context, Message, Model,
    ModelCostRates, ModelThinkingLevel, ProviderHeaders,
};
use crate::utils::event_stream::AssistantMessageEventStream;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleGenerativeAiProviderConfig {
    pub base_url: String,
    pub env_var: String,
    pub default_models: Vec<Model>,
}

#[derive(Clone)]
struct GoogleGenAiAuth {
    env_var: String,
    base_url: String,
}

#[async_trait]
impl ApiKeyAuth for GoogleGenAiAuth {
    fn name(&self) -> &str {
        "Google Gemini API key"
    }

    async fn login(
        &self,
        interaction: &(dyn AuthInteraction + Send + Sync),
    ) -> anyhow::Result<crate::auth::types::ApiKeyCredential> {
        use crate::auth::types::AuthPrompt;
        let key = interaction
            .prompt(AuthPrompt::Secret {
                message: "Enter your Google Gemini API key:".into(),
                placeholder: Some("AIza...".into()),
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
        Some(AuthResult {
            auth: ModelAuth {
                api_key,
                headers: {
                    let mut h: ProviderHeaders = std::collections::HashMap::new();
                    h.insert("x-goog-api-client".into(), "otter-ai/0.1".into());
                    Some(h)
                },
                base_url: Some(self.base_url.clone()),
            },
            env: None,
            source,
        })
    }
}

struct AuthHolder {
    auth: ProviderAuth,
}

pub struct GoogleGenerativeAiProvider {
    id: String,
    name: String,
    config: GoogleGenerativeAiProviderConfig,
    auth_holder: Arc<AuthHolder>,
}

impl GoogleGenerativeAiProvider {
    pub fn new(id: &str, name: &str, config: GoogleGenerativeAiProviderConfig) -> Self {
        let auth = GoogleGenAiAuth {
            env_var: config.env_var.clone(),
            base_url: config.base_url.clone(),
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

pub fn google_generative_ai_provider() -> GoogleGenerativeAiProvider {
    let default_models = vec![
        Model {
            id: "gemini-2.5-pro".into(),
            provider_id: "google".into(),
            name: "Gemini 2.5 Pro".into(),
            api: "google-generative-ai".into(),
            max_input_tokens: Some(1_000_000),
            max_output_tokens: Some(65_536),
            supports_images: true,
            supports_audio: true,
            supports_video: true,
            supports_pdf: true,
            supports_tool_calling: true,
            supports_structured_output: true,
            supports_system_prompt: true,
            thinking: ModelThinkingLevel::None,
            reasoning: false,
            cost_rates: ModelCostRates {
                input_per_million: Some(1.25),
                output_per_million: Some(5.00),
                ..Default::default()
            },
            context_window: Some(1_000_000),
            default_temperature: Some(1.0),
            thinking_level_map: None,
        },
        Model {
            id: "gemini-2.0-flash".into(),
            provider_id: "google".into(),
            name: "Gemini 2.0 Flash".into(),
            api: "google-generative-ai".into(),
            max_input_tokens: Some(1_000_000),
            max_output_tokens: Some(65_536),
            supports_images: true,
            supports_audio: true,
            supports_video: true,
            supports_pdf: true,
            supports_tool_calling: true,
            supports_structured_output: true,
            supports_system_prompt: true,
            thinking: ModelThinkingLevel::None,
            reasoning: false,
            cost_rates: ModelCostRates {
                input_per_million: Some(0.10),
                output_per_million: Some(0.40),
                ..Default::default()
            },
            context_window: Some(1_000_000),
            default_temperature: Some(1.0),
            thinking_level_map: None,
        },
    ];
    GoogleGenerativeAiProvider::new(
        "google",
        "Google Generative AI",
        GoogleGenerativeAiProviderConfig {
            base_url: "https://generativelanguage.googleapis.com/v1beta".into(),
            env_var: "GEMINI_API_KEY".into(),
            default_models,
        },
    )
}

#[async_trait]
impl Provider for GoogleGenerativeAiProvider {
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
            .with_error_message("Google Generative AI live adapter not available in this build");
        es.push(AssistantMessageEvent::Error {
            reason: "error".into(),
            error: "Google Generative AI live adapter not available in this build".into(),
        });
        es.end(Some(msg));
        es
    }
}
