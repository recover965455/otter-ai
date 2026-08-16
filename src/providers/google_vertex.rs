//! Google Vertex AI — enterprise-grade Google Cloud LLM platform.
//! Uses the same model family as Google Generative AI (Gemini) but routes
//! through a regional Vertex AI endpoint and authenticates with Google
//! Cloud service-account credentials (`GOOGLE_APPLICATION_CREDENTIALS`)
//! or `gcloud auth print-access-token`.
//!
//! Wire protocol: `google-vertex-ai` (same shape as `google-generative-ai`
//! but with a different URL pattern).

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
pub struct GoogleVertexProviderConfig {
    pub region: String,
    pub project_id: String,
    pub env_var: String,
    pub default_models: Vec<Model>,
}

#[derive(Clone)]
struct GoogleVertexAuth {
    env_var: String,
    region: String,
    project_id: String,
}

#[async_trait]
impl ApiKeyAuth for GoogleVertexAuth {
    fn name(&self) -> &str {
        "Google Vertex AI access token"
    }

    async fn login(
        &self,
        interaction: &(dyn AuthInteraction + Send + Sync),
    ) -> anyhow::Result<crate::auth::types::ApiKeyCredential> {
        use crate::auth::types::AuthPrompt;
        let key = interaction
            .prompt(AuthPrompt::Secret {
                message: format!(
                    "Paste a Google Cloud access token (gcloud auth print-access-token) for project {}:",
                    self.project_id
                ),
                placeholder: Some("ya29...".into()),
                signal: interaction.signal().cloned(),
            })
            .await?;
        Ok(crate::auth::types::ApiKeyCredential {
            r#type: "api_key".into(),
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
        // Vertex AI resolves the access token from:
        // 1. credential store (stored OAuth token)
        // 2. GOOGLE_APPLICATION_CREDENTIALS env (service-account JSON → token exchange TBD)
        // 3. explicit env var override
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
        let base_url = format!("https://{}-aiplatform.googleapis.com/v1", self.region);
        let mut headers: ProviderHeaders = std::collections::HashMap::new();
        headers.insert("x-goog-api-client".into(), "otter-ai/0.1".into());
        Some(AuthResult {
            auth: ModelAuth {
                api_key,
                headers: Some(headers),
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

pub struct GoogleVertexProvider {
    id: String,
    name: String,
    config: GoogleVertexProviderConfig,
    auth_holder: Arc<AuthHolder>,
}

impl GoogleVertexProvider {
    pub fn new(id: &str, name: &str, config: GoogleVertexProviderConfig) -> Self {
        let auth = GoogleVertexAuth {
            env_var: config.env_var.clone(),
            region: config.region.clone(),
            project_id: config.project_id.clone(),
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

pub fn google_vertex_provider() -> GoogleVertexProvider {
    let default_models = vec![
        Model {
            id: "gemini-2.5-pro".into(),
            provider_id: "google-vertex".into(),
            name: "Gemini 2.5 Pro (Vertex AI)".into(),
            api: "google-vertex-ai".into(),
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
        },
        Model {
            id: "gemini-2.0-flash".into(),
            provider_id: "google-vertex".into(),
            name: "Gemini 2.0 Flash (Vertex AI)".into(),
            api: "google-vertex-ai".into(),
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
        },
    ];
    GoogleVertexProvider::new(
        "google-vertex",
        "Google Vertex AI",
        GoogleVertexProviderConfig {
            region: "us-central1".into(),
            project_id: String::new(), // resolved at auth time from GOOGLE_CLOUD_PROJECT
            env_var: "GOOGLE_APPLICATION_CREDENTIALS".into(),
            default_models,
        },
    )
}

#[async_trait]
impl Provider for GoogleVertexProvider {
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
            .with_error_message("Google Vertex AI live adapter not available in this build");
        es.push(AssistantMessageEvent::Error {
            reason: "error".into(),
            error: "Google Vertex AI live adapter not available in this build".into(),
        });
        es.end(Some(msg));
        es
    }
}
