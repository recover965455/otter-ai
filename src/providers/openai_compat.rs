//! Shared building blocks for **OpenAI Chat Completions**-compatible providers.
//!
//! The vast majority of third-party LLM vendors expose a wire-compatible
//! `/v1/chat/completions` endpoint and authenticate via a bearer API key read
//! from an environment variable.  Rather than copying 200+ lines of identical
//! code for each vendor, concrete providers construct themselves through the
//! helpers in this module:
//!
//! * [`GenericCompatConfig`] — base URL, env var, default model catalog.
//! * [`GenericCompatApiKeyAuth`] — reusable [`ApiKeyAuth`] implementation that
//!   additionally supports injecting static extra headers (e.g. for providers
//!   like Vercel / Cloudflare that need `Authorization: Bearer ...` **and** a
//!   secondary identification header).
//! * [`GenericCompatProvider`] / [`build_compat_provider`] — zero-boilerplate
//!   [`Provider`] impl that stubs the live HTTP stream for now (the actual
//!   request/response adapter will live in a single shared place, same as
//!   `@earendil-works/pi-ai` does for all its OpenAI-compatible backends).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::{Provider, RefreshModelsContext};
use crate::auth::types::{
    ApiKeyAuth, AuthContext, AuthInteraction, AuthResult, ModelAuth, ProviderAuth,
};
use crate::types::{
    ApiStreamOptions, AssistantMessageEvent, CancellationToken, Context, Message, Model,
    ProviderHeaders,
};
use crate::utils::event_stream::AssistantMessageEventStream;

// ---------------------------------------------------------------------------
// Config & constructor primitives
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericCompatConfig {
    pub base_url: String,
    pub env_var: String,
    /// Human-friendly label shown in `/login` prompts and error messages.
    pub display_name: String,
    /// Placeholder shown in the secret prompt, e.g. `"sk-..."` or `"groq-..."`.
    pub key_placeholder: String,
    /// Static extra headers injected on every request (besides Authorization),
    /// e.g. `cf-account-id` for Cloudflare AI Gateway or `anthropic-version`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_headers: Option<ProviderHeaders>,
    /// Default model catalogue bundled with the provider.
    pub default_models: Vec<Model>,
}

// ---------------------------------------------------------------------------
// Auth impl
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct GenericCompatApiKeyAuth {
    env_var: String,
    base_url: String,
    name: String,
    placeholder: String,
    extra_headers: Option<ProviderHeaders>,
}

impl GenericCompatApiKeyAuth {
    pub fn new(cfg: &GenericCompatConfig) -> Self {
        Self {
            env_var: cfg.env_var.clone(),
            base_url: cfg.base_url.clone(),
            name: format!("{} API key", cfg.display_name),
            placeholder: cfg.key_placeholder.clone(),
            extra_headers: cfg.extra_headers.clone(),
        }
    }
}

#[async_trait]
impl ApiKeyAuth for GenericCompatApiKeyAuth {
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
                message: format!(
                    "Enter your {} API key:",
                    self.name.trim_end_matches(" API key")
                ),
                placeholder: Some(self.placeholder.clone()),
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
                headers: self.extra_headers.clone(),
                base_url: Some(self.base_url.clone()),
            },
            env: None,
            source,
        })
    }
}

// ---------------------------------------------------------------------------
// Provider struct + impl
// ---------------------------------------------------------------------------

struct CompatAuthHolder {
    auth: ProviderAuth,
}

pub struct GenericCompatProvider {
    id: String,
    name: String,
    config: GenericCompatConfig,
    auth_holder: Arc<CompatAuthHolder>,
}

impl GenericCompatProvider {
    pub fn new(id: &str, name: &str, config: GenericCompatConfig) -> Self {
        let auth_impl = GenericCompatApiKeyAuth::new(&config);
        Self {
            id: id.to_string(),
            name: name.to_string(),
            config,
            auth_holder: Arc::new(CompatAuthHolder {
                auth: ProviderAuth {
                    api_key: Some(Box::new(auth_impl)),
                    oauth: None,
                },
            }),
        }
    }

    pub fn config(&self) -> &GenericCompatConfig {
        &self.config
    }
}

#[async_trait]
impl Provider for GenericCompatProvider {
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
        let err_msg = format!(
            "{} live adapter not available in this build (OpenAI-compat wire protocol shared adapter TBD)",
            self.name
        );
        let msg = Message::assistant_default(self.id.clone(), self.id.clone())
            .with_error_message(&err_msg);
        es.push(AssistantMessageEvent::Error {
            reason: "error".into(),
            error: err_msg,
        });
        es.end(Some(msg));
        es
    }
}

// ---------------------------------------------------------------------------
// Helper shared by all concrete compat providers to build their register_*
// functions with zero boilerplate.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CompatProviderSpec<'a> {
    pub id: &'a str,
    pub display_name: &'a str,
    pub base_url: &'a str,
    pub env_var: &'a str,
    pub key_placeholder: &'a str,
    pub api_label: &'a str,
    pub default_models_fn: fn() -> Vec<Model>,
    #[allow(clippy::type_complexity)]
    pub extra_headers: Option<ProviderHeaders>,
}

pub fn build_compat_provider(spec: CompatProviderSpec) -> GenericCompatProvider {
    let provider_id = spec.id.to_string();
    let api_label = spec.api_label.to_string();
    let mut models = (spec.default_models_fn)();
    for m in &mut models {
        if m.provider_id.is_empty() {
            m.provider_id = provider_id.clone();
        }
        if m.api.is_empty() {
            m.api = api_label.clone();
        }
    }
    GenericCompatProvider::new(
        spec.id,
        spec.display_name,
        GenericCompatConfig {
            base_url: spec.base_url.to_string(),
            env_var: spec.env_var.to_string(),
            display_name: spec.display_name.to_string(),
            key_placeholder: spec.key_placeholder.to_string(),
            extra_headers: spec.extra_headers,
            default_models: models,
        },
    )
}
