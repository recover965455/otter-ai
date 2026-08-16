//! Shared building blocks for **OAuth-based subscription providers**
//! (ChatGPT Plus/Codex, Claude Pro/Max, GitHub Copilot, xAI subscription,
//! OpenRouter PKCE, Radius).
//!
//! Each concrete OAuth provider is a thin spec around
//! [`GenericOAuthProvider`] / [`GenericOAuthAuth`], exactly mirroring how
//! `openai_compat` works for API-key providers.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::{Provider, RefreshModelsContext};
use crate::auth::types::{
    AuthEvent, AuthInteraction, AuthResult, ModelAuth, OAuthAuth, OAuthCredential,
    OAuthCredentials, ProviderAuth,
};
use crate::types::{
    ApiStreamOptions, AssistantMessageEvent, CancellationToken, Context, Message, Model,
    ProviderHeaders,
};
use crate::utils::event_stream::AssistantMessageEventStream;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthProviderConfig {
    pub base_url: String,
    pub client_id: String,
    pub scopes: Vec<String>,
    /// Authorization endpoint (browser-based PKCE flow).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_url: Option<String>,
    /// Token endpoint (code → access+refresh exchange).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_url: Option<String>,
    /// Device authorization endpoint (device-code flow, e.g. GitHub Copilot).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_auth_url: Option<String>,
    /// Redirect URI for PKCE flow.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub redirect_uri: Option<String>,
    pub display_name: String,
    pub is_subscription: bool,
    pub login_label: Option<String>,
    /// Extra static headers to inject on every request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_headers: Option<ProviderHeaders>,
    pub default_models: Vec<Model>,
}

// ---------------------------------------------------------------------------
// OAuthAuth impl
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct GenericOAuthAuth {
    config: OAuthProviderConfig,
}

impl GenericOAuthAuth {
    pub fn new(config: OAuthProviderConfig) -> Self {
        Self { config }
    }

    /// Build the authorization URL for the PKCE browser flow.
    fn build_auth_url(&self) -> String {
        let auth_url = self.config.auth_url.as_deref().unwrap_or("");
        let redirect = self
            .config
            .redirect_uri
            .as_deref()
            .unwrap_or("http://localhost:1455/callback");
        let scopes = self.config.scopes.join(" ");
        format!(
            "{}?client_id={}&redirect_uri={}&response_type=code&scope={}",
            auth_url, self.config.client_id, redirect, scopes
        )
    }
}

#[async_trait]
impl OAuthAuth for GenericOAuthAuth {
    fn name(&self) -> &str {
        &self.config.display_name
    }

    fn is_subscription(&self) -> bool {
        self.config.is_subscription
    }

    fn login_label(&self) -> Option<&str> {
        self.config.login_label.as_deref()
    }

    async fn login(
        &self,
        interaction: &(dyn AuthInteraction + Send + Sync),
    ) -> anyhow::Result<OAuthCredential> {
        // --- Device-code flow (e.g. GitHub Copilot, xAI) ---
        if let Some(device_url) = &self.config.device_auth_url {
            interaction.notify(AuthEvent::AuthUrl {
                url: device_url.clone(),
                instructions: Some(format!(
                    "Open {} to start device authorization for {}, then paste the code below.",
                    device_url, self.config.display_name
                )),
            });
        } else if let Some(_auth_url) = &self.config.auth_url {
            // --- PKCE browser flow (e.g. OpenRouter, ChatGPT, Claude) ---
            let url = self.build_auth_url();
            interaction.notify(AuthEvent::AuthUrl {
                url: url.clone(),
                instructions: Some(format!(
                    "Open this URL to authorize {}, then paste the redirect URL or authorization code below.",
                    self.config.display_name
                )),
            });
        } else {
            // --- No known auth endpoint; just ask for a token manually ---
            interaction.notify(AuthEvent::Info {
                message: format!(
                    "Manual login for {}. Paste an OAuth access token below.",
                    self.config.display_name
                ),
                links: vec![],
            });
        }

        let code = interaction
            .prompt(crate::auth::types::AuthPrompt::ManualCode {
                message: format!(
                    "Paste the authorization code or token for {}:",
                    self.config.display_name
                ),
                placeholder: Some("code or redirect URL…".into()),
                signal: interaction.signal().cloned(),
            })
            .await?;

        // TODO: exchange `code` for access+refresh tokens via HTTP POST to
        // `self.config.token_url`.  Until the shared HTTP adapter lands, we
        // treat the user-supplied value as the access token directly.
        let mut extra = std::collections::HashMap::new();
        if let Some(refresh) = self.config.token_url.as_ref() {
            extra.insert(
                "token_url".to_string(),
                serde_json::Value::String(refresh.clone()),
            );
        }

        Ok(OAuthCredential {
            r#type: "oauth".to_string(),
            inner: OAuthCredentials {
                access: code,
                refresh: String::new(),
                expires: 0, // expired → will trigger refresh on first use
                extra,
            },
        })
    }

    async fn refresh(
        &self,
        credential: &OAuthCredential,
        _signal: &CancellationToken,
    ) -> anyhow::Result<OAuthCredential> {
        // TODO: POST refresh_token to token_url, return new credential.
        // Until the HTTP adapter lands, return the credential unchanged.
        Ok(credential.clone())
    }

    async fn to_auth(&self, credential: &OAuthCredential) -> anyhow::Result<ModelAuth> {
        Ok(ModelAuth {
            api_key: Some(credential.inner.access.clone()),
            headers: self.config.extra_headers.clone(),
            base_url: Some(self.config.base_url.clone()),
        })
    }
}

// ---------------------------------------------------------------------------
// Provider struct + impl
// ---------------------------------------------------------------------------

struct AuthHolder {
    auth: ProviderAuth,
}

pub struct GenericOAuthProvider {
    id: String,
    name: String,
    config: OAuthProviderConfig,
    auth_holder: Arc<AuthHolder>,
}

impl GenericOAuthProvider {
    pub fn new(id: &str, name: &str, config: OAuthProviderConfig) -> Self {
        let auth_impl = GenericOAuthAuth::new(config.clone());
        Self {
            id: id.to_string(),
            name: name.to_string(),
            config,
            auth_holder: Arc::new(AuthHolder {
                auth: ProviderAuth {
                    api_key: None,
                    oauth: Some(Box::new(auth_impl)),
                },
            }),
        }
    }
}

#[async_trait]
impl Provider for GenericOAuthProvider {
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
            "{} live adapter not available in this build (OAuth wire protocol shared adapter TBD)",
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
// Helper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct OAuthProviderSpec<'a> {
    pub id: &'a str,
    pub display_name: &'a str,
    pub base_url: &'a str,
    pub client_id: &'a str,
    pub scopes: &'a [&'a str],
    pub auth_url: Option<&'a str>,
    pub token_url: Option<&'a str>,
    pub device_auth_url: Option<&'a str>,
    pub redirect_uri: Option<&'a str>,
    pub is_subscription: bool,
    pub login_label: Option<&'a str>,
    pub api_label: &'a str,
    pub default_models_fn: fn() -> Vec<Model>,
    pub extra_headers: Option<ProviderHeaders>,
}

#[allow(clippy::too_many_arguments)]
pub fn build_oauth_provider(spec: OAuthProviderSpec) -> GenericOAuthProvider {
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
    GenericOAuthProvider::new(
        spec.id,
        spec.display_name,
        OAuthProviderConfig {
            base_url: spec.base_url.to_string(),
            client_id: spec.client_id.to_string(),
            scopes: spec.scopes.iter().map(|s| s.to_string()).collect(),
            auth_url: spec.auth_url.map(|s| s.to_string()),
            token_url: spec.token_url.map(|s| s.to_string()),
            device_auth_url: spec.device_auth_url.map(|s| s.to_string()),
            redirect_uri: spec.redirect_uri.map(|s| s.to_string()),
            display_name: spec.display_name.to_string(),
            is_subscription: spec.is_subscription,
            login_label: spec.login_label.map(|s| s.to_string()),
            extra_headers: spec.extra_headers,
            default_models: models,
        },
    )
}

// Convenience: resolve an OAuth credential into AuthResult for the
// resolve_provider_auth path (used by Models registry).
pub async fn resolve_oauth_auth(
    auth: &dyn OAuthAuth,
    credential: &OAuthCredential,
) -> anyhow::Result<AuthResult> {
    let model_auth = auth.to_auth(credential).await?;
    Ok(AuthResult {
        auth: model_auth,
        env: None,
        source: Some("credential_store".to_string()),
    })
}
