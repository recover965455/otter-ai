pub mod types;
pub mod context;

use crate::types::CancellationToken;
pub use context::{default_provider_auth_context, DefaultAuthContext, InMemoryCredentialStore};
pub use types::{
    ApiKeyAuth, ApiKeyCredential, AuthCheck, AuthContext, AuthEvent, AuthInfoLink,
    AuthInteraction, AuthOperationOptions, AuthPrompt, AuthResult, AuthSelectOption, AuthType,
    Credential, CredentialInfo, CredentialStore, ModelAuth, OAuthAuth, OAuthCredential,
    OAuthCredentials, ProviderAuth,
};

pub struct AuthResolutionOverrides {
    pub credential: Option<Credential>,
    pub base_url: Option<String>,
    pub headers: Option<std::collections::HashMap<String, String>>,
}

impl Default for AuthResolutionOverrides {
    fn default() -> Self {
        Self {
            credential: None,
            base_url: None,
            headers: None,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ModelsErrorCode {
    #[error("auth error: {0}")]
    Auth(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("network error: {0}")]
    Network(String),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error("provider error: {0}")]
    Provider(String),
    #[error("{0}")]
    Other(String),
}

#[derive(thiserror::Error, Debug)]
#[error("{code}")]
pub struct ModelsError {
    pub code: ModelsErrorCode,
    #[source]
    pub source: Option<anyhow::Error>,
}

pub async fn resolve_provider_auth(
    _provider: &dyn crate::providers::Provider,
    _ctx: &dyn AuthContext,
    _store: &dyn CredentialStore,
    overrides: AuthResolutionOverrides,
    _signal: &CancellationToken,
) -> Result<AuthResult, ModelsError> {
    // Override-first resolution: if the caller already handed in a credential or base_url,
    // use those. Otherwise, return a minimal auth result for providers that don't need keys.
    let mut auth = ModelAuth::default();
    if let Some(b) = overrides.base_url {
        auth.base_url = Some(b);
    }
    if let Some(h) = overrides.headers {
        auth.headers = Some(h);
    }
    if let Some(cred) = overrides.credential {
        match &cred {
            Credential::ApiKey(k) => {
                auth.api_key = k.key.clone();
                if let Some(env) = &k.env {
                    return Ok(AuthResult {
                        auth,
                        env: Some(env.clone()),
                        source: Some("override".to_string()),
                    });
                }
            }
            Credential::OAuth(_) => {
                // Simplified: skip OAuth refresh
            }
        }
    }
    Ok(AuthResult {
        auth,
        env: None,
        source: Some("override_or_default".to_string()),
    })
}
