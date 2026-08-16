pub mod context;
pub mod types;

use crate::types::CancellationToken;
pub use context::{default_provider_auth_context, parse_loose_credential, DefaultAuthContext, FileCredentialStore, InMemoryCredentialStore};
pub use types::{
    ApiKeyAuth, ApiKeyCredential, AuthCheck, AuthContext, AuthEvent, AuthInfoLink, AuthInteraction,
    AuthOperationOptions, AuthPrompt, AuthResult, AuthSelectOption, AuthType, Credential,
    CredentialInfo, CredentialStore, ModelAuth, ModifyFnOutput, OAuthAuth, OAuthCredential,
    OAuthCredentials, ProviderAuth,
};
pub use crate::types::ProviderEnv;

#[derive(Default)]
pub struct AuthResolutionOverrides {
    pub credential: Option<Credential>,
    pub base_url: Option<String>,
    pub headers: Option<std::collections::HashMap<String, String>>,
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
    provider: &dyn crate::providers::Provider,
    ctx: &dyn AuthContext,
    store: &dyn CredentialStore,
    overrides: AuthResolutionOverrides,
    signal: &CancellationToken,
) -> Result<AuthResult, ModelsError> {
    const REFRESH_SKEW_MS: i64 = 5 * 60 * 1000;

    let provider_auth = provider.auth();
    let op = AuthOperationOptions {
        signal: Some(signal.clone()),
    };

    // Stored credential; an override credential always wins.
    let stored = if let Some(cred) = overrides.credential {
        Some(cred)
    } else {
        store.read(provider.id(), op.clone()).await.unwrap_or(None)
    };

    let mut auth = ModelAuth::default();
    let mut env: Option<ProviderEnv> = None;
    let mut source: Option<String> = None;

    match (provider_auth.oauth.as_deref(), &stored) {
        // OAuth subscription providers with a stored OAuth credential.
        (Some(oauth), Some(Credential::OAuth(oc))) => {
            let now = chrono::Utc::now().timestamp_millis();
            let needs_refresh = oc.inner.expires == 0 || oc.inner.expires < now + REFRESH_SKEW_MS;
            let cred = if needs_refresh && !oc.inner.refresh.is_empty() {
                let refreshed = oauth.refresh(oc, signal).await.map_err(|e| ModelsError {
                    code: ModelsErrorCode::Auth(format!(
                        "OAuth token refresh failed for {}: {}",
                        provider.id(),
                        e
                    )),
                    source: Some(e),
                })?;
                // Persist the refreshed credential back into the store.
                let persisted = Credential::OAuth(refreshed.clone());
                store
                    .modify_fn(
                        provider.id(),
                        Box::new(move |_| {
                            Box::pin(async move { Ok(Some(persisted)) }) as ModifyFnOutput
                        }),
                        op.clone(),
                    )
                    .await
                    .unwrap_or(None);
                refreshed
            } else {
                oc.clone()
            };
            auth = oauth.to_auth(&cred).await.map_err(|e| ModelsError {
                code: ModelsErrorCode::Auth(format!(
                    "OAuth credential conversion failed for {}: {}",
                    provider.id(),
                    e
                )),
                source: Some(e),
            })?;
            source = Some("credential_store".to_string());
        }
        // API-key style providers (or mismatched credential types).
        _ => {
            if let Some(Credential::ApiKey(k)) = &stored {
                auth.api_key = k.key.clone();
                env = k.env.clone();
                source = Some("credential_store".to_string());
            } else if let Some(api_key_auth) = provider_auth.api_key.as_deref() {
                // No stored credential — fall back to the provider's own
                // ambient resolution (usually an environment variable).
                if let Some(result) = api_key_auth.resolve(ctx, None, signal).await {
                    auth = result.auth;
                    env = result.env;
                    source = result.source;
                }
            }
        }
    }

    // Merge base_url / header overrides on top of whatever we resolved.
    if let Some(b) = overrides.base_url {
        auth.base_url = Some(b);
    }
    if let Some(h) = overrides.headers {
        let mut merged = auth.headers.unwrap_or_default();
        for (k, v) in h {
            merged.insert(k, v);
        }
        auth.headers = Some(merged);
    }

    if source.is_none() {
        source = Some("ambient".to_string());
    }

    Ok(AuthResult {
        auth,
        env,
        source,
    })
}
