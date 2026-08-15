use super::types::{
    AuthContext as AuthContextTrait, AuthOperationOptions, Credential, CredentialInfo,
    CredentialStore, ModifyFn,
};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Default)]
pub struct InMemoryCredentialStore {
    inner: Arc<Mutex<HashMap<String, Credential>>>,
}

impl InMemoryCredentialStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait::async_trait]
impl CredentialStore for InMemoryCredentialStore {
    async fn read(
        &self,
        provider_id: &str,
        _options: AuthOperationOptions,
    ) -> anyhow::Result<Option<Credential>> {
        Ok(self.inner.lock().get(provider_id).cloned())
    }

    async fn list(
        &self,
        _options: AuthOperationOptions,
    ) -> anyhow::Result<Vec<CredentialInfo>> {
        let store = self.inner.lock();
        Ok(store
            .iter()
            .map(|(id, cred)| CredentialInfo {
                provider_id: id.clone(),
                r#type: match cred {
                    Credential::ApiKey(_) => "api_key".to_string(),
                    Credential::OAuth(_) => "oauth".to_string(),
                },
            })
            .collect())
    }

    async fn modify_fn(
        &self,
        provider_id: &str,
        f: ModifyFn,
        _options: AuthOperationOptions,
    ) -> anyhow::Result<Option<Credential>> {
        let current = self.inner.lock().get(provider_id).cloned();
        let result = f(current).await?;
        if let Some(cred) = result.clone() {
            self.inner.lock().insert(provider_id.to_string(), cred);
        }
        Ok(result)
    }

    async fn delete(
        &self,
        provider_id: &str,
        _options: AuthOperationOptions,
    ) -> anyhow::Result<()> {
        self.inner.lock().remove(provider_id);
        Ok(())
    }
}

#[derive(Clone, Default)]
pub struct DefaultAuthContext;

impl AuthContextTrait for DefaultAuthContext {
    fn env<'a>(
        &'a self,
        name: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<String>> + Send + 'a>> {
        Box::pin(async move { std::env::var(name).ok() })
    }

    fn file_exists<'a>(
        &'a self,
        path: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + 'a>> {
        let expanded = if let Some(rest) = path.strip_prefix("~/") {
            if let Ok(home) = std::env::var("HOME") {
                std::path::PathBuf::from(home).join(rest)
            } else {
                std::path::PathBuf::from(path)
            }
        } else {
            std::path::PathBuf::from(path)
        };
        Box::pin(async move { expanded.exists() })
    }
}

pub fn default_provider_auth_context() -> DefaultAuthContext {
    DefaultAuthContext
}
