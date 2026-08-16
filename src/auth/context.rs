use super::types::{
    AuthContext as AuthContextTrait, AuthOperationOptions, Credential, CredentialInfo,
    CredentialStore, ModifyFn,
};
use parking_lot::Mutex;
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
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

    async fn list(&self, _options: AuthOperationOptions) -> anyhow::Result<Vec<CredentialInfo>> {
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

// ---------------------------------------------------------------------------
// File-backed credential store (~/.otter/credentials.json)
// ---------------------------------------------------------------------------

/// JSON-file backed [`CredentialStore`], persisted under the otter-ai config
/// directory (`credentials.json`, usually `~/.otter/credentials.json`).
///
/// The loader accepts both otter-ai's native serde format
/// (`{"type":"api_key","key":"…"}` / `{"type":"oauth","access":"…"}`) and the
/// looser hand-written format (`{"type":"api","key":"…"}` /
/// `{"type":"oauth","access_token":"…","refresh_token":"…","expires_at":"…"}`
/// with RFC 3339 timestamps), so the file can be shared with other tools.
pub struct FileCredentialStore {
    path: PathBuf,
    lock: Arc<Mutex<()>>,
}

impl FileCredentialStore {
    /// Open (or lazily create) the store at the default config location.
    pub fn open() -> anyhow::Result<Self> {
        let path = crate::utils::config_path("credentials.json")?
            .ok_or_else(|| anyhow::anyhow!("no home directory: cannot locate credentials.json"))?;
        Ok(Self {
            path,
            lock: Arc::new(Mutex::new(())),
        })
    }

    pub fn with_path(path: PathBuf) -> Self {
        Self {
            path,
            lock: Arc::new(Mutex::new(())),
        }
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    fn load_all(&self) -> HashMap<String, Credential> {
        let Ok(raw) = std::fs::read_to_string(&self.path) else {
            return HashMap::new();
        };
        let Ok(value) = serde_json::from_str::<Value>(&raw) else {
            return HashMap::new();
        };
        let Some(map) = value.as_object() else {
            return HashMap::new();
        };
        map.iter()
            .filter_map(|(k, v)| parse_loose_credential(v).map(|c| (k.clone(), c)))
            .collect()
    }

    fn write_all(&self, entries: &HashMap<String, Credential>) -> anyhow::Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Native serde format for round-trips written by otter-ai itself.
        let mut out = serde_json::Map::new();
        for (k, v) in entries {
            out.insert(k.clone(), serde_json::to_value(v)?);
        }
        let json = serde_json::to_string_pretty(&Value::Object(out))?;
        std::fs::write(&self.path, json)?;
        restrict_permissions(&self.path);
        Ok(())
    }
}

#[cfg(unix)]
fn restrict_permissions(path: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
}

#[cfg(not(unix))]
fn restrict_permissions(_path: &std::path::Path) {}

/// Parse a credential from either the native serde shape or the loose
/// hand-written shape used by `~/.otter/credentials.json`.
pub fn parse_loose_credential(v: &Value) -> Option<Credential> {
    let obj = v.as_object()?;
    let kind = obj.get("type").and_then(|t| t.as_str())?;
    match kind {
        "api" | "api_key" => {
            let key = obj
                .get("key")
                .and_then(|k| k.as_str())
                .map(|s| s.to_string());
            let env = obj
                .get("env")
                .and_then(|e| serde_json::from_value(e.clone()).ok());
            Some(Credential::ApiKey(crate::auth::types::ApiKeyCredential {
                key,
                env,
            }))
        }
        "oauth" => {
            // Native format deserialises directly.
            if obj.contains_key("access") && !obj.contains_key("access_token") {
                return serde_json::from_value(v.clone()).ok();
            }
            let access = obj
                .get("access_token")
                .or_else(|| obj.get("access"))
                .and_then(|a| a.as_str())?
                .to_string();
            let refresh = obj
                .get("refresh_token")
                .or_else(|| obj.get("refresh"))
                .and_then(|r| r.as_str())
                .unwrap_or_default()
                .to_string();
            let expires = obj
                .get("expires")
                .and_then(|e| e.as_i64())
                .or_else(|| {
                    obj.get("expires_at")
                        .and_then(|e| e.as_str())
                        .and_then(|s| {
                            chrono::DateTime::parse_from_rfc3339(s)
                                .ok()
                                .map(|d| d.timestamp_millis())
                        })
                })
                .unwrap_or(0);
            let mut extra = HashMap::new();
            if let Some(acc) = obj.get("chatgpt_account_id").and_then(|a| a.as_str()) {
                extra.insert("account_id".to_string(), Value::String(acc.to_string()));
            }
            if let Some(scope) = obj.get("scope").and_then(|s| s.as_str()) {
                extra.insert("scope".to_string(), Value::String(scope.to_string()));
            }
            Some(Credential::OAuth(crate::auth::types::OAuthCredential {
                inner: crate::auth::types::OAuthCredentials {
                    access,
                    refresh,
                    expires,
                    extra,
                },
            }))
        }
        _ => None,
    }
}

/// Alternate storage keys accepted per provider id. The ChatGPT Plus/Pro
/// (Codex) subscription OAuth credential is conventionally stored under the
/// `openai` key (matching the hand-written `~/.otter/credentials.json`
/// format), so `chatgpt-plus` lookups fall back to it.
const KEY_ALIASES: &[(&str, &[&str])] = &[("chatgpt-plus", &["openai"])];

fn alias_keys(provider_id: &str) -> Vec<String> {
    let mut keys = vec![provider_id.to_string()];
    if let Some((_, aliases)) = KEY_ALIASES.iter().find(|(id, _)| *id == provider_id) {
        keys.extend(aliases.iter().map(|a| a.to_string()));
    }
    keys
}

#[async_trait::async_trait]
impl CredentialStore for FileCredentialStore {
    async fn read(
        &self,
        provider_id: &str,
        _options: AuthOperationOptions,
    ) -> anyhow::Result<Option<Credential>> {
        let all = self.load_all();
        for key in alias_keys(provider_id) {
            if let Some(cred) = all.get(&key) {
                return Ok(Some(cred.clone()));
            }
        }
        Ok(None)
    }

    async fn list(&self, _options: AuthOperationOptions) -> anyhow::Result<Vec<CredentialInfo>> {
        Ok(self
            .load_all()
            .into_iter()
            .map(|(id, cred)| CredentialInfo {
                provider_id: id,
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
        // Note: the guard is deliberately NOT held across `f(...)` — user
        // callbacks may call back into the store. The read-modify-write is
        // therefore best-effort atomic, matching pi-ai's file store.
        let current = self.load_all().get(provider_id).cloned();
        let result = f(current).await?;
        let _guard = self.lock.lock();
        let mut all = self.load_all();
        match &result {
            Some(cred) => {
                all.insert(provider_id.to_string(), cred.clone());
            }
            None => {
                all.remove(provider_id);
            }
        }
        self.write_all(&all)?;
        Ok(result)
    }

    async fn delete(
        &self,
        provider_id: &str,
        _options: AuthOperationOptions,
    ) -> anyhow::Result<()> {
        let _guard = self.lock.lock();
        let mut all = self.load_all();
        all.remove(provider_id);
        self.write_all(&all)?;
        Ok(())
    }
}
