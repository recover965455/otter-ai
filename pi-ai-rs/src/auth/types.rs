use crate::types::{CancellationToken, ProviderEnv, ProviderHeaders};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelAuth {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<ProviderHeaders>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyCredential {
    #[serde(default = "default_type_api_key")]
    pub r#type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<ProviderEnv>,
}

fn default_type_api_key() -> String {
    "api_key".to_string()
}

impl Default for ApiKeyCredential {
    fn default() -> Self {
        Self {
            r#type: "api_key".to_string(),
            key: None,
            env: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthCredentials {
    pub refresh: String,
    pub access: String,
    pub expires: i64,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthCredential {
    #[serde(default = "default_type_oauth")]
    pub r#type: String,
    #[serde(flatten)]
    pub inner: OAuthCredentials,
}

fn default_type_oauth() -> String {
    "oauth".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Credential {
    ApiKey(ApiKeyCredential),
    OAuth(OAuthCredential),
}

impl Credential {
    pub fn api_key(key: impl Into<String>) -> Self {
        Credential::ApiKey(ApiKeyCredential {
            r#type: "api_key".to_string(),
            key: Some(key.into()),
            env: None,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialInfo {
    pub provider_id: String,
    pub r#type: String,
}

#[derive(Debug, Clone, Default)]
pub struct AuthOperationOptions {
    pub signal: Option<CancellationToken>,
}

pub type ModifyFnOutput =
    std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<Option<Credential>>> + Send>>;

pub type ModifyFn = Box<dyn FnOnce(Option<Credential>) -> ModifyFnOutput + Send>;

#[async_trait::async_trait]
pub trait CredentialStore: Send + Sync {
    async fn read(
        &self,
        provider_id: &str,
        options: AuthOperationOptions,
    ) -> anyhow::Result<Option<Credential>>;

    async fn list(
        &self,
        options: AuthOperationOptions,
    ) -> anyhow::Result<Vec<CredentialInfo>>;

    async fn modify_fn(
        &self,
        provider_id: &str,
        f: ModifyFn,
        options: AuthOperationOptions,
    ) -> anyhow::Result<Option<Credential>>;

    async fn delete(
        &self,
        provider_id: &str,
        options: AuthOperationOptions,
    ) -> anyhow::Result<()>;
}

pub trait AuthContext: Send + Sync {
    fn env<'a>(
        &'a self,
        name: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<String>> + Send + 'a>>;
    fn file_exists<'a>(
        &'a self,
        path: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + 'a>>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthResult {
    pub auth: ModelAuth,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<ProviderEnv>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthCheck {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    pub r#type: String,
}

pub type AuthType = String;

#[derive(Debug, Clone)]
pub enum AuthPrompt {
    Text {
        message: String,
        placeholder: Option<String>,
        signal: Option<CancellationToken>,
    },
    Secret {
        message: String,
        placeholder: Option<String>,
        signal: Option<CancellationToken>,
    },
    Select {
        message: String,
        options: Vec<AuthSelectOption>,
        signal: Option<CancellationToken>,
    },
    ManualCode {
        message: String,
        placeholder: Option<String>,
        signal: Option<CancellationToken>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthSelectOption {
    pub id: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthInfoLink {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AuthEvent {
    Info {
        message: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        links: Vec<AuthInfoLink>,
    },
    AuthUrl {
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        instructions: Option<String>,
    },
    DeviceCode {
        user_code: String,
        verification_uri: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        interval_seconds: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        expires_in_seconds: Option<u64>,
    },
    Progress {
        message: String,
    },
}

#[async_trait::async_trait]
pub trait AuthInteraction: Send + Sync {
    fn signal(&self) -> Option<&CancellationToken>;
    async fn prompt(&self, prompt: AuthPrompt) -> anyhow::Result<String>;
    fn notify(&self, event: AuthEvent);
}

pub struct ProviderAuth {
    pub api_key: Option<Box<dyn ApiKeyAuth + Send + Sync>>,
    pub oauth: Option<Box<dyn OAuthAuth + Send + Sync>>,
}

#[async_trait::async_trait]
pub trait ApiKeyAuth {
    fn name(&self) -> &str;

    async fn login(
        &self,
        interaction: &(dyn AuthInteraction + Send + Sync),
    ) -> anyhow::Result<ApiKeyCredential>;

    async fn check(
        &self,
        _ctx: &(dyn AuthContext + Send + Sync),
        _credential: Option<&ApiKeyCredential>,
        _signal: &CancellationToken,
    ) -> Option<AuthCheck> {
        None
    }

    async fn resolve(
        &self,
        ctx: &(dyn AuthContext + Send + Sync),
        credential: Option<&ApiKeyCredential>,
        signal: &CancellationToken,
    ) -> Option<AuthResult>;
}

#[async_trait::async_trait]
pub trait OAuthAuth {
    fn name(&self) -> &str;
    fn is_subscription(&self) -> bool { false }
    fn login_label(&self) -> Option<&str> { None }

    async fn login(
        &self,
        interaction: &(dyn AuthInteraction + Send + Sync),
    ) -> anyhow::Result<OAuthCredential>;

    async fn refresh(
        &self,
        credential: &OAuthCredential,
        signal: &CancellationToken,
    ) -> anyhow::Result<OAuthCredential>;

    async fn to_auth(&self, credential: &OAuthCredential) -> anyhow::Result<ModelAuth>;
}
