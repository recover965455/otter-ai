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
    ProviderHeaders, SimpleStreamOptions,
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
    /// Wire protocol label used to dispatch `stream` to the right adapter
    /// (e.g. `"openai-codex-responses"` for ChatGPT Plus/Pro).
    #[serde(default)]
    pub api_label: String,
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

impl GenericOAuthAuth {
    /// PKCE browser login: start a local callback server on the redirect URI,
    /// race it against a manual code prompt, then exchange the authorization
    /// code for access+refresh tokens (TS: `loginOpenAICodex`).
    async fn login_browser(
        &self,
        interaction: &(dyn AuthInteraction + Send + Sync),
    ) -> anyhow::Result<OAuthCredential> {
        let (verifier, challenge) = generate_pkce();
        let state = create_state();
        let redirect_uri = self
            .config
            .redirect_uri
            .as_deref()
            .unwrap_or("http://localhost:1455/auth/callback");

        let mut params: Vec<(String, String)> = vec![
            ("response_type".into(), "code".into()),
            ("client_id".into(), self.config.client_id.clone()),
            ("redirect_uri".into(), redirect_uri.to_string()),
            ("scope".into(), self.config.scopes.join(" ")),
            ("code_challenge".into(), challenge.clone()),
            ("code_challenge_method".into(), "S256".into()),
            ("state".into(), state.clone()),
        ];
        if self.config.api_label == "openai-codex-responses" {
            params.push(("id_token_add_organizations".into(), "true".into()));
            params.push(("codex_cli_simplified_flow".into(), "true".into()));
            params.push(("originator".into(), "pi".into()));
        }
        let query = params
            .iter()
            .map(|(k, v)| format!("{}={}", urlencode(k), urlencode(v)))
            .collect::<Vec<_>>()
            .join("&");
        let url = format!(
            "{}?{}",
            self.config.auth_url.as_deref().unwrap_or(""),
            query
        );

        // Start the callback server BEFORE publishing the URL (pi-ai order),
        // so a fast browser redirect is never lost.
        let server = start_local_oauth_server(redirect_uri, &state).await;

        interaction.notify(AuthEvent::AuthUrl {
            url: url.clone(),
            instructions: Some(format!(
                "A browser window should open. Complete login for {} to finish.",
                self.config.display_name
            )),
        });
        let prompt_token = CancellationToken::new();
        let manual = interaction.prompt(crate::auth::types::AuthPrompt::ManualCode {
            message: "Complete login in your browser, or paste the authorization code / redirect URL here:"
                .to_string(),
            placeholder: Some(redirect_uri.to_string()),
            signal: Some(prompt_token.clone()),
        });
        tokio::pin!(manual);

        let server_wait = async {
            match &server {
                Some(s) => s.wait_for_code().await,
                None => futures::future::pending().await,
            }
        };
        tokio::pin!(server_wait);

        let code: String = tokio::select! {
            server_result = &mut server_wait, if server.is_some() => {
                prompt_token.cancel();
                match server_result {
                    Ok(Ok(code)) => code,
                    Ok(Err(e)) => anyhow::bail!("OAuth callback error: {}", e),
                    Err(_) => anyhow::bail!("OAuth callback server shut down unexpectedly"),
                }
            }
            manual_result = &mut manual => {
                if let Some(s) = &server {
                    s.shutdown.cancel();
                }
                let input = manual_result?;
                let (parsed_code, parsed_state) = parse_authorization_input(&input);
                if let Some(s) = &parsed_state {
                    if *s != state {
                        anyhow::bail!("OAuth state mismatch");
                    }
                }
                parsed_code.ok_or_else(|| anyhow::anyhow!("Missing authorization code"))?
            }
        };

        let token_url = self.config.token_url.as_deref().ok_or_else(|| {
            anyhow::anyhow!("No token URL configured for {}", self.config.display_name)
        })?;
        let response = reqwest::Client::new()
            .post(token_url)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(format!(
                "grant_type=authorization_code&client_id={}&code={}&code_verifier={}&redirect_uri={}",
                urlencode(&self.config.client_id),
                urlencode(&code),
                urlencode(&verifier),
                urlencode(redirect_uri),
            ))
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("{} token exchange error: {}", self.config.display_name, e))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!(
                "{} token exchange failed ({}): {}",
                self.config.display_name,
                status,
                text
            );
        }

        let json: serde_json::Value = response.json().await.map_err(|e| {
            anyhow::anyhow!(
                "Invalid token exchange response for {}: {}",
                self.config.display_name,
                e
            )
        })?;
        let access = json
            .get("access_token")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Token exchange response missing access_token"))?;
        let refresh = json
            .get("refresh_token")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Token exchange response missing refresh_token"))?;
        let expires_in = json
            .get("expires_in")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow::anyhow!("Token exchange response missing expires_in"))?;

        let mut extra = std::collections::HashMap::new();
        if let Ok(account_id) = crate::providers::openai_responses::extract_account_id(access) {
            extra.insert(
                "account_id".to_string(),
                serde_json::Value::String(account_id),
            );
        }

        Ok(OAuthCredential {
            inner: OAuthCredentials {
                access: access.to_string(),
                refresh: refresh.to_string(),
                expires: chrono::Utc::now().timestamp_millis() + expires_in * 1000,
                extra,
            },
        })
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
        } else if self.config.auth_url.is_some() && self.config.token_url.is_some() {
            // --- PKCE browser flow (ChatGPT Plus/Codex, OpenRouter, Claude) ---
            return self.login_browser(interaction).await;
        } else if let Some(_auth_url) = &self.config.auth_url {
            // --- Authorization URL without token exchange support ---
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

        let mut extra = std::collections::HashMap::new();
        if let Some(refresh) = self.config.token_url.as_ref() {
            extra.insert(
                "token_url".to_string(),
                serde_json::Value::String(refresh.clone()),
            );
        }

        Ok(OAuthCredential {
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
        let token_url = self.config.token_url.as_deref().ok_or_else(|| {
            anyhow::anyhow!("No token URL configured for {}", self.config.display_name)
        })?;
        if credential.inner.refresh.is_empty() {
            anyhow::bail!(
                "No refresh token available for {}",
                self.config.display_name
            );
        }

        let client = reqwest::Client::new();
        let response = client
            .post(token_url)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(format!(
                "grant_type=refresh_token&refresh_token={}&client_id={}",
                urlencode(&credential.inner.refresh),
                urlencode(&self.config.client_id),
            ))
            .send()
            .await
            .map_err(|e| {
                anyhow::anyhow!("{} token refresh error: {}", self.config.display_name, e)
            })?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!(
                "{} token refresh failed ({}): {}",
                self.config.display_name,
                status,
                text
            );
        }

        let json: serde_json::Value = response.json().await.map_err(|e| {
            anyhow::anyhow!(
                "Invalid token refresh response for {}: {}",
                self.config.display_name,
                e
            )
        })?;
        let access = json
            .get("access_token")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Token refresh response missing access_token"))?;
        let refresh = json
            .get("refresh_token")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Token refresh response missing refresh_token"))?;
        let expires_in = json
            .get("expires_in")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow::anyhow!("Token refresh response missing expires_in"))?;

        let mut extra = credential.inner.extra.clone();
        if let Ok(account_id) = crate::providers::openai_responses::extract_account_id(access) {
            extra.insert(
                "account_id".to_string(),
                serde_json::Value::String(account_id),
            );
        }

        Ok(OAuthCredential {
            inner: OAuthCredentials {
                access: access.to_string(),
                refresh: refresh.to_string(),
                expires: chrono::Utc::now().timestamp_millis() + expires_in * 1000,
                extra,
            },
        })
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
// PKCE browser login (TS: auth/oauth/openai-codex.ts loginOpenAICodex)
// ---------------------------------------------------------------------------

fn base64url_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | (b[2] as u32);
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            TABLE[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[n as usize & 63] as char
        } else {
            '='
        });
    }
    // PKCE verifiers/challenges are unpadded base64url.
    out.trim_end_matches('=').to_string()
}

/// TS: `generatePKCE` — 32 random bytes → verifier; S256 of the verifier
/// (RFC 7636) → challenge.
fn generate_pkce() -> (String, String) {
    use rand::RngCore;
    let mut verifier_bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut verifier_bytes);
    let verifier = base64url_encode(&verifier_bytes);

    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(verifier.as_bytes());
    let challenge = base64url_encode(&hasher.finalize());
    (verifier, challenge)
}

fn create_state() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut bytes);
    base64url_encode(&bytes)
}

fn url_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                if let Ok(b) = u8::from_str_radix(hex, 16) {
                    out.push(b);
                    i += 3;
                } else {
                    out.push(bytes[i]);
                    i += 1;
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

fn parse_query(query: &str) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            out.insert(url_decode(k), url_decode(v));
        }
    }
    out
}

/// TS: `parseAuthorizationInput` — accepts a full redirect URL, a
/// `code#state` fragment, or a bare authorization code.
fn parse_authorization_input(input: &str) -> (Option<String>, Option<String>) {
    let input = input.trim();
    if let Some(q) = input.find('?') {
        let params = parse_query(&input[q + 1..]);
        return (params.get("code").cloned(), params.get("state").cloned());
    }
    if let Some((code, state)) = input.split_once('#') {
        return (
            Some(code.trim().to_string()),
            Some(state.trim().to_string()),
        );
    }
    (Some(input.to_string()), None)
}

/// Split a redirect URI into (host, port, path) for the local callback
/// server (TS: `REDIRECT_URI` + `getCallbackHost()`).
fn redirect_uri_parts(redirect_uri: &str) -> Option<(String, u16, String)> {
    let rest = redirect_uri
        .strip_prefix("https://")
        .or_else(|| redirect_uri.strip_prefix("http://"))?;
    let (authority, path) = match rest.split_once('/') {
        Some((a, p)) => (a, format!("/{}", p)),
        None => (rest, "/".to_string()),
    };
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse().ok()?),
        None => (authority.to_string(), 80),
    };
    Some((host, port, path))
}

struct LocalOAuthServer {
    _handle: tokio::task::JoinHandle<()>,
    code_rx: std::sync::Mutex<Option<tokio::sync::oneshot::Receiver<Result<String, String>>>>,
    shutdown: CancellationToken,
}

impl LocalOAuthServer {
    async fn wait_for_code(
        &self,
    ) -> Result<Result<String, String>, tokio::sync::oneshot::error::RecvError> {
        let rx = self.code_rx.lock().unwrap().take();
        match rx {
            Some(rx) => rx.await,
            None => futures::future::pending().await,
        }
    }
}

fn oauth_page_html(status: &str, message: &str) -> String {
    format!(
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>{}</title></head>\
         <body style=\"font-family:system-ui;margin:2rem\"><h1>{}</h1><p>{}</p></body></html>",
        status, status, message
    )
}

async fn write_html_response(
    stream: &mut tokio::net::TcpStream,
    status: u16,
    reason: &str,
    body: &str,
) {
    use tokio::io::AsyncWriteExt;
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status,
        reason,
        body.len()
    );
    let _ = stream.write_all(head.as_bytes()).await;
    let _ = stream.write_all(body.as_bytes()).await;
}

/// Bind the local callback server on the redirect URI's host/port/path.
/// Returns `None` when the port is unavailable (e.g. another login is already
/// listening), in which case the login proceeds via manual code only — same
/// as pi-ai's `startLocalOAuthServer` error path.
async fn start_local_oauth_server(redirect_uri: &str, state: &str) -> Option<LocalOAuthServer> {
    let (host, port, path) = redirect_uri_parts(redirect_uri)?;
    let listener = tokio::net::TcpListener::bind((host.as_str(), port))
        .await
        .ok()?;
    let (code_tx, code_rx) = tokio::sync::oneshot::channel();
    let shutdown = CancellationToken::new();
    let shutdown_task = shutdown.clone();
    let state = state.to_string();
    let handle = tokio::spawn(async move {
        let code_tx = code_tx;
        loop {
            let (mut stream, _) = tokio::select! {
                _ = shutdown_task.cancelled_fut() => break,
                r = listener.accept() => match r {
                    Ok(x) => x,
                    Err(_) => break,
                },
            };
            // Read the request head (path + query).
            let mut buf = Vec::new();
            let mut tmp = [0u8; 4096];
            let head_end = loop {
                if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                    break Some(buf.len());
                }
                match tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    tokio::io::AsyncReadExt::read(&mut stream, &mut tmp),
                )
                .await
                {
                    Ok(Ok(0)) => break None,
                    Ok(Ok(n)) => buf.extend_from_slice(&tmp[..n]),
                    _ => break None,
                }
            };
            let Some(head_end) = head_end else { continue };
            let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
            let request_line = head.lines().next().unwrap_or("");
            let mut parts = request_line.split_whitespace();
            let _method = parts.next().unwrap_or("");
            let target = parts.next().unwrap_or("");

            let (request_path, query) = match target.split_once('?') {
                Some((p, q)) => (p.to_string(), Some(q.to_string())),
                None => (target.to_string(), None),
            };

            if request_path != path {
                write_html_response(
                    &mut stream,
                    404,
                    "Not Found",
                    &oauth_page_html(
                        "Callback route not found.",
                        "Open the exact URL shown in the login prompt.",
                    ),
                )
                .await;
                continue;
            }
            let params = query.as_deref().map(parse_query).unwrap_or_default();
            if params.get("state").map(|s| s.as_str()) != Some(state.as_str()) {
                write_html_response(
                    &mut stream,
                    400,
                    "Bad Request",
                    &oauth_page_html(
                        "State mismatch.",
                        "The OAuth state did not match; please retry the login.",
                    ),
                )
                .await;
                continue;
            }
            let Some(code) = params.get("code").cloned() else {
                write_html_response(
                    &mut stream,
                    400,
                    "Bad Request",
                    &oauth_page_html(
                        "Missing authorization code.",
                        "No authorization code was provided.",
                    ),
                )
                .await;
                continue;
            };
            write_html_response(
                &mut stream,
                200,
                "OK",
                &oauth_page_html(
                    "Authentication completed.",
                    "You can close this window and return to the terminal.",
                ),
            )
            .await;
            let _ = code_tx.send(Ok(code));
            break;
        }
    });
    Some(LocalOAuthServer {
        _handle: handle,
        code_rx: std::sync::Mutex::new(Some(code_rx)),
        shutdown,
    })
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
    fn apply_simple_options(
        &self,
        _ctx: &Context,
        options: &SimpleStreamOptions,
        api_opts: &mut ApiStreamOptions,
    ) {
        // Serialize simple-stream knobs into `extra_body` so the wire adapter
        // dispatched from `stream` can read them back (session cache info is
        // written there as well by the Models layer).
        let mut extra = api_opts
            .request_options
            .extra_body
            .clone()
            .unwrap_or_else(|| serde_json::json!({}));
        if let Some(obj) = extra.as_object_mut() {
            if let Some(temp) = options.temperature {
                obj.insert("temperature".into(), serde_json::json!(temp));
            }
            if let Some(reasoning) = &options.reasoning {
                obj.insert("reasoning".into(), serde_json::json!(reasoning));
            } else if let Some(level) = options.thinking {
                let effort = match level {
                    crate::types::ModelThinkingLevel::None => None,
                    crate::types::ModelThinkingLevel::Low => Some("low"),
                    crate::types::ModelThinkingLevel::Medium => Some("medium"),
                    crate::types::ModelThinkingLevel::High => Some("high"),
                };
                if let Some(e) = effort {
                    // The wire adapter re-maps this through the model's
                    // thinking_level_map at request-build time.
                    obj.insert("reasoning".into(), serde_json::json!(e));
                }
            }
            if let Some(choice) = &options.tool_choice {
                let v = match choice {
                    crate::types::ToolChoice::Auto => serde_json::json!("auto"),
                    crate::types::ToolChoice::None => serde_json::json!("none"),
                    crate::types::ToolChoice::Required => serde_json::json!("required"),
                    crate::types::ToolChoice::Tool { name } => {
                        serde_json::json!({ "type": "function", "name": name })
                    }
                };
                obj.insert("tool_choice".into(), v);
            }
            if let Some(b) = &options.base_url {
                obj.insert("base_url_override".into(), serde_json::json!(b));
            }
        }
        api_opts.request_options.extra_body = Some(extra);
    }
    async fn refresh_models(
        &self,
        _cx: Box<dyn RefreshModelsContext + Send + 'static>,
    ) -> Result<(), String> {
        Ok(())
    }
    fn stream(
        &self,
        model: &Model,
        context: Context,
        options: ApiStreamOptions,
    ) -> AssistantMessageEventStream {
        if self.config.api_label == "openai-codex-responses" {
            let codex_opts = crate::providers::openai_responses::codex_options_from_api(
                model,
                &self.config,
                &options,
            );
            return crate::providers::openai_responses::stream_codex(model, context, codex_opts);
        }
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

/// Minimal percent-encoding for `application/x-www-form-urlencoded` bodies.
fn urlencode(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for b in input.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

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
            api_label: spec.api_label.to_string(),
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
