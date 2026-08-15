use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::{Provider, RefreshModelsContext};
use crate::auth::types::{
    ApiKeyAuth, AuthContext, AuthInteraction, AuthResult, ModelAuth, ProviderAuth,
};
use crate::types::{
    ApiStreamOptions, AssistantMessage, AssistantMessageEvent, Context, CancellationToken,
    ContentBlock, Message, Model, ModelCostRates, ModelThinkingLevel, Usage,
};
use crate::utils::validation::calculate_usage_cost;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicProviderConfig {
    pub base_url: String,
    pub env_var: String,
    pub default_models: Vec<Model>,
}

#[derive(Clone)]
struct AnthropicApiKeyAuth {
    env_var: String,
    base_url: String,
    name: String,
}

#[async_trait]
impl ApiKeyAuth for AnthropicApiKeyAuth {
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
                message: format!("Enter your {} API key:", self.name),
                placeholder: Some("sk-ant-...".to_string()),
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
                    .and_then(|_| Some("credential_store".to_string()))
                    .or(Some(format!("env:{}", self.env_var))),
            ),
            None => (None, None),
        };
        Some(AuthResult {
            auth: ModelAuth {
                api_key,
                headers: Some(
                    vec![("anthropic-version".to_string(), "2023-06-01".to_string())]
                        .into_iter()
                        .collect(),
                ),
                base_url: Some(self.base_url.clone()),
            },
            env: None,
            source,
        })
    }
}

struct AnthropicAuthHolder {
    auth: ProviderAuth,
}

pub struct AnthropicProvider {
    id: String,
    name: String,
    config: AnthropicProviderConfig,
    auth_holder: Arc<AnthropicAuthHolder>,
}

impl AnthropicProvider {
    pub fn new(id: &str, name: &str, config: AnthropicProviderConfig) -> Self {
        let auth_impl = AnthropicApiKeyAuth {
            env_var: config.env_var.clone(),
            base_url: config.base_url.clone(),
            name: format!("{} API key", name),
        };
        Self {
            id: id.to_string(),
            name: name.to_string(),
            config,
            auth_holder: Arc::new(AnthropicAuthHolder {
                auth: ProviderAuth {
                    api_key: Some(Box::new(auth_impl)),
                    oauth: None,
                },
            }),
        }
    }
}

pub fn anthropic_provider() -> AnthropicProvider {
    let default_models = vec![
        Model {
            id: "claude-3-5-sonnet-latest".to_string(),
            provider_id: "anthropic".to_string(),
            name: "Claude 3.5 Sonnet".to_string(),
            api: "anthropic-messages".to_string(),
            max_input_tokens: Some(200000),
            max_output_tokens: Some(8192),
            supports_images: true,
            supports_audio: false,
            supports_video: false,
            supports_pdf: true,
            supports_tool_calling: true,
            supports_structured_output: true,
            supports_system_prompt: true,
            thinking: ModelThinkingLevel::None,
            cost_rates: ModelCostRates {
                input_per_million: Some(3.0),
                output_per_million: Some(15.0),
                input_cache_read_per_million: Some(0.30),
                input_cache_write_per_million: Some(3.75),
            },
            context_window: Some(200000),
            default_temperature: Some(1.0),
        },
        Model {
            id: "claude-3-opus-latest".to_string(),
            provider_id: "anthropic".to_string(),
            name: "Claude 3 Opus".to_string(),
            api: "anthropic-messages".to_string(),
            max_input_tokens: Some(200000),
            max_output_tokens: Some(8192),
            supports_images: true,
            supports_audio: false,
            supports_video: false,
            supports_pdf: true,
            supports_tool_calling: true,
            supports_structured_output: true,
            supports_system_prompt: true,
            thinking: ModelThinkingLevel::None,
            cost_rates: ModelCostRates {
                input_per_million: Some(15.0),
                output_per_million: Some(75.0),
                input_cache_read_per_million: Some(1.50),
                input_cache_write_per_million: Some(18.75),
            },
            context_window: Some(200000),
            default_temperature: Some(1.0),
        },
        Model {
            id: "claude-3-7-sonnet-latest".to_string(),
            provider_id: "anthropic".to_string(),
            name: "Claude 3.7 Sonnet".to_string(),
            api: "anthropic-messages".to_string(),
            max_input_tokens: Some(200000),
            max_output_tokens: Some(64000),
            supports_images: true,
            supports_audio: false,
            supports_video: false,
            supports_pdf: true,
            supports_tool_calling: true,
            supports_structured_output: true,
            supports_system_prompt: true,
            thinking: ModelThinkingLevel::High,
            cost_rates: ModelCostRates {
                input_per_million: Some(3.0),
                output_per_million: Some(15.0),
                input_cache_read_per_million: Some(0.30),
                input_cache_write_per_million: Some(3.75),
            },
            context_window: Some(200000),
            default_temperature: Some(1.0),
        },
    ];
    AnthropicProvider::new(
        "anthropic",
        "Anthropic",
        AnthropicProviderConfig {
            base_url: "https://api.anthropic.com".to_string(),
            env_var: "ANTHROPIC_API_KEY".to_string(),
            default_models,
        },
    )
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn auth(&self) -> &ProviderAuth {
        &self.auth_holder.auth
    }

    async fn refresh_models(&self, _ctx: RefreshModelsContext<'_>) -> anyhow::Result<Vec<Model>> {
        Ok(self.config.default_models.clone())
    }

    fn stream(
        &self,
        model: &Model,
        auth: ModelAuth,
        context: Context,
        _options: ApiStreamOptions,
    ) -> std::pin::Pin<
        Box<dyn futures::Stream<Item = AssistantMessageEvent> + Send + 'static>,
    > {
        let model = model.clone();
        let base_url = auth.base_url.unwrap_or_else(|| self.config.base_url.clone());
        let api_key = auth.api_key.unwrap_or_default();
        let headers = auth.headers.unwrap_or_default();
        let request_body = build_anthropic_request(&model, &context, false);

        let stream = async_stream::stream! {
            let client = reqwest::Client::new();
            let url = format!("{}/v1/messages", base_url.trim_end_matches('/'));
            let mut req = client
                .post(&url)
                .header("x-api-key", &api_key)
                .header("Content-Type", "application/json");
            for (k, v) in &headers {
                req = req.header(k, v);
            }
            let res = match req.json(&request_body).send().await {
                Ok(r) => r,
                Err(e) => {
                    yield AssistantMessageEvent::Error { error: e.to_string() };
                    return;
                }
            };
            if !res.status().is_success() {
                let status = res.status();
                let body = res.text().await.unwrap_or_default();
                yield AssistantMessageEvent::Error {
                    error: format!("HTTP {}: {}", status, body),
                };
                return;
            }
            let json: serde_json::Value = match res.json().await {
                Ok(j) => j,
                Err(e) => {
                    yield AssistantMessageEvent::Error { error: e.to_string() };
                    return;
                }
            };
            for evt in from_anthropic_response_json(&model, &json) {
                yield evt;
            }
        };
        Box::pin(stream)
    }

    fn complete(
        &self,
        model: &Model,
        auth: ModelAuth,
        context: Context,
        _options: ApiStreamOptions,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<AssistantMessage>> + Send>>
    {
        let model = model.clone();
        let base_url = auth.base_url.unwrap_or_else(|| self.config.base_url.clone());
        let api_key = auth.api_key.unwrap_or_default();
        let headers = auth.headers.unwrap_or_default();
        let request_body = build_anthropic_request(&model, &context, false);

        Box::pin(async move {
            let client = reqwest::Client::new();
            let url = format!("{}/v1/messages", base_url.trim_end_matches('/'));
            let mut req = client
                .post(&url)
                .header("x-api-key", &api_key)
                .header("Content-Type", "application/json");
            for (k, v) in &headers {
                req = req.header(k, v);
            }
            let res = req.json(&request_body).send().await?;
            if !res.status().is_success() {
                let status = res.status();
                let body = res.text().await?;
                anyhow::bail!("HTTP {}: {}", status, body);
            }
            let json: serde_json::Value = res.json().await?;
            let events = from_anthropic_response_json(&model, &json);
            for evt in events {
                if let AssistantMessageEvent::Done { message, .. } = evt {
                    return Ok(message);
                }
            }
            anyhow::bail!("No done event received")
        })
    }
}

fn build_anthropic_request(
    model: &Model,
    ctx: &Context,
    _stream: bool,
) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": model.id,
        "max_tokens": ctx.max_tokens.unwrap_or(model.max_output_tokens.unwrap_or(4096)),
    });
    if let Some(temp) = ctx.temperature.or(model.default_temperature) {
        body["temperature"] = serde_json::json!(temp);
    }
    if let Some(sys) = &ctx.system_prompt {
        body["system"] = serde_json::json!(sys);
    }

    let mut messages: Vec<serde_json::Value> = vec![];
    let mut last_role: Option<String> = None;

    for msg in &ctx.messages {
        match msg {
            Message::User { content, .. } => {
                let content_arr: Vec<serde_json::Value> = content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } => {
                            Some(serde_json::json!({ "type": "text", "text": text }))
                        }
                        ContentBlock::Image(img) => {
                            let mime = img.mime_type.clone().unwrap_or_else(|| {
                                if img.data.starts_with("data:image/png") {
                                    "image/png".to_string()
                                } else if img.data.starts_with("data:image/jpeg") {
                                    "image/jpeg".to_string()
                                } else {
                                    "image/png".to_string()
                                }
                            });
                            let data = if img.data.starts_with("data:") {
                                img.data.split(',').nth(1).unwrap_or(&img.data).to_string()
                            } else {
                                img.data.clone()
                            };
                            Some(serde_json::json!({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime,
                                    "data": data,
                                }
                            }))
                        }
                        _ => None,
                    })
                    .collect();

                if last_role.as_deref() == Some("user") {
                    messages.push(serde_json::json!({
                        "role": "assistant",
                        "content": [{ "type": "text", "text": " " }]
                    }));
                }
                messages.push(serde_json::json!({
                    "role": "user",
                    "content": content_arr,
                }));
                last_role = Some("user".to_string());
            }
            Message::Assistant { content, .. } => {
                let content_arr: Vec<serde_json::Value> = content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } => {
                            Some(serde_json::json!({ "type": "text", "text": text }))
                        }
                        ContentBlock::ToolCall { id, name, arguments } => {
                            Some(serde_json::json!({
                                "type": "tool_use",
                                "id": id,
                                "name": name,
                                "input": arguments.clone(),
                            }))
                        }
                        ContentBlock::Thinking { thinking, .. } => {
                            Some(serde_json::json!({ "type": "text", "text": thinking }))
                        }
                        _ => None,
                    })
                    .collect();
                if !content_arr.is_empty() {
                    messages.push(serde_json::json!({
                        "role": "assistant",
                        "content": content_arr,
                    }));
                    last_role = Some("assistant".to_string());
                }
            }
            Message::ToolResult { tool_call_id, content, is_error, .. } => {
                let text = crate::types::content_text(content);
                if last_role.as_deref() == Some("user") {
                    messages.push(serde_json::json!({
                        "role": "assistant",
                        "content": [{ "type": "text", "text": " " }]
                    }));
                }
                messages.push(serde_json::json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": text,
                        "is_error": is_error,
                    }]
                }));
                last_role = Some("user".to_string());
            }
            Message::System { content, .. } => {
                let text = crate::types::content_text(content);
                if let Some(existing) = body.get("system").and_then(|s| s.as_str()) {
                    body["system"] = serde_json::json!(format!("{}\n\n{}", existing, text));
                } else {
                    body["system"] = serde_json::json!(text);
                }
            }
        }
    }

    body["messages"] = serde_json::Value::Array(messages);

    if !ctx.tools.is_empty() {
        let anthropic_tools: Vec<serde_json::Value> = ctx
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description.clone().unwrap_or_default(),
                    "input_schema": t.parameters.clone(),
                })
            })
            .collect();
        body["tools"] = serde_json::Value::Array(anthropic_tools);

        if let Some(tc) = &ctx.tool_choice {
            let serialized = match tc {
                crate::types::ToolChoice::Auto => serde_json::json!({ "type": "auto" }),
                crate::types::ToolChoice::None => serde_json::json!({ "type": "auto" }),
                crate::types::ToolChoice::Required => serde_json::json!({ "type": "any" }),
                crate::types::ToolChoice::Tool { name } => {
                    serde_json::json!({ "type": "tool", "name": name })
                }
            };
            body["tool_choice"] = serialized;
        }
    }
    body
}

fn from_anthropic_response_json(
    model: &Model,
    json: &serde_json::Value,
) -> Vec<AssistantMessageEvent> {
    let mut events: Vec<AssistantMessageEvent> = vec![];

    let partial_msg = Message::Assistant {
        content: vec![],
        usage: Usage::default(),
        model: Some(model.id.clone()),
        stop_reason: None,
        timestamp: chrono::Utc::now().timestamp_millis(),
    };
    events.push(AssistantMessageEvent::Start {
        model: model.id.clone(),
        partial: partial_msg,
    });

    let mut content_blocks: Vec<ContentBlock> = vec![];
    let mut usage: Usage = Usage::default();
    let stop_reason = json
        .get("stop_reason")
        .and_then(|s| s.as_str())
        .map(|s| s.to_string());

    if let Some(arr) = json.get("content").and_then(|c| c.as_array()) {
        let mut tool_idx = 0usize;
        for block in arr {
            match block.get("type").and_then(|t| t.as_str()) {
                Some("text") => {
                    let text = block.get("text").and_then(|s| s.as_str()).unwrap_or("").to_string();
                    if !text.is_empty() {
                        events.push(AssistantMessageEvent::TextStart);
                        events.push(AssistantMessageEvent::TextDelta { delta: text.clone() });
                        events.push(AssistantMessageEvent::TextEnd);
                        content_blocks.push(ContentBlock::Text { text });
                    }
                }
                Some("tool_use") => {
                    let id = block.get("id").and_then(|s| s.as_str()).unwrap_or("").to_string();
                    let name = block.get("name").and_then(|s| s.as_str()).unwrap_or("").to_string();
                    let input = block.get("input").cloned().unwrap_or(serde_json::Value::Null);

                    let idx = tool_idx;
                    tool_idx += 1;

                    let partial = Message::Assistant {
                        content: vec![ContentBlock::ToolCall {
                            id: String::new(),
                            name: String::new(),
                            arguments: serde_json::Value::Null,
                        }],
                        usage: Usage::default(),
                        model: Some(model.id.clone()),
                        stop_reason: None,
                        timestamp: chrono::Utc::now().timestamp_millis(),
                    };
                    events.push(AssistantMessageEvent::ToolcallStart {
                        content_index: idx,
                        partial,
                    });
                    let partial2 = Message::Assistant {
                        content: vec![ContentBlock::ToolCall {
                            id: id.clone(),
                            name: name.clone(),
                            arguments: input.clone(),
                        }],
                        usage: Usage::default(),
                        model: Some(model.id.clone()),
                        stop_reason: None,
                        timestamp: chrono::Utc::now().timestamp_millis(),
                    };
                    events.push(AssistantMessageEvent::ToolcallDelta {
                        content_index: idx,
                        partial: partial2,
                    });
                    let tc_block = ContentBlock::ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: input.clone(),
                    };
                    events.push(AssistantMessageEvent::ToolcallEnd {
                        tool_call: tc_block.clone(),
                    });
                    content_blocks.push(tc_block);
                }
                _ => {}
            }
        }
    }

    if let Some(u) = json.get("usage") {
        let input = u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        let output = u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        let cache_write = u
            .get("cache_creation_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cache_read = u
            .get("cache_read_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        usage.input = input;
        usage.output = output;
        usage.cache_read_input = cache_read;
        usage.cache_write_input = cache_write;
        usage.cost = calculate_usage_cost(input, output, cache_read, cache_write, model);
        events.push(AssistantMessageEvent::Usage {
            usage: usage.clone(),
        });
    }

    let msg = Message::Assistant {
        content: content_blocks,
        usage: usage.clone(),
        model: Some(model.id.clone()),
        stop_reason: stop_reason.clone(),
        timestamp: chrono::Utc::now().timestamp_millis(),
    };
    events.push(AssistantMessageEvent::Done {
        reason: stop_reason.unwrap_or_else(|| "unknown".to_string()),
        message: msg,
    });

    events
}
