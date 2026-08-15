use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
pub struct OpenAIProviderConfig {
    pub base_url: String,
    pub env_var: String,
    pub default_models: Vec<Model>,
}

#[derive(Clone)]
struct OpenAIApiKeyAuth {
    env_var: String,
    base_url: String,
    name: String,
}

#[async_trait]
impl ApiKeyAuth for OpenAIApiKeyAuth {
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
                placeholder: Some("sk-...".to_string()),
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
                headers: None,
                base_url: Some(self.base_url.clone()),
            },
            env: None,
            source,
        })
    }
}

struct OpenAIAuthHolder {
    auth: ProviderAuth,
}

pub struct OpenAIProvider {
    id: String,
    name: String,
    config: OpenAIProviderConfig,
    auth_holder: Arc<OpenAIAuthHolder>,
}

impl OpenAIProvider {
    pub fn new(id: &str, name: &str, config: OpenAIProviderConfig) -> Self {
        let auth_impl = OpenAIApiKeyAuth {
            env_var: config.env_var.clone(),
            base_url: config.base_url.clone(),
            name: format!("{} API key", name),
        };
        Self {
            id: id.to_string(),
            name: name.to_string(),
            config,
            auth_holder: Arc::new(OpenAIAuthHolder {
                auth: ProviderAuth {
                    api_key: Some(Box::new(auth_impl)),
                    oauth: None,
                },
            }),
        }
    }
}

pub fn openai_provider() -> OpenAIProvider {
    let default_models = vec![
        Model {
            id: "gpt-4o-mini".to_string(),
            provider_id: "openai".to_string(),
            name: "GPT-4o Mini".to_string(),
            api: "openai-chat-completions".to_string(),
            max_input_tokens: Some(128000),
            max_output_tokens: Some(16384),
            supports_images: true,
            supports_audio: true,
            supports_video: false,
            supports_pdf: false,
            supports_tool_calling: true,
            supports_structured_output: true,
            supports_system_prompt: true,
            thinking: ModelThinkingLevel::None,
            cost_rates: ModelCostRates {
                input_per_million: Some(0.15),
                output_per_million: Some(0.60),
                ..Default::default()
            },
            context_window: Some(128000),
            default_temperature: Some(1.0),
        },
        Model {
            id: "gpt-4o".to_string(),
            provider_id: "openai".to_string(),
            name: "GPT-4o".to_string(),
            api: "openai-chat-completions".to_string(),
            max_input_tokens: Some(128000),
            max_output_tokens: Some(16384),
            supports_images: true,
            supports_audio: true,
            supports_video: true,
            supports_pdf: false,
            supports_tool_calling: true,
            supports_structured_output: true,
            supports_system_prompt: true,
            thinking: ModelThinkingLevel::None,
            cost_rates: ModelCostRates {
                input_per_million: Some(2.50),
                output_per_million: Some(10.0),
                ..Default::default()
            },
            context_window: Some(128000),
            default_temperature: Some(1.0),
        },
    ];
    OpenAIProvider::new(
        "openai",
        "OpenAI",
        OpenAIProviderConfig {
            base_url: "https://api.openai.com/v1".to_string(),
            env_var: "OPENAI_API_KEY".to_string(),
            default_models,
        },
    )
}

#[async_trait]
impl Provider for OpenAIProvider {
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
        let (request_body, _) = build_openai_request(&model, &context, false);

        let stream = async_stream::stream! {
            let client = reqwest::Client::new();
            let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));

            let res = match client
                .post(&url)
                .bearer_auth(&api_key)
                .header("Content-Type", "application/json")
                .json(&request_body)
                .send()
                .await
            {
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

            for evt in from_openai_response_json(&model, &json) {
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
        let (request_body, _) = build_openai_request(&model, &context, false);

        Box::pin(async move {
            let client = reqwest::Client::new();
            let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
            let res = client
                .post(&url)
                .bearer_auth(&api_key)
                .header("Content-Type", "application/json")
                .json(&request_body)
                .send()
                .await?;
            if !res.status().is_success() {
                let status = res.status();
                let body = res.text().await?;
                anyhow::bail!("HTTP {}: {}", status, body);
            }
            let json: serde_json::Value = res.json().await?;
            let events = from_openai_response_json(&model, &json);
            for evt in events {
                if let AssistantMessageEvent::Done { message, .. } = evt {
                    return Ok(message);
                }
            }
            anyhow::bail!("No done event received")
        })
    }
}

fn build_openai_request(
    model: &Model,
    ctx: &Context,
    stream: bool,
) -> (serde_json::Value, HashMap<String, usize>) {
    let mut body = serde_json::json!({
        "model": model.id,
        "stream": stream,
    });
    if let Some(temp) = ctx.temperature.or(model.default_temperature) {
        body["temperature"] = serde_json::json!(temp);
    }
    if let Some(max) = ctx.max_tokens {
        body["max_tokens"] = serde_json::json!(max);
    }
    if let Some(fmt) = &ctx.response_format {
        body["response_format"] = serde_json::to_value(fmt).unwrap_or(serde_json::Value::Null);
    }

    let mut messages: Vec<serde_json::Value> = vec![];
    if let Some(sys) = &ctx.system_prompt {
        messages.push(serde_json::json!({
            "role": "system",
            "content": sys,
        }));
    }

    for msg in ctx.messages.iter() {
        match msg {
            Message::User { content, .. } => {
                let parts: Vec<serde_json::Value> = content
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
                            let url = if img.data.starts_with("data:") {
                                img.data.clone()
                            } else {
                                format!("data:{};base64,{}", mime, img.data)
                            };
                            Some(serde_json::json!({
                                "type": "image_url",
                                "image_url": { "url": url }
                            }))
                        }
                        _ => None,
                    })
                    .collect();
                let content_val = if parts.len() == 1 {
                    if let Some(text) = parts[0].get("text").and_then(|t| t.as_str()) {
                        serde_json::Value::String(text.to_string())
                    } else {
                        serde_json::Value::Array(parts)
                    }
                } else {
                    serde_json::Value::Array(parts)
                };
                messages.push(serde_json::json!({ "role": "user", "content": content_val }));
            }
            Message::Assistant { content, .. } => {
                let mut oai_content: Vec<serde_json::Value> = vec![];
                let mut tool_calls: Vec<serde_json::Value> = vec![];
                for block in content {
                    match block {
                        ContentBlock::Text { text } => oai_content.push(serde_json::json!({
                            "type": "text",
                            "text": text,
                        })),
                        ContentBlock::ToolCall { id, name, arguments } => {
                            tool_calls.push(serde_json::json!({
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": serde_json::to_string(arguments).unwrap_or_default(),
                                }
                            }));
                        }
                        _ => {}
                    }
                }
                let mut msg_obj = serde_json::json!({ "role": "assistant" });
                if !oai_content.is_empty() {
                    let cval = if oai_content.len() == 1 {
                        oai_content[0].get("text").cloned().unwrap_or_else(|| {
                            serde_json::Value::Array(oai_content.clone())
                        })
                    } else {
                        serde_json::Value::Array(oai_content)
                    };
                    msg_obj["content"] = cval;
                } else {
                    msg_obj["content"] = serde_json::Value::Null;
                }
                if !tool_calls.is_empty() {
                    msg_obj["tool_calls"] = serde_json::Value::Array(tool_calls);
                }
                messages.push(msg_obj);
            }
            Message::ToolResult { tool_call_id, content, .. } => {
                let text = crate::types::content_text(content);
                messages.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": text,
                }));
            }
            Message::System { content, .. } => {
                let text = crate::types::content_text(content);
                messages.push(serde_json::json!({
                    "role": "system",
                    "content": text,
                }));
            }
        }
    }
    body["messages"] = serde_json::Value::Array(messages);

    if !ctx.tools.is_empty() {
        let oai_tools: Vec<serde_json::Value> = ctx
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description.clone().unwrap_or_default(),
                        "parameters": t.parameters.clone(),
                    }
                })
            })
            .collect();
        body["tools"] = serde_json::Value::Array(oai_tools);
    }
    (body, HashMap::new())
}

fn from_openai_response_json(
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
    let mut stop_reason: Option<String> = None;

    if let Some(choice) = json
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
    {
        stop_reason = choice
            .get("finish_reason")
            .and_then(|s| s.as_str())
            .map(|s| s.to_string());
        if let Some(msg) = choice.get("message") {
            if let Some(text) = msg.get("content").and_then(|c| c.as_str()) {
                if !text.is_empty() {
                    events.push(AssistantMessageEvent::TextStart);
                    events.push(AssistantMessageEvent::TextDelta {
                        delta: text.to_string(),
                    });
                    events.push(AssistantMessageEvent::TextEnd);
                    content_blocks.push(ContentBlock::Text {
                        text: text.to_string(),
                    });
                }
            }
            if let Some(tcs) = msg.get("tool_calls").and_then(|t| t.as_array()) {
                for (idx, tc) in tcs.iter().enumerate() {
                    let id = tc
                        .get("id")
                        .and_then(|s| s.as_str())
                        .unwrap_or("")
                        .to_string();
                    let func = tc.get("function").unwrap_or(&serde_json::Value::Null);
                    let name = func
                        .get("name")
                        .and_then(|s| s.as_str())
                        .unwrap_or("")
                        .to_string();
                    let args_str = func
                        .get("arguments")
                        .and_then(|s| s.as_str())
                        .unwrap_or("{}");
                    let arguments: serde_json::Value =
                        serde_json::from_str(args_str).unwrap_or(serde_json::Value::Null);

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
                            arguments: arguments.clone(),
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
                        arguments: arguments.clone(),
                    };
                    events.push(AssistantMessageEvent::ToolcallEnd {
                        tool_call: tc_block.clone(),
                    });
                    content_blocks.push(tc_block);
                }
            }
        }
    }

    if let Some(u) = json.get("usage") {
        let prompt = u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        let comp = u.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        let cache_read = u
            .get("prompt_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        usage.input = prompt;
        usage.output = comp;
        usage.cache_read_input = cache_read;
        usage.cost = calculate_usage_cost(prompt, comp, cache_read, 0, model);
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
