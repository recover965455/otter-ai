use async_trait::async_trait;
use std::sync::Arc;

use super::{Provider, RefreshModelsContext};
use crate::auth::types::{
    ApiKeyAuth, AuthContext, AuthInteraction, AuthResult, ModelAuth, ProviderAuth,
};
use crate::types::{
    ApiStreamOptions, AssistantMessage, AssistantMessageEvent, Context, CancellationToken,
    ContentBlock, Message, Model, ModelCostRates, ModelThinkingLevel, Usage,
};

#[derive(Debug, Clone)]
pub struct FauxResponseConfig {
    pub text: String,
    pub tool_calls: Vec<(String, serde_json::Value)>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub stop_reason: String,
    pub thinking: Option<String>,
    pub stream_delay_ms: u64,
    pub error: Option<String>,
}

impl Default for FauxResponseConfig {
    fn default() -> Self {
        Self {
            text: "Hello from Faux provider!".to_string(),
            tool_calls: vec![],
            input_tokens: 0,
            output_tokens: 0,
            stop_reason: "end_turn".to_string(),
            thinking: None,
            stream_delay_ms: 2,
            error: None,
        }
    }
}

struct FauxApiKeyAuth;

#[async_trait]
impl ApiKeyAuth for FauxApiKeyAuth {
    fn name(&self) -> &str {
        "Faux API key"
    }

    async fn login(
        &self,
        _interaction: &(dyn AuthInteraction + Send + Sync),
    ) -> anyhow::Result<crate::auth::types::ApiKeyCredential> {
        Ok(crate::auth::types::ApiKeyCredential {
            r#type: "api_key".to_string(),
            key: Some("faux-test-key".to_string()),
            env: None,
        })
    }

    async fn resolve(
        &self,
        _ctx: &(dyn AuthContext + Send + Sync),
        credential: Option<&crate::auth::types::ApiKeyCredential>,
        _signal: &CancellationToken,
    ) -> Option<AuthResult> {
        Some(AuthResult {
            auth: ModelAuth {
                api_key: credential.and_then(|c| c.key.clone()),
                headers: None,
                base_url: Some("http://localhost/faux".to_string()),
            },
            env: None,
            source: Some("faux-in-memory".to_string()),
        })
    }
}

struct FauxAuthHolder {
    auth: ProviderAuth,
}

impl Default for FauxAuthHolder {
    fn default() -> Self {
        Self {
            auth: ProviderAuth {
                api_key: Some(Box::new(FauxApiKeyAuth)),
                oauth: None,
            },
        }
    }
}

pub struct FauxProvider {
    models: Vec<Model>,
    responses: std::sync::Mutex<Vec<FauxResponseConfig>>,
    auth_holder: Arc<FauxAuthHolder>,
}

impl Default for FauxProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl FauxProvider {
    pub fn new() -> Self {
        let models = vec![
            Model {
                id: "faux-mini".to_string(),
                provider_id: "faux".to_string(),
                name: "Faux Mini".to_string(),
                api: "faux-api".to_string(),
                max_input_tokens: Some(128000),
                max_output_tokens: Some(16384),
                supports_images: false,
                supports_audio: false,
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
                default_temperature: Some(0.7),
            },
            Model {
                id: "faux-pro".to_string(),
                provider_id: "faux".to_string(),
                name: "Faux Pro".to_string(),
                api: "faux-api".to_string(),
                max_input_tokens: Some(200000),
                max_output_tokens: Some(65536),
                supports_images: true,
                supports_audio: false,
                supports_video: false,
                supports_pdf: false,
                supports_tool_calling: true,
                supports_structured_output: true,
                supports_system_prompt: true,
                thinking: ModelThinkingLevel::Medium,
                cost_rates: ModelCostRates {
                    input_per_million: Some(3.0),
                    output_per_million: Some(15.0),
                    ..Default::default()
                },
                context_window: Some(200000),
                default_temperature: Some(0.7),
            },
        ];
        Self {
            models,
            responses: std::sync::Mutex::new(vec![]),
            auth_holder: Arc::new(FauxAuthHolder::default()),
        }
    }

    pub fn enqueue_response(&self, config: FauxResponseConfig) {
        self.responses.lock().unwrap().push(config);
    }

    fn next_response(&self) -> FauxResponseConfig {
        let mut guard = self.responses.lock().unwrap();
        if !guard.is_empty() {
            guard.remove(0)
        } else {
            FauxResponseConfig::default()
        }
    }

    fn build_assistant_message(
        &self,
        model: &Model,
        config: &FauxResponseConfig,
    ) -> AssistantMessage {
        use crate::utils::validation::calculate_usage_cost;

        let mut content: Vec<ContentBlock> = vec![];

        if let Some(thinking) = &config.thinking {
            content.push(ContentBlock::Thinking {
                thinking: thinking.clone(),
                signature: None,
            });
        }

        if !config.text.is_empty() {
            content.push(ContentBlock::Text {
                text: config.text.clone(),
            });
        }

        for (name, args) in &config.tool_calls {
            content.push(ContentBlock::ToolCall {
                id: crate::types::uuidv7(),
                name: name.clone(),
                arguments: args.clone(),
            });
        }

        let cost = calculate_usage_cost(
            config.input_tokens,
            config.output_tokens,
            0,
            0,
            model,
        );

        Message::Assistant {
            content,
            usage: Usage {
                input: config.input_tokens,
                output: config.output_tokens,
                cache_read_input: 0,
                cache_write_input: 0,
                cost,
            },
            model: Some(model.id.clone()),
            stop_reason: Some(config.stop_reason.clone()),
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }
}

#[async_trait]
impl Provider for FauxProvider {
    fn id(&self) -> &str {
        "faux"
    }

    fn name(&self) -> &str {
        "Faux (Test Provider)"
    }

    fn auth(&self) -> &ProviderAuth {
        &self.auth_holder.auth
    }

    async fn refresh_models(&self, _ctx: RefreshModelsContext<'_>) -> anyhow::Result<Vec<Model>> {
        Ok(self.models.clone())
    }

    fn stream(
        &self,
        model: &Model,
        _auth: ModelAuth,
        _context: Context,
        _options: ApiStreamOptions,
    ) -> std::pin::Pin<
        Box<dyn futures::Stream<Item = AssistantMessageEvent> + Send + 'static>,
    > {
        let config = self.next_response();
        let model = model.clone();
        let provider_cloned = self.shallow_clone();

        let stream = async_stream::stream! {
            if let Some(err) = &config.error {
                yield AssistantMessageEvent::Error { error: err.clone() };
                return;
            }

            let partial_msg = Message::Assistant {
                content: vec![],
                usage: Usage::default(),
                model: Some(model.id.clone()),
                stop_reason: None,
                timestamp: chrono::Utc::now().timestamp_millis(),
            };
            yield AssistantMessageEvent::Start {
                model: model.id.clone(),
                partial: partial_msg,
            };

            if let Some(thinking) = &config.thinking {
                yield AssistantMessageEvent::ThinkingStart;
                for chunk in thinking.chars().collect::<Vec<char>>().chunks(3) {
                    tokio::time::sleep(tokio::time::Duration::from_millis(config.stream_delay_ms)).await;
                    yield AssistantMessageEvent::ThinkingDelta { delta: chunk.iter().collect() };
                }
                yield AssistantMessageEvent::ThinkingEnd { signature: None };
            }

            if !config.text.is_empty() {
                yield AssistantMessageEvent::TextStart;
                for chunk in config.text.chars().collect::<Vec<char>>().chunks(4) {
                    tokio::time::sleep(tokio::time::Duration::from_millis(config.stream_delay_ms)).await;
                    yield AssistantMessageEvent::TextDelta { delta: chunk.iter().collect() };
                }
                yield AssistantMessageEvent::TextEnd;
            }

            for (idx, (name, args)) in config.tool_calls.iter().enumerate() {
                let partial_msg = Message::Assistant {
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
                yield AssistantMessageEvent::ToolcallStart {
                    content_index: idx,
                    partial: partial_msg,
                };

                let id = crate::types::uuidv7();

                let partial = Message::Assistant {
                    content: vec![ContentBlock::ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: serde_json::Value::Null,
                    }],
                    usage: Usage::default(),
                    model: Some(model.id.clone()),
                    stop_reason: None,
                    timestamp: chrono::Utc::now().timestamp_millis(),
                };
                yield AssistantMessageEvent::ToolcallDelta {
                    content_index: idx,
                    partial,
                };

                let tc = ContentBlock::ToolCall {
                    id,
                    name: name.clone(),
                    arguments: args.clone(),
                };
                yield AssistantMessageEvent::ToolcallEnd { tool_call: tc };
            }

            let final_msg = provider_cloned.build_assistant_message(&model, &config);
            if let Message::Assistant { usage, .. } = &final_msg {
                yield AssistantMessageEvent::Usage { usage: usage.clone() };
            }

            yield AssistantMessageEvent::Done {
                reason: config.stop_reason.clone(),
                message: final_msg,
            };
        };

        Box::pin(stream)
    }

    fn complete(
        &self,
        model: &Model,
        _auth: ModelAuth,
        _context: Context,
        _options: ApiStreamOptions,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<AssistantMessage>> + Send>>
    {
        let config = self.next_response();
        let model = model.clone();
        let provider_cloned = self.shallow_clone();
        Box::pin(async move {
            if let Some(err) = config.error {
                anyhow::bail!(err);
            }
            Ok(provider_cloned.build_assistant_message(&model, &config))
        })
    }
}

impl FauxProvider {
    fn shallow_clone(&self) -> FauxProvider {
        FauxProvider {
            models: self.models.clone(),
            responses: std::sync::Mutex::new(
                self.responses.lock().unwrap().clone(),
            ),
            auth_holder: self.auth_holder.clone(),
        }
    }
}

pub fn faux_provider() -> FauxProvider {
    FauxProvider::new()
}
