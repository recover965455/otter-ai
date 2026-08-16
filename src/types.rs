use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

pub type Api = String;
pub type ProviderId = String;
pub type ProviderHeaders = HashMap<String, String>;
pub type ProviderEnv = HashMap<String, String>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum KnownApi {
    OpenaiCompletions,
    MistralConversations,
    OpenaiResponses,
    AzureOpenaiResponses,
    OpenaiCodexResponses,
    AnthropicMessages,
    BedrockConverseStream,
    GoogleGenerativeAi,
    GoogleVertex,
    PiMessages,
}

// ---- Cost tier (TS: model.cost.tiers) ----
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CostTier {
    pub input_tokens_above: u64,
    pub input_per_million: f64,
    pub output_per_million: f64,
    pub cache_read_per_million: f64,
    pub cache_write_per_million: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ModelCostRates {
    pub input_per_million: Option<f64>,
    pub output_per_million: Option<f64>,
    pub input_cache_read_per_million: Option<f64>,
    pub input_cache_write_per_million: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tiers: Vec<CostTier>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct UsageCost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub total: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub total_tokens: u64,
    pub cost: UsageCost,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum ModelThinkingLevel {
    #[default]
    None,
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Model {
    pub id: String,
    pub provider_id: ProviderId,
    pub name: String,
    pub api: Api,
    pub max_input_tokens: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub supports_images: bool,
    pub supports_audio: bool,
    pub supports_video: bool,
    pub supports_pdf: bool,
    pub supports_tool_calling: bool,
    pub supports_structured_output: bool,
    pub supports_system_prompt: bool,
    pub thinking: ModelThinkingLevel,
    pub reasoning: bool,
    pub cost_rates: ModelCostRates,
    pub context_window: Option<u64>,
    pub default_temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub data: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    ToolCall {
        id: String,
        name: String,
        #[serde(default)]
        arguments: serde_json::Value,
    },
    ToolResult {
        tool_call_id: String,
        tool_name: String,
        content: Vec<ContentBlock>,
        #[serde(default)]
        is_error: bool,
    },
    Image(ImageContent),
}

fn default_timestamp() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "role", rename_all = "camelCase")]
pub enum Message {
    #[serde(rename = "user")]
    User {
        content: Vec<ContentBlock>,
        #[serde(default = "default_timestamp")]
        timestamp: i64,
    },
    #[serde(rename = "assistant")]
    Assistant {
        content: Vec<ContentBlock>,
        api: Api,
        provider: ProviderId,
        model: Option<String>,
        #[serde(default)]
        usage: Usage,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stop_reason: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error_message: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        response_id: Option<String>,
        #[serde(default = "default_timestamp")]
        timestamp: i64,
    },
    #[serde(rename = "toolResult")]
    ToolResult {
        tool_call_id: String,
        tool_name: String,
        content: Vec<ContentBlock>,
        #[serde(default)]
        is_error: bool,
        #[serde(default = "default_timestamp")]
        timestamp: i64,
    },
    #[serde(rename = "system")]
    System {
        content: Vec<ContentBlock>,
        #[serde(default = "default_timestamp")]
        timestamp: i64,
    },
}

// Allow constructing a Message from a simple string (TS: user content can be string)
impl Message {
    pub fn user_from_string<S: Into<String>>(text: S) -> Self {
        Message::User {
            content: vec![ContentBlock::Text { text: text.into() }],
            timestamp: default_timestamp(),
        }
    }

    pub fn assistant_default(api: Api, provider: ProviderId) -> Self {
        Message::Assistant {
            content: vec![],
            api,
            provider,
            model: None,
            usage: Usage::default(),
            stop_reason: None,
            error_message: None,
            response_id: None,
            timestamp: default_timestamp(),
        }
    }

    pub fn with_error_message<S: Into<String>>(mut self, msg: S) -> Self {
        if let Message::Assistant {
            ref mut error_message,
            ..
        } = self
        {
            *error_message = Some(msg.into());
        }
        self
    }

    pub fn stop_reason(&self) -> Option<&str> {
        match self {
            Message::Assistant { stop_reason, .. } => stop_reason.as_deref(),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Context {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub tools: Vec<Tool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Tool { name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JsonSchemaFormat {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>,
    #[serde(default)]
    pub strict: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema(JsonSchemaFormat),
}

#[derive(Debug, Clone, Default)]
pub struct ProviderRequestOptions {
    pub headers: ProviderHeaders,
    pub extra_body: Option<serde_json::Value>,
    pub query_params: HashMap<String, String>,
}

#[derive(Debug, Clone, Default)]
pub struct ApiStreamOptions {
    pub signal: Option<CancellationToken>,
    pub request_options: ProviderRequestOptions,
}

#[derive(Debug, Clone, Default)]
pub struct SimpleStreamOptions {
    pub signal: Option<CancellationToken>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u64>,
    pub response_format: Option<ResponseFormat>,
    pub tool_choice: Option<ToolChoice>,
    pub thinking: Option<ModelThinkingLevel>,
    pub provider_extra: Option<serde_json::Value>,
    pub session_id: Option<String>,
    pub cache_retention: Option<CacheRetention>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "camelCase")]
pub enum CacheRetention {
    #[default]
    None,
    Short,
    Long,
}

pub type AssistantMessage = Message;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum AssistantMessageEvent {
    Start {
        partial: AssistantMessage,
    },
    TextStart,
    TextDelta {
        delta: String,
    },
    TextEnd,
    ThinkingStart,
    ThinkingDelta {
        delta: String,
    },
    ThinkingEnd {
        signature: Option<String>,
    },
    ToolcallStart {
        content_index: usize,
        id: Option<String>,
        name: Option<String>,
    },
    ToolcallDelta {
        content_index: usize,
        delta: String,
    },
    ToolcallEnd {
        content_index: usize,
        tool_call: ContentBlock,
    },
    Usage {
        usage: Usage,
    },
    Done {
        reason: String,
        message: AssistantMessage,
    },
    Error {
        reason: String,
        error: String,
    },
}

pub fn content_text(content: &[ContentBlock]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn uuidv7() -> String {
    Uuid::now_v7().to_string()
}

// Custom CancellationToken replacement (simplified) — replaces tokio::sync::CancellationToken
// which has feature-gating inconsistencies across tokio versions.
#[derive(Debug, Default, Clone)]
pub struct CancellationToken {
    inner: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            inner: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    pub fn cancel(&self) {
        self.inner.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.load(std::sync::atomic::Ordering::SeqCst)
    }

    pub fn child_token(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}
