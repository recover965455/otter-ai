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
    /// Reasoning tokens reported via `output_tokens_details.reasoning_tokens`.
    #[serde(default)]
    pub reasoning: u64,
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
    /// Provider-specific thinking-level remapping (TS: `thinkingLevelMap`),
    /// e.g. `{ "minimal": "low", "xhigh": "xhigh" }` for Codex models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
    /// TS: `constrainedSampling` — grammar constrained sampling config used
    /// by Codex models; grammar tools stream as `custom_tool_call` items.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constrained_sampling: Option<ToolConstrainedSampling>,
}

/// TS: `ToolConstrainedSampling` (only `type: "grammar"` is honoured).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolConstrainedSampling {
    #[serde(rename = "type")]
    pub sampling_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variants: Option<ToolGrammarVariants>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolGrammarVariants {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai_lark: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai_regex: Option<String>,
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
        /// Encoded message identity for OpenAI Responses replay
        /// (TS: `textSignature`, `{"v":1,"id":"msg_…"}` shape). Lets the
        /// assistant message round-trip with its server-assigned item id so
        /// multi-turn prompt-cache affinity and websocket delta
        /// continuations keep matching prefixes.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text_signature: Option<String>,
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
        /// Codex `response.end_turn` flag (TS: `endTurn`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        end_turn: Option<bool>,
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
            content: vec![ContentBlock::Text { text: text.into(), text_signature: None}],
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
            end_turn: None,
            timestamp: default_timestamp(),
        }
    }

    pub fn with_content(mut self, content: Vec<ContentBlock>) -> Self {
        if let Message::Assistant {
            content: ref mut c, ..
        } = self
        {
            *c = content;
        }
        self
    }

    pub fn with_model(mut self, model: Option<String>) -> Self {
        if let Message::Assistant {
            model: ref mut m, ..
        } = self
        {
            *m = model;
        }
        self
    }

    pub fn with_usage(mut self, usage: Usage) -> Self {
        if let Message::Assistant {
            usage: ref mut u, ..
        } = self
        {
            *u = usage;
        }
        self
    }

    pub fn with_stop_reason(mut self, reason: Option<String>) -> Self {
        if let Message::Assistant {
            stop_reason: ref mut sr,
            ..
        } = self
        {
            *sr = reason;
        }
        self
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
    /// Per-request base URL override (TS: provider config / stream options
    /// `baseUrl`). When `None`, the provider's configured base URL is used.
    pub base_url: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ApiStreamOptions {
    pub signal: Option<CancellationToken>,
    pub request_options: ProviderRequestOptions,
    /// Maximum number of retries after rate limits / transient errors
    /// (TS: `maxRetries`; Codex default 0).
    pub max_retries: Option<u32>,
    /// Upper bound for server-requested retry delays (TS: `maxRetryDelayMs`;
    /// Codex default 60 000 ms; `Some(0)` disables the cap).
    pub max_retry_delay_ms: Option<u64>,
    /// HTTP response-header timeout (TS: `timeoutMs`).
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Default)]
pub struct SimpleStreamOptions {
    pub signal: Option<CancellationToken>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u64>,
    pub response_format: Option<ResponseFormat>,
    pub tool_choice: Option<ToolChoice>,
    pub thinking: Option<ModelThinkingLevel>,
    /// Raw reasoning effort override (TS: `reasoning`), e.g. `"minimal"`,
    /// `"xhigh"`, `"max"` or `"off"`.
    pub reasoning: Option<String>,
    pub provider_extra: Option<serde_json::Value>,
    pub session_id: Option<String>,
    pub cache_retention: Option<CacheRetention>,
    /// Per-request base URL override routed through to the provider adapter.
    pub base_url: Option<String>,
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
            ContentBlock::Text { text, .. } => Some(text.clone()),
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

    /// Future that resolves once the token is cancelled (for `select!`).
    pub async fn cancelled_fut(&self) {
        while !self.is_cancelled() {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    }
}
