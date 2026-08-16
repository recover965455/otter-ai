//! Faux test provider: fully mirrors @earendil-works/pi-ai's faux-provider implementation.
//!
//! Exposes:
//! * Helper constructors: `faux_text`, `faux_thinking`, `faux_tool_call`, `faux_assistant_message`
//! * Two registration modes: `faux_provider()` (use with explicit `Models`), and a
//!   standalone handle `FauxRegistration` produced by `register_faux_provider()` that mimics
//!   the compat API surface used in the faux-provider.test.ts suite.
//! * Full state/queue lifecycle, serialized-context usage estimation, session-scoped
//!   prompt cache simulation, deterministic token-sized delta streaming, and signal-based
//!   abort behavior.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use rand::Rng;

use crate::auth::types::{
    ApiKeyAuth, ApiKeyCredential, AuthCheck, AuthContext, AuthInteraction, AuthResult, ProviderAuth,
};
use crate::providers::{Provider, RefreshModelsContext};
use crate::types::{
    ApiStreamOptions, AssistantMessage, AssistantMessageEvent, CacheRetention, CancellationToken,
    ContentBlock, Context, Message, Model, ModelThinkingLevel, SimpleStreamOptions, Usage,
    UsageCost,
};
use crate::utils::event_stream::{
    create_assistant_message_event_stream, AssistantMessageEventStream,
};

// ---------- constants ----------
const DEFAULT_API: &str = "faux";
const DEFAULT_PROVIDER: &str = "faux";
const DEFAULT_MODEL_ID: &str = "faux-1";
const DEFAULT_MODEL_NAME: &str = "Faux Model";
const DEFAULT_MIN_TOKEN_SIZE: usize = 3;
const DEFAULT_MAX_TOKEN_SIZE: usize = 5;

fn default_usage() -> Usage {
    Usage {
        input: 0,
        output: 0,
        cache_read: 0,
        cache_write: 0,
        total_tokens: 0,
        cost: UsageCost::default(),
    }
}

// ---------- public helper types/functions (TS: fauxText, fauxThinking, fauxToolCall, fauxAssistantMessage) ----------
pub fn faux_text(text: impl Into<String>) -> ContentBlock {
    ContentBlock::Text { text: text.into() }
}

pub fn faux_thinking(thinking: impl Into<String>) -> ContentBlock {
    ContentBlock::Thinking {
        thinking: thinking.into(),
        signature: None,
    }
}

pub fn faux_tool_call(
    name: impl Into<String>,
    arguments: serde_json::Value,
    options: FauxToolCallOptions,
) -> ContentBlock {
    let id = options.id.unwrap_or_else(|| random_id("tool"));
    ContentBlock::ToolCall {
        id,
        name: name.into(),
        arguments,
    }
}

#[derive(Debug, Clone, Default)]
pub struct FauxToolCallOptions {
    pub id: Option<String>,
}

pub type FauxContentBlock = ContentBlock;

fn normalize_content(content: impl Into<FauxContentInput>) -> Vec<ContentBlock> {
    match content.into() {
        FauxContentInput::String(s) => vec![faux_text(s)],
        FauxContentInput::Block(b) => vec![b],
        FauxContentInput::Blocks(bs) => bs,
    }
}

pub enum FauxContentInput {
    String(String),
    Block(ContentBlock),
    Blocks(Vec<ContentBlock>),
}

impl From<String> for FauxContentInput {
    fn from(s: String) -> Self {
        FauxContentInput::String(s)
    }
}
impl From<&str> for FauxContentInput {
    fn from(s: &str) -> Self {
        FauxContentInput::String(s.into())
    }
}
impl From<ContentBlock> for FauxContentInput {
    fn from(b: ContentBlock) -> Self {
        FauxContentInput::Block(b)
    }
}
impl From<Vec<ContentBlock>> for FauxContentInput {
    fn from(b: Vec<ContentBlock>) -> Self {
        FauxContentInput::Blocks(b)
    }
}

#[derive(Debug, Clone, Default)]
pub struct FauxAssistantMessageOptions {
    pub stop_reason: Option<String>,
    pub error_message: Option<String>,
    pub response_id: Option<String>,
    pub timestamp: Option<i64>,
}

pub fn faux_assistant_message(
    content: impl Into<FauxContentInput>,
    options: FauxAssistantMessageOptions,
) -> AssistantMessage {
    let ts = options
        .timestamp
        .unwrap_or_else(|| chrono::Utc::now().timestamp_millis());
    let stop_reason = options.stop_reason.unwrap_or_else(|| "stop".to_string());
    Message::Assistant {
        content: normalize_content(content),
        api: DEFAULT_API.to_string(),
        provider: DEFAULT_PROVIDER.to_string(),
        model: Some(DEFAULT_MODEL_ID.to_string()),
        usage: default_usage(),
        stop_reason: Some(stop_reason),
        error_message: options.error_message,
        response_id: options.response_id,
        timestamp: ts,
    }
}

// ---------- state & response types ----------
#[derive(Debug, Clone, Default)]
pub struct FauxProviderState {
    pub call_count: u64,
    pub deferred_fetch_count: u64,
}

pub type FauxResponseFactory = Arc<
    dyn Fn(
            Context,
            Option<SimpleStreamOptions>,
            FauxProviderState,
            Model,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = AssistantMessage> + Send>>
        + Send
        + Sync,
>;

#[allow(clippy::large_enum_variant)]
pub enum FauxResponseStep {
    Message(AssistantMessage),
    Factory(FauxResponseFactory),
}

#[derive(Debug, Clone)]
pub struct FauxModelDefinition {
    pub id: String,
    pub name: Option<String>,
    pub reasoning: bool,
    pub context_window: Option<u64>,
    pub max_tokens: Option<u64>,
    pub supports_images: bool,
    pub cost_rates: crate::types::ModelCostRates,
}

impl Default for FauxModelDefinition {
    fn default() -> Self {
        Self {
            id: DEFAULT_MODEL_ID.into(),
            name: Some(DEFAULT_MODEL_NAME.into()),
            reasoning: false,
            context_window: Some(128_000),
            max_tokens: Some(16_384),
            supports_images: true,
            cost_rates: crate::types::ModelCostRates::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RegisterFauxProviderOptions {
    pub api: Option<String>,
    pub provider: Option<String>,
    pub models: Vec<FauxModelDefinition>,
    pub tokens_per_second: Option<u64>,
    pub token_size: Option<(usize, usize)>, // (min, max)
}

impl Default for RegisterFauxProviderOptions {
    fn default() -> Self {
        Self {
            api: None,
            provider: None,
            models: vec![FauxModelDefinition::default()],
            tokens_per_second: None,
            token_size: None,
        }
    }
}

// ---------- id / estimation helpers ----------
fn random_id(prefix: &str) -> String {
    use rand::RngCore;
    let mut buf = [0u8; 8];
    rand::thread_rng().fill_bytes(&mut buf);
    format!(
        "{}:{}:{}",
        prefix,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
        hex::encode(&buf[..4])
    )
}

fn estimate_tokens(text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }
    ((text.len() as f64) / 4.0).ceil() as u64
}

fn content_to_text(content: &[ContentBlock]) -> String {
    content
        .iter()
        .map(|b| match b {
            ContentBlock::Text { text } => text.clone(),
            ContentBlock::Image(img) => {
                format!(
                    "[image:{}:{}]",
                    img.mime_type
                        .as_deref()
                        .unwrap_or("application/octet-stream"),
                    img.data.len()
                )
            }
            _ => String::new(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn assistant_content_to_text(content: &[ContentBlock]) -> String {
    content
        .iter()
        .map(|b| match b {
            ContentBlock::Text { text } => text.clone(),
            ContentBlock::Thinking { thinking, .. } => thinking.clone(),
            ContentBlock::ToolCall {
                name, arguments, ..
            } => {
                format!("{}:{}", name, arguments)
            }
            _ => String::new(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn message_to_text(msg: &Message) -> String {
    match msg {
        Message::User { content, .. } => content_to_text(content),
        Message::Assistant { content, .. } => assistant_content_to_text(content),
        Message::ToolResult {
            tool_name, content, ..
        } => {
            let body = std::iter::once(tool_name.clone())
                .chain(content.iter().map(|b| match b {
                    ContentBlock::Text { text } => text.clone(),
                    _ => String::new(),
                }))
                .collect::<Vec<_>>()
                .join("\n");
            format!("toolResult:{}", body)
        }
        Message::System { content, .. } => content_to_text(content),
    }
}

pub fn serialize_context(ctx: &Context) -> String {
    let mut parts: Vec<String> = vec![];
    if let Some(sys) = &ctx.system_prompt {
        parts.push(format!("system:{}", sys));
    }
    for m in &ctx.messages {
        let role = match m {
            Message::User { .. } => "user",
            Message::Assistant { .. } => "assistant",
            Message::ToolResult { .. } => "toolResult",
            Message::System { .. } => "system",
        };
        parts.push(format!("{}:{}", role, message_to_text(m)));
    }
    if !ctx.tools.is_empty() {
        parts.push(format!(
            "tools:{}",
            serde_json::to_string(&ctx.tools).unwrap_or_default()
        ));
    }
    parts.join("\n\n")
}

fn common_prefix_len(a: &str, b: &str) -> usize {
    a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count()
}

fn with_usage_estimate(
    mut message: AssistantMessage,
    ctx: &Context,
    options: Option<&SimpleStreamOptions>,
    prompt_cache: &mut HashMap<String, String>,
) -> AssistantMessage {
    let prompt_text = serialize_context(ctx);
    let prompt_tokens = estimate_tokens(&prompt_text);
    let output_text = assistant_content_to_text(match &message {
        Message::Assistant { content, .. } => content,
        _ => &[],
    });
    let output_tokens = estimate_tokens(&output_text);

    let mut input = prompt_tokens;
    let mut cache_read = 0u64;
    let mut cache_write = 0u64;

    let session_id = options.and_then(|o| o.session_id.clone());
    let cache_retention = options.and_then(|o| o.cache_retention).unwrap_or_default();
    if let Some(ref sid) = session_id {
        if !matches!(cache_retention, CacheRetention::None) {
            if let Some(prev) = prompt_cache.get(sid) {
                let cached_chars = common_prefix_len(prev, &prompt_text);
                cache_read = estimate_tokens(&prev[..cached_chars]);
                cache_write = estimate_tokens(&prompt_text[cached_chars..]);
                input = prompt_tokens.saturating_sub(cache_read);
            } else {
                cache_write = prompt_tokens;
            }
            prompt_cache.insert(sid.clone(), prompt_text);
        }
    }

    let total_tokens = input + output_tokens + cache_read + cache_write;
    if let Message::Assistant { usage, .. } = &mut message {
        *usage = Usage {
            input,
            output: output_tokens,
            cache_read,
            cache_write,
            total_tokens,
            cost: UsageCost::default(),
        };
    }
    message
}

fn split_by_token_size(text: &str, min: usize, max: usize) -> Vec<String> {
    let mut out = Vec::new();
    if text.is_empty() {
        out.push(String::new());
        return out;
    }
    let mut idx = 0usize;
    while idx < text.len() {
        let tok: usize = if min == max {
            min
        } else {
            rand::thread_rng().gen_range(min..=max)
        };
        let chars = std::cmp::max(1, tok * 4);
        let end = std::cmp::min(idx + chars, text.len());
        out.push(text[idx..end].to_string());
        idx = end;
    }
    out
}

fn clone_message(
    mut m: AssistantMessage,
    api: &str,
    provider: &str,
    model_id: &str,
) -> AssistantMessage {
    if let Message::Assistant {
        api: ref mut a,
        provider: ref mut p,
        model: ref mut mid,
        timestamp: ref mut ts,
        usage: ref mut u,
        ..
    } = &mut m
    {
        *a = api.to_string();
        *p = provider.to_string();
        *mid = Some(model_id.to_string());
        if *ts == 0 {
            *ts = chrono::Utc::now().timestamp_millis();
        }
        if u.total_tokens == 0 && u.input == 0 && u.output == 0 {
            // leave usage alone for explicit ones
        }
    }
    m
}

fn create_error_message(
    err: String,
    api: &str,
    provider: &str,
    model_id: &str,
) -> AssistantMessage {
    Message::Assistant {
        content: vec![],
        api: api.to_string(),
        provider: provider.to_string(),
        model: Some(model_id.to_string()),
        usage: default_usage(),
        stop_reason: Some("error".into()),
        error_message: Some(err),
        response_id: None,
        timestamp: chrono::Utc::now().timestamp_millis(),
    }
}

fn create_aborted_message(mut partial: AssistantMessage) -> AssistantMessage {
    if let Message::Assistant {
        stop_reason: sr,
        error_message: em,
        ..
    } = &mut partial
    {
        *sr = Some("aborted".into());
        *em = Some("Request was aborted".into());
    }
    partial
}

fn error_str(msg: &Message) -> String {
    match msg {
        Message::Assistant {
            error_message: Some(e),
            ..
        } => e.clone(),
        Message::Assistant {
            stop_reason: Some(r),
            ..
        } => r.clone(),
        _ => "unknown error".to_string(),
    }
}

async fn schedule_chunk(chars: &str, tokens_per_second: Option<u64>) {
    let Some(tps) = tokens_per_second else {
        // microtask-ish: yield once
        tokio::task::yield_now().await;
        return;
    };
    if tps == 0 {
        tokio::task::yield_now().await;
        return;
    }
    let tokens = estimate_tokens(chars).max(1);
    let delay_ms = ((tokens as f64) / (tps as f64)) * 1000.0;
    if delay_ms > 0.0 {
        let dur = std::time::Duration::from_millis(delay_ms.ceil() as u64);
        tokio::time::sleep(dur).await;
    } else {
        tokio::task::yield_now().await;
    }
}

fn terminal_stop_reason(m: &AssistantMessage) -> Option<&str> {
    if let Message::Assistant {
        stop_reason: Some(sr),
        ..
    } = m
    {
        Some(sr.as_str())
    } else {
        None
    }
}

async fn stream_with_deltas(
    stream: &AssistantMessageEventStream,
    message: AssistantMessage,
    min_token_size: usize,
    max_token_size: usize,
    tokens_per_second: Option<u64>,
    signal: Option<&CancellationToken>,
) {
    // Build a partial message by cloning then resetting state to "pending".
    let mut partial = message.clone();
    if let Message::Assistant {
        ref mut content,
        ref mut stop_reason,
        ..
    } = &mut partial
    {
        content.clear();
        *stop_reason = Some("pending".into());
    }

    if signal.map(|s| s.is_cancelled()).unwrap_or(false) {
        let aborted = create_aborted_message(partial);
        stream.push(AssistantMessageEvent::Error {
            reason: "aborted".into(),
            error: error_str(&aborted),
        });
        stream.end(Some(aborted));
        return;
    }

    stream.push(AssistantMessageEvent::Start {
        partial: partial.clone(),
    });

    let blocks: Vec<ContentBlock> = match &message {
        Message::Assistant { content, .. } => content.clone(),
        _ => vec![],
    };

    let mut aborted_here = false;
    let abort_emit_and_return =
        |stream: &AssistantMessageEventStream, partial: &AssistantMessage| -> bool {
            let a = create_aborted_message(partial.clone());
            stream.push(AssistantMessageEvent::Error {
                reason: "aborted".into(),
                error: error_str(&a),
            });
            stream.end(Some(a));
            true
        };

    for (index, block) in blocks.iter().enumerate() {
        if signal.map(|s| s.is_cancelled()).unwrap_or(false) {
            aborted_here = abort_emit_and_return(stream, &partial);
            break;
        }
        match block {
            ContentBlock::Thinking {
                thinking,
                signature,
            } => {
                if let Message::Assistant {
                    ref mut content, ..
                } = &mut partial
                {
                    content.push(ContentBlock::Thinking {
                        thinking: String::new(),
                        signature: signature.clone(),
                    });
                }
                stream.push(AssistantMessageEvent::ThinkingStart);
                for chunk in split_by_token_size(thinking, min_token_size, max_token_size) {
                    schedule_chunk(&chunk, tokens_per_second).await;
                    if signal.map(|s| s.is_cancelled()).unwrap_or(false) {
                        aborted_here = abort_emit_and_return(stream, &partial);
                        break;
                    }
                    if let Message::Assistant {
                        ref mut content, ..
                    } = &mut partial
                    {
                        if let Some(ContentBlock::Thinking {
                            ref mut thinking, ..
                        }) = content.get_mut(index)
                        {
                            thinking.push_str(&chunk);
                        }
                    }
                    stream.push(AssistantMessageEvent::ThinkingDelta { delta: chunk });
                }
                if aborted_here {
                    break;
                }
                stream.push(AssistantMessageEvent::ThinkingEnd {
                    signature: signature.clone(),
                });
            }
            ContentBlock::Text { text } => {
                if let Message::Assistant {
                    ref mut content, ..
                } = &mut partial
                {
                    content.push(ContentBlock::Text {
                        text: String::new(),
                    });
                }
                stream.push(AssistantMessageEvent::TextStart);
                for chunk in split_by_token_size(text, min_token_size, max_token_size) {
                    schedule_chunk(&chunk, tokens_per_second).await;
                    if signal.map(|s| s.is_cancelled()).unwrap_or(false) {
                        aborted_here = abort_emit_and_return(stream, &partial);
                        break;
                    }
                    if let Message::Assistant {
                        ref mut content, ..
                    } = &mut partial
                    {
                        if let Some(ContentBlock::Text { ref mut text }) = content.get_mut(index) {
                            text.push_str(&chunk);
                        }
                    }
                    stream.push(AssistantMessageEvent::TextDelta { delta: chunk });
                }
                if aborted_here {
                    break;
                }
                stream.push(AssistantMessageEvent::TextEnd);
            }
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => {
                if let Message::Assistant {
                    ref mut content, ..
                } = &mut partial
                {
                    content.push(ContentBlock::ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: serde_json::Value::Object(serde_json::Map::new()),
                    });
                }
                stream.push(AssistantMessageEvent::ToolcallStart {
                    content_index: index,
                    id: Some(id.clone()),
                    name: Some(name.clone()),
                });
                let args_str = arguments.to_string();
                for chunk in split_by_token_size(&args_str, min_token_size, max_token_size) {
                    schedule_chunk(&chunk, tokens_per_second).await;
                    if signal.map(|s| s.is_cancelled()).unwrap_or(false) {
                        aborted_here = abort_emit_and_return(stream, &partial);
                        break;
                    }
                    stream.push(AssistantMessageEvent::ToolcallDelta {
                        content_index: index,
                        delta: chunk,
                    });
                }
                if aborted_here {
                    break;
                }
                if let Message::Assistant {
                    ref mut content, ..
                } = &mut partial
                {
                    if let Some(ContentBlock::ToolCall {
                        ref mut arguments, ..
                    }) = content.get_mut(index)
                    {
                        *arguments = arguments.clone();
                    }
                }
                stream.push(AssistantMessageEvent::ToolcallEnd {
                    content_index: index,
                    tool_call: block.clone(),
                });
            }
            _ => { /* ignore Image for streaming deltas */ }
        }
    }
    if aborted_here {
        return;
    }

    match terminal_stop_reason(&message) {
        Some("pending") | None => {
            let err = create_error_message(
                "Faux response ended without a stop reason".into(),
                match &message {
                    Message::Assistant { api, .. } => api,
                    _ => "",
                },
                match &message {
                    Message::Assistant { provider, .. } => provider,
                    _ => "",
                },
                match &message {
                    Message::Assistant { model: Some(m), .. } => m.as_str(),
                    _ => "",
                },
            );
            stream.push(AssistantMessageEvent::Error {
                reason: "error".into(),
                error: error_str(&err),
            });
            stream.end(Some(err));
        }
        Some("error") | Some("aborted") => {
            let reason = terminal_stop_reason(&message).unwrap().to_string();
            stream.push(AssistantMessageEvent::Error {
                reason: reason.clone(),
                error: error_str(&message),
            });
            stream.end(Some(message));
        }
        Some(other) => {
            stream.push(AssistantMessageEvent::Usage {
                usage: match &message {
                    Message::Assistant { usage, .. } => usage.clone(),
                    _ => Usage::default(),
                },
            });
            stream.push(AssistantMessageEvent::Done {
                reason: other.to_string(),
                message,
            });
            stream.end(None);
        }
    }
}

// ---------- Faux core / handle structure ----------
struct CoreInner {
    pending_responses: std::collections::VecDeque<FauxResponseStep>,
    prompt_cache: HashMap<String, String>,
    state: FauxProviderState,
}

pub struct FauxCore {
    pub api: String,
    pub provider_id: String,
    pub models: Vec<Model>,
    pub min_token_size: usize,
    pub max_token_size: usize,
    pub tokens_per_second: Option<u64>,
    inner: Mutex<CoreInner>,
}

impl FauxCore {
    pub fn new(options: RegisterFauxProviderOptions) -> Arc<Self> {
        let min_candidate = options
            .token_size
            .map(|(mn, mx)| mn.min(mx))
            .unwrap_or(DEFAULT_MIN_TOKEN_SIZE)
            .max(1);
        let max_candidate = options
            .token_size
            .map(|(_mn, mx)| mx)
            .unwrap_or(DEFAULT_MAX_TOKEN_SIZE)
            .max(min_candidate);

        let api = options.api.unwrap_or_else(|| DEFAULT_API.into());
        let provider_id = options.provider.unwrap_or_else(|| DEFAULT_PROVIDER.into());
        let defs: Vec<FauxModelDefinition> = if options.models.is_empty() {
            vec![FauxModelDefinition::default()]
        } else {
            options.models
        };
        let models: Vec<Model> = defs
            .into_iter()
            .map(|def| Model {
                id: def.id.clone(),
                provider_id: provider_id.clone(),
                name: def.name.unwrap_or(def.id),
                api: api.clone(),
                max_input_tokens: None,
                max_output_tokens: def.max_tokens,
                supports_images: def.supports_images,
                supports_audio: false,
                supports_video: false,
                supports_pdf: false,
                supports_tool_calling: true,
                supports_structured_output: false,
                supports_system_prompt: true,
                thinking: if def.reasoning {
                    ModelThinkingLevel::Medium
                } else {
                    ModelThinkingLevel::None
                },
                reasoning: def.reasoning,
                cost_rates: def.cost_rates,
                context_window: def.context_window,
                default_temperature: None,
            })
            .collect();

        Arc::new(Self {
            api,
            provider_id,
            models,
            min_token_size: min_candidate,
            max_token_size: max_candidate,
            tokens_per_second: options.tokens_per_second,
            inner: Mutex::new(CoreInner {
                pending_responses: std::collections::VecDeque::new(),
                prompt_cache: HashMap::new(),
                state: FauxProviderState::default(),
            }),
        })
    }

    pub fn set_responses(&self, responses: Vec<FauxResponseStep>) {
        let mut inner = self.inner.lock().unwrap();
        inner.pending_responses = responses.into();
    }

    pub fn append_responses(&self, responses: Vec<FauxResponseStep>) {
        let mut inner = self.inner.lock().unwrap();
        inner.pending_responses.extend(responses);
    }

    pub fn pending_response_count(&self) -> usize {
        self.inner.lock().unwrap().pending_responses.len()
    }

    pub fn state(&self) -> FauxProviderState {
        self.inner.lock().unwrap().state.clone()
    }

    pub fn get_model(&self, requested: Option<&str>) -> Option<Model> {
        match requested {
            None => self.models.first().cloned(),
            Some(id) => self.models.iter().find(|m| m.id == id).cloned(),
        }
    }

    async fn resolve_response(
        self: &Arc<Self>,
        step: FauxResponseStep,
        ctx: Context,
        stream_opts: Option<SimpleStreamOptions>,
        request_model: Model,
    ) -> AssistantMessage {
        let (state_snap, mut cache_snap) = {
            let inner = self.inner.lock().unwrap();
            (inner.state.clone(), inner.prompt_cache.clone())
        };
        let resolved = match step {
            FauxResponseStep::Message(m) => m,
            FauxResponseStep::Factory(factory) => {
                factory(
                    ctx.clone(),
                    stream_opts.clone(),
                    state_snap,
                    request_model.clone(),
                )
                .await
            }
        };
        let cloned = clone_message(resolved, &self.api, &self.provider_id, &request_model.id);
        let with_usage = with_usage_estimate(cloned, &ctx, stream_opts.as_ref(), &mut cache_snap);
        // commit prompt cache back
        let mut inner = self.inner.lock().unwrap();
        inner.prompt_cache = cache_snap;
        with_usage
    }

    pub fn stream(
        self: &Arc<Self>,
        model: &Model,
        context: Context,
        options: Option<SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        let stream = create_assistant_message_event_stream();
        let step = {
            let mut inner = self.inner.lock().unwrap();
            inner.state.call_count += 1;
            inner.pending_responses.pop_front()
        };
        let this = self.clone();
        let model = model.clone();
        let stream_clone = stream.clone();
        let signal = options.as_ref().and_then(|o| o.signal.clone());
        tokio::spawn(async move {
            let res: Result<(), ()> = async {
                if let Some(step) = step {
                    let resolved = this
                        .resolve_response(step, context, options, model.clone())
                        .await;
                    stream_with_deltas(
                        &stream_clone,
                        resolved,
                        this.min_token_size,
                        this.max_token_size,
                        this.tokens_per_second,
                        signal.as_ref(),
                    )
                    .await;
                } else {
                    let mut msg = create_error_message(
                        "No more faux responses queued".into(),
                        &this.api,
                        &this.provider_id,
                        &model.id,
                    );
                    let (mut cache_snap, _state) = {
                        let inner = this.inner.lock().unwrap();
                        (inner.prompt_cache.clone(), inner.state.clone())
                    };
                    msg = with_usage_estimate(msg, &context, None, &mut cache_snap);
                    let mut inner = this.inner.lock().unwrap();
                    inner.prompt_cache = cache_snap;
                    drop(inner);
                    stream_clone.push(AssistantMessageEvent::Error {
                        reason: "error".into(),
                        error: error_str(&msg),
                    });
                    stream_clone.end(Some(msg));
                }
                Ok(())
            }
            .await;
            if res.is_err() {
                // ignored — end already called inside, or caller will see Error event
            }
        });
        stream
    }
}

// ---------- public Provider handle ----------
pub struct FauxProvider {
    pub api: String,
    pub provider_id: String,
    pub core: Arc<FauxCore>,
    pub auth_obj: ProviderAuth,
}

impl FauxProvider {
    fn new_auth_obj() -> ProviderAuth {
        struct FauxApiKeyAuth;
        #[async_trait::async_trait]
        impl ApiKeyAuth for FauxApiKeyAuth {
            fn name(&self) -> &str {
                "Faux"
            }
            async fn login(
                &self,
                _: &(dyn AuthInteraction + Send + Sync),
            ) -> anyhow::Result<ApiKeyCredential> {
                Ok(ApiKeyCredential {
                    r#type: "api_key".into(),
                    key: Some("faux-api-key".into()),
                    env: None,
                })
            }
            async fn check(
                &self,
                _: &(dyn AuthContext + Send + Sync),
                _: Option<&ApiKeyCredential>,
                _: &CancellationToken,
            ) -> Option<AuthCheck> {
                None
            }
            async fn resolve(
                &self,
                _: &(dyn AuthContext + Send + Sync),
                _: Option<&ApiKeyCredential>,
                _: &CancellationToken,
            ) -> Option<AuthResult> {
                None
            }
        }
        ProviderAuth {
            api_key: Some(Box::new(FauxApiKeyAuth)),
            oauth: None,
        }
    }
}

#[async_trait::async_trait]
impl Provider for FauxProvider {
    fn id(&self) -> &str {
        &self.provider_id
    }
    fn name(&self) -> &str {
        &self.provider_id
    }
    fn auth(&self) -> &ProviderAuth {
        &self.auth_obj
    }
    fn get_models(&self) -> Vec<Model> {
        self.core.models.clone()
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
        let simple = SimpleStreamOptions {
            signal: options.signal,
            provider_extra: options.request_options.extra_body,
            ..Default::default()
        };
        self.core.stream(model, context, Some(simple))
    }
    fn stream_simple(
        &self,
        model: &Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        self.core.stream(model, context, Some(options))
    }
}

// `fauxProvider()`-style factory: returns a handle that exposes state,
// setResponses, appendResponses, pending count, and a Provider to register
// with a `Models` collection.
pub struct FauxProviderHandle {
    pub api: String,
    pub models: Vec<Model>,
    pub core: Arc<FauxCore>,
    pub provider: Arc<dyn Provider + Send + Sync>,
}

impl FauxProviderHandle {
    pub fn provider(&self) -> Arc<dyn Provider + Send + Sync> {
        self.provider.clone()
    }
    pub fn set_responses(&self, responses: Vec<FauxResponseStep>) {
        self.core.set_responses(responses);
    }
    pub fn append_responses(&self, responses: Vec<FauxResponseStep>) {
        self.core.append_responses(responses);
    }
    pub fn get_pending_response_count(&self) -> usize {
        self.core.pending_response_count()
    }
    pub fn state(&self) -> FauxProviderState {
        self.core.state()
    }
    pub fn get_model(&self, id: Option<&str>) -> Option<Model> {
        self.core.get_model(id)
    }
}

pub fn faux_provider(options: Option<RegisterFauxProviderOptions>) -> FauxProviderHandle {
    let opts = options.unwrap_or_default();
    let core = FauxCore::new(opts);
    let provider_impl = FauxProvider {
        api: core.api.clone(),
        provider_id: core.provider_id.clone(),
        core: core.clone(),
        auth_obj: FauxProvider::new_auth_obj(),
    };
    FauxProviderHandle {
        api: core.api.clone(),
        models: core.models.clone(),
        core,
        provider: Arc::new(provider_impl),
    }
}

// Standalone `register_faux_provider()`: produces a registration handle that
// owns its provider and can be passed directly to `complete`/`stream` helper
// functions. Mirrors the TypeScript `registerFauxProvider()` compat API.
pub struct FauxRegistration {
    pub api: String,
    pub provider_id: String,
    pub models: Vec<Model>,
    pub core: Arc<FauxCore>,
    pub provider: Arc<dyn Provider + Send + Sync>,
}

impl FauxRegistration {
    pub fn set_responses(&self, responses: Vec<FauxResponseStep>) {
        self.core.set_responses(responses);
    }
    pub fn append_responses(&self, responses: Vec<FauxResponseStep>) {
        self.core.append_responses(responses);
    }
    pub fn get_pending_response_count(&self) -> usize {
        self.core.pending_response_count()
    }
    pub fn state(&self) -> FauxProviderState {
        self.core.state()
    }
    pub fn get_model(&self, id: Option<&str>) -> Option<Model> {
        self.core.get_model(id)
    }
    pub fn unregister(&self) {
        // In standalone mode we don't push into a global registry, but we
        // clear pending responses so subsequent calls see "exhausted".
        self.core.set_responses(vec![]);
    }
}

pub fn register_faux_provider(options: Option<RegisterFauxProviderOptions>) -> FauxRegistration {
    let opts = options.unwrap_or_default();
    let core = FauxCore::new(opts);
    let api = core.api.clone();
    let provider_id = core.provider_id.clone();
    let models = core.models.clone();
    let provider_impl = FauxProvider {
        api: api.clone(),
        provider_id: provider_id.clone(),
        core: core.clone(),
        auth_obj: FauxProvider::new_auth_obj(),
    };
    FauxRegistration {
        api,
        provider_id,
        models,
        core,
        provider: Arc::new(provider_impl),
    }
}

// Helper for converting a simple `AssistantMessage` into a queue step.
pub fn step_from_message(m: AssistantMessage) -> FauxResponseStep {
    FauxResponseStep::Message(m)
}

// Convenience: build a FauxResponseStep that simply returns a static string.
pub fn step_from_string<S: Into<String>>(s: S) -> FauxResponseStep {
    FauxResponseStep::Message(faux_assistant_message(
        s.into(),
        FauxAssistantMessageOptions::default(),
    ))
}

// ------ Standalone `complete()`/`stream()` helpers for registerFauxProvider tests ------
/// Use a faux registration *directly* (without `Models`) to obtain a single
/// assembled assistant message, identical to the TS compat `complete()`.
pub async fn complete(
    registration: &FauxRegistration,
    context: Context,
    options: Option<SimpleStreamOptions>,
) -> Result<AssistantMessage, String> {
    let model = registration
        .get_model(None)
        .ok_or_else(|| "faux registration has no default model".to_string())?;
    let stream = registration.core.stream(&model, context, options);
    let result_fut = stream.result_future();
    let mut st = stream;
    let mut last_error: Option<String> = None;
    while let Some(evt) = st.next().await {
        if let AssistantMessageEvent::Error { error, .. } = evt {
            last_error = Some(error);
        }
    }
    let msg = result_fut.await;
    if msg.stop_reason().map(|s| s == "error").unwrap_or(false) {
        // pass through as Ok(assistant error message) per TS semantics
        Ok(msg)
    } else if let Some(e) = last_error {
        Err(e)
    } else {
        Ok(msg)
    }
}

/// Companion to `complete()`: returns the raw event stream, identical to the
/// TS compat `stream()` used by faux-provider tests.
pub fn stream(
    registration: &FauxRegistration,
    context: Context,
    options: Option<SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    let model = registration.get_model(None).expect("default model");
    registration.core.stream(&model, context, options)
}
