//! OpenAI Codex **Responses** wire adapter (`chatgpt.com/backend-api/codex/responses`).
//!
//! Rust port of `@earendil-works/pi-ai`'s `api/openai-codex-responses.ts` (and
//! the shared `api/openai-responses-shared.ts` projections):
//!
//! * Request building (`store: false`, `instructions`, Responses `input`
//!   items, `include: ["reasoning.encrypted_content"]`, …).
//! * SSE parsing + retry (429/5xx, `retry-after`/`retry-after-ms`, terminal
//!   usage-limit detection, exponential backoff, header timeouts).
//! * Event projection into [`crate::types::AssistantMessageEvent`] with
//!   thinking / text / tool-call slots keyed by `output_index`.
//! * Codex terminal-event normalisation (`response.done` /
//!   `response.completed` / `response.incomplete`, `end_turn` capture).
//! * JWT `chatgpt-account-id` extraction, `session-id` /
//!   `x-client-request-id` / `prompt_cache_key` cache affinity (clamped to 64
//!   chars), service-tier cost multipliers.
//!
//! `max_output_tokens` is deliberately never sent: the Codex backend rejects
//! it with 400 "Unsupported parameter".

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use serde_json::{json, Value};

use crate::types::{
    AssistantMessage, AssistantMessageEvent, CacheRetention, CancellationToken, ContentBlock,
    Context, Message, Model, ModelThinkingLevel, ProviderHeaders, ToolChoice, Usage,
};
use crate::utils::event_stream::{
    create_assistant_message_event_stream, AssistantMessageEventStream,
};

pub const DEFAULT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api";
const JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";
const DEFAULT_MAX_RETRIES: u32 = 0;
const BASE_DELAY_MS: u64 = 1000;
const DEFAULT_MAX_RETRY_DELAY_MS: u64 = 60_000;
const OPENAI_BETA_RESPONSES_EXPERIMENTAL: &str = "responses=experimental";
pub const OPENAI_BETA_RESPONSES_WEBSOCKETS: &str = "responses_websockets=2026-02-06";
const PROMPT_CACHE_KEY_MAX_CHARS: usize = 64;

/// Providers whose `toolCall` ids already use the `call_id|item_id` shape the
/// Responses API understands (TS: `CODEX_TOOL_CALL_PROVIDERS`). `chatgpt-plus`
/// is otter-ai's subscription provider id for the same wire protocol.
const CODEX_TOOL_CALL_PROVIDERS: &[&str] = &["openai", "openai-codex", "opencode", "chatgpt-plus"];

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum CodexTransport {
    /// WebSocket first, SSE fallback (pi-ai default).
    #[default]
    Auto,
    Sse,
    Websocket,
    WebsocketCached,
}

impl CodexTransport {
    pub fn parse(s: &str) -> Self {
        match s {
            "sse" => CodexTransport::Sse,
            "websocket" => CodexTransport::Websocket,
            "websocket-cached" => CodexTransport::WebsocketCached,
            _ => CodexTransport::Auto,
        }
    }
}

/// Request-body inspection hook (TS: `options.onPayload`).
pub type OnPayloadHook = Arc<dyn Fn(&Value) + Send + Sync>;

/// Full Codex stream options (TS: `OpenAICodexResponsesOptions`).
#[derive(Clone)]
pub struct CodexStreamOptions {
    pub api_key: String,
    pub base_url: Option<String>,
    pub extra_headers: ProviderHeaders,
    pub session_id: Option<String>,
    pub cache_retention: CacheRetention,
    pub signal: Option<CancellationToken>,
    pub reasoning_effort: Option<String>,
    pub reasoning_summary: Option<String>,
    pub text_verbosity: Option<String>,
    pub tool_choice: Option<ToolChoice>,
    pub temperature: Option<f64>,
    pub service_tier: Option<String>,
    pub max_retries: Option<u32>,
    pub max_retry_delay_ms: Option<u64>,
    pub timeout_ms: Option<u64>,
    /// WebSocket connect timeout (TS: `websocketConnectTimeoutMs`, default 15 s).
    #[cfg(feature = "codex-websocket")]
    pub websocket_connect_timeout_ms: Option<u64>,
    /// Idle lifetime of a cached websocket connection (TS: 5 min).
    #[cfg(feature = "codex-websocket")]
    pub websocket_cache_ttl_ms: Option<u64>,
    /// Age limit of a cached websocket connection (TS: 55 min).
    #[cfg(feature = "codex-websocket")]
    pub websocket_max_age_ms: Option<u64>,
    pub transport: CodexTransport,
    /// Request-body inspection hook (TS: `options.onPayload`).
    pub on_payload: Option<OnPayloadHook>,
}

impl Default for CodexStreamOptions {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: None,
            extra_headers: ProviderHeaders::new(),
            session_id: None,
            cache_retention: CacheRetention::None,
            signal: None,
            reasoning_effort: None,
            reasoning_summary: None,
            text_verbosity: None,
            tool_choice: None,
            temperature: None,
            service_tier: None,
            max_retries: None,
            max_retry_delay_ms: None,
            timeout_ms: None,
            #[cfg(feature = "codex-websocket")]
            websocket_connect_timeout_ms: None,
            #[cfg(feature = "codex-websocket")]
            websocket_cache_ttl_ms: None,
            #[cfg(feature = "codex-websocket")]
            websocket_max_age_ms: None,
            transport: CodexTransport::Auto,
            on_payload: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Text signature (TS: encodeTextSignatureV1 / parseTextSignature)
// ---------------------------------------------------------------------------

fn encode_text_signature_v1(id: &str, phase: Option<&str>) -> String {
    match phase {
        Some(p) => format!("{{\"v\":1,\"id\":\"{}\",\"phase\":\"{}\"}}", id, p),
        None => format!("{{\"v\":1,\"id\":\"{}\"}}", id),
    }
}

fn parse_text_signature(signature: Option<&str>) -> Option<(String, Option<String>)> {
    let sig = signature?;
    if sig.starts_with('{') {
        if let Ok(v) = serde_json::from_str::<Value>(sig) {
            if v.get("v").and_then(|x| x.as_i64()) == Some(1) {
                if let Some(id) = v.get("id").and_then(|x| x.as_str()) {
                    let phase = v.get("phase").and_then(|x| x.as_str());
                    if phase.is_none_or(|p| p == "commentary" || p == "final_answer") {
                        return Some((id.to_string(), phase.map(|p| p.to_string())));
                    }
                    return Some((id.to_string(), None));
                }
            }
        }
    }
    // Legacy plain-string signature.
    Some((sig.to_string(), None))
}

// ---------------------------------------------------------------------------
// Structured stream errors (TS: CodexApiError / CodexProtocolError)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CodexErrorKind {
    /// `error` / `response.failed` frames — never fall back to SSE for these
    /// (except the two retryable codes handled by the websocket loop).
    Api,
    /// Wire-level JSON violations.
    Protocol,
    /// Everything else (connect/close/idle timeouts/abort) — SSE-fallback
    /// eligible when nothing was streamed yet.
    Other,
}

#[derive(Debug, Clone)]
pub(crate) struct CodexStreamError {
    pub message: String,
    pub code: Option<String>,
    pub kind: CodexErrorKind,
}

impl CodexStreamError {
    fn other(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: None,
            kind: CodexErrorKind::Other,
        }
    }

    fn api(message: impl Into<String>, code: Option<String>) -> Self {
        Self {
            message: message.into(),
            code,
            kind: CodexErrorKind::Api,
        }
    }

    fn protocol(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: None,
            kind: CodexErrorKind::Protocol,
        }
    }
}

impl std::fmt::Display for CodexStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn stream_codex(
    model: &Model,
    context: Context,
    opts: CodexStreamOptions,
) -> AssistantMessageEventStream {
    let es = create_assistant_message_event_stream();
    let es_handle = es.clone();
    let model = model.clone();
    tokio::spawn(async move {
        let mut output = Message::assistant_default(
            "openai-codex-responses".to_string(),
            model.provider_id.clone(),
        )
        .with_model(Some(model.id.clone()))
        .with_stop_reason(Some("pending".to_string()));

        let result = run_codex(&model, context, &opts, &mut output, &es_handle).await;

        match result {
            Ok(()) => {
                let reason = match &output {
                    Message::Assistant {
                        stop_reason: Some(r), ..
                    } => r.clone(),
                    _ => "stop".to_string(),
                };
                es_handle.push(AssistantMessageEvent::Done {
                    reason,
                    message: output,
                });
                es_handle.end(None);
            }
            Err(run_error) => {
                let aborted = opts
                    .signal
                    .as_ref()
                    .map(|s| s.is_cancelled())
                    .unwrap_or(false);
                let (reason, error) = if aborted {
                    ("aborted".to_string(), "Request was aborted".to_string())
                } else {
                    ("error".to_string(), run_error)
                };
                if let Message::Assistant {
                    stop_reason,
                    error_message,
                    ..
                } = &mut output
                {
                    *stop_reason = Some(reason.clone());
                    *error_message = Some(error.clone());
                }
                es_handle.push(AssistantMessageEvent::Error {
                    reason,
                    error: error.clone(),
                });
                es_handle.end(Some(output));
            }
        }
    });
    es
}

async fn run_codex(
    model: &Model,
    context: Context,
    opts: &CodexStreamOptions,
    output: &mut AssistantMessage,
    es: &AssistantMessageEventStream,
) -> Result<(), String> {
    if opts.api_key.is_empty() {
        return Err(format!("No API key for provider: {}", model.provider_id));
    }

    let account_id = extract_account_id(&opts.api_key)?;

    // Cache affinity: `cacheRetention: "none"` disables the session key.
    let cache_session_id: Option<String> = match opts.cache_retention {
        CacheRetention::None => None,
        _ => opts.session_id.clone(),
    };
    let codex_session_id: Option<String> =
        cache_session_id.as_deref().map(clamp_prompt_cache_key);

    let grammar_tool_input_properties =
        create_grammar_tool_input_properties(&context.tools).unwrap_or_default();
    let mut body = build_request_body(model, &context, opts, codex_session_id.as_deref())?;
    if let Some(hook) = &opts.on_payload {
        hook(&body);
    }
    let _ = &mut body;

    // WebSocket transport first (TS: `transport !== "sse"`); SSE is both the
    // explicit `sse` mode and the fallback when a websocket fails before any
    // output was produced. Matches pi-ai's retry ladder:
    //   1. `previous_response_not_found`  → retry once via websocket (full context)
    //   2. connection-limit before start  → retry once on a fresh connection
    //   3. API/protocol errors            → hard failure (no SSE fallback)
    //   4. transport errors before start  → SSE fallback + session marking
    #[cfg(feature = "codex-websocket")]
    if opts.transport != CodexTransport::Sse
        && !websocket::is_sse_fallback_active(codex_session_id.as_deref())
    {
        let mut retried_connection_limit = false;
        let mut retried_missing_continuation = false;
        loop {
            match websocket::run_websocket(
                model,
                opts,
                &body,
                &account_id,
                codex_session_id.as_deref(),
                es,
                output,
                &grammar_tool_input_properties,
            )
            .await
            {
                Ok(()) => {
                    return assert_successful_output(output).map_err(|e| e.message);
                }
                Err(err) => {
                    let aborted = opts
                        .signal
                        .as_ref()
                        .map(|s| s.is_cancelled())
                        .unwrap_or(false);
                    let connection_limit_before_start = !err.started
                        && err.code.as_deref()
                            == Some(websocket::WEBSOCKET_CONNECTION_LIMIT_REACHED);
                    let previous_response_not_found =
                        err.code.as_deref() == Some(websocket::PREVIOUS_RESPONSE_NOT_FOUND);

                    if !aborted && previous_response_not_found && !retried_missing_continuation {
                        retried_missing_continuation = true;
                        continue;
                    }
                    if !aborted && connection_limit_before_start && !retried_connection_limit {
                        retried_connection_limit = true;
                        continue;
                    }
                    if aborted || (err.kind == CodexErrorKind::Api && !connection_limit_before_start)
                    {
                        return Err(err.message);
                    }
                    websocket::record_websocket_failure(
                        codex_session_id.as_deref(),
                        &err.message,
                    );
                    if err.started {
                        return Err(err.message);
                    }
                    websocket::record_sse_fallback(codex_session_id.as_deref());
                    break; // → SSE fallback
                }
            }
        }
    } else if opts.transport != CodexTransport::Sse {
        // WebSocket disabled for this session after an earlier failure.
        #[cfg(feature = "codex-websocket")]
        websocket::record_sse_fallback(codex_session_id.as_deref());
    }

    let headers = build_sse_headers(
        &opts.extra_headers,
        &account_id,
        &opts.api_key,
        codex_session_id.as_deref(),
    );

    process_sse_with_retry(model, opts, &headers, &body, output, es, &grammar_tool_input_properties).await
}

/// Terminal checks (TS: `assertSuccessfulOutput`).
fn assert_successful_output(output: &AssistantMessage) -> Result<(), CodexStreamError> {
    match output {
        Message::Assistant {
            stop_reason: Some(r),
            error_message,
            ..
        } => match r.as_str() {
            "pending" => Err(CodexStreamError::other(
                "Codex stream ended without a stop reason",
            )),
            "error" | "aborted" => Err(CodexStreamError::other(
                error_message
                    .clone()
                    .unwrap_or_else(|| "An unknown error occurred".to_string()),
            )),
            _ => Ok(()),
        },
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Request building (TS: buildRequestBody)
// ---------------------------------------------------------------------------

pub fn build_request_body(
    model: &Model,
    context: &Context,
    opts: &CodexStreamOptions,
    cache_session_id: Option<&str>,
) -> Result<Value, String> {
    let grammar_tool_input_properties = create_grammar_tool_input_properties(&context.tools)?;
    let input = convert_responses_messages(
        model,
        context,
        CODEX_TOOL_CALL_PROVIDERS,
        false,
        &grammar_tool_input_properties,
    );

    let mut body = json!({
        "model": model.id,
        "store": false,
        "stream": true,
        "instructions": context
            .system_prompt
            .clone()
            .unwrap_or_else(|| "You are a helpful assistant.".to_string()),
        "input": input,
        "text": { "verbosity": opts.text_verbosity.clone().unwrap_or_else(|| "low".to_string()) },
        "include": ["reasoning.encrypted_content"],
        "tool_choice": serialize_tool_choice(opts.tool_choice.as_ref(), context.tool_choice.as_ref()),
        "parallel_tool_calls": true,
    });

    if let Some(sid) = cache_session_id {
        body["prompt_cache_key"] = json!(sid);
    }

    if let Some(temp) = opts.temperature {
        body["temperature"] = json!(temp);
    }

    if let Some(tier) = &opts.service_tier {
        body["service_tier"] = json!(tier);
    }

    if !context.tools.is_empty() {
        body["tools"] = Value::Array(
            context
                .tools
                .iter()
                .map(|t| {
                    if let Some(grammar) = resolve_grammar_sampling(t) {
                        json!({
                            "type": "custom",
                            "name": t.name,
                            "description": t.description.clone().unwrap_or_default(),
                            "format": {
                                "type": "grammar",
                                "syntax": grammar.0,
                                "definition": grammar.1,
                            },
                        })
                    } else {
                        json!({
                            "type": "function",
                            "name": t.name,
                            "description": t.description.clone().unwrap_or_default(),
                            "parameters": t.parameters,
                            "strict": null,
                        })
                    }
                })
                .collect(),
        );
    }

    if let Some(effort_request) = opts.reasoning_effort.as_deref() {
        let effort = if effort_request == "off" {
            model
                .thinking_level_map
                .as_ref()
                .and_then(|m| m.get("off").cloned())
                .unwrap_or_else(|| "none".to_string())
        } else {
            model
                .thinking_level_map
                .as_ref()
                .and_then(|m| m.get(effort_request).cloned())
                .unwrap_or_else(|| effort_request.to_string())
        };
        body["reasoning"] = json!({
            "effort": effort,
            "summary": opts
                .reasoning_summary
                .clone()
                .unwrap_or_else(|| "auto".to_string()),
        });
    }

    Ok(body)
}

fn serialize_tool_choice(
    stream_choice: Option<&ToolChoice>,
    ctx_choice: Option<&ToolChoice>,
) -> Value {
    let choice = stream_choice.or(ctx_choice);
    match choice {
        None | Some(ToolChoice::Auto) => json!("auto"),
        Some(ToolChoice::None) => json!("none"),
        Some(ToolChoice::Required) => json!("required"),
        Some(ToolChoice::Tool { name }) => json!({ "type": "function", "name": name }),
    }
}

// ---------------------------------------------------------------------------
// Message conversion (TS: convertResponsesMessages, codex flavour)
// ---------------------------------------------------------------------------

/// TS: `resolveGrammarConstrainedSampling` — returns (syntax, definition)
/// for grammar-constrained tools, or None for plain function tools.
fn resolve_grammar_sampling(tool: &crate::types::Tool) -> Option<(String, String)> {
    let config = tool.constrained_sampling.as_ref()?;
    if config.sampling_type != "grammar" {
        return None;
    }
    let variants = config.variants.as_ref()?;
    let lark = variants
        .openai_lark
        .as_deref()
        .filter(|v| !v.trim().is_empty());
    let regex = variants
        .openai_regex
        .as_deref()
        .filter(|v| !v.trim().is_empty());
    match (lark, regex) {
        (Some(l), _) => Some(("lark".to_string(), l.to_string())),
        (None, Some(r)) => Some(("regex".to_string(), r.to_string())),
        (None, None) => None,
    }
}

/// TS: `inferGrammarInputProperty` — the single required string property of
/// the tool's object schema becomes the custom-tool input property.
fn infer_grammar_input_property(tool: &crate::types::Tool) -> Option<String> {
    let schema = tool.parameters.as_object()?;
    if schema.get("type").and_then(|v| v.as_str()) != Some("object") {
        return None;
    }
    let required = schema.get("required")?.as_array()?;
    if required.len() != 1 {
        return None;
    }
    let property = required[0].as_str()?;
    let props = schema.get("properties")?.as_object()?;
    let prop_schema = props.get(property)?.as_object()?;
    if prop_schema.get("type").and_then(|v| v.as_str()) != Some("string") {
        return None;
    }
    Some(property.to_string())
}

/// TS: `createGrammarToolInputProperties` — maps tool name → input property
/// for grammar-constrained tools.
fn create_grammar_tool_input_properties(
    tools: &[crate::types::Tool],
) -> Result<std::collections::HashMap<String, String>, String> {
    let mut properties = std::collections::HashMap::new();
    for tool in tools {
        if tool.constrained_sampling.is_some() {
            if resolve_grammar_sampling(tool).is_none() {
                return Err(format!(
                    "Tool \"{}\" cannot use grammar constrained sampling: no supported grammar variant was provided.",
                    tool.name
                ));
            }
            let property = infer_grammar_input_property(tool).ok_or_else(|| {
                format!(
                    "Tool \"{}\" cannot use grammar constrained sampling: requires an object schema with exactly one required string property.",
                    tool.name
                )
            })?;
            properties.insert(tool.name.clone(), property);
        }
    }
    Ok(properties)
}

pub fn convert_responses_messages(
    model: &Model,
    context: &Context,
    allowed_tool_call_providers: &[&str],
    include_system_prompt: bool,
    grammar_tool_input_properties: &std::collections::HashMap<String, String>,
) -> Vec<Value> {
    let mut messages: Vec<Value> = Vec::new();

    if include_system_prompt {
        if let Some(sys) = &context.system_prompt {
            messages.push(json!({ "role": "system", "content": sys }));
        }
    }

    let allowed = allowed_tool_call_providers.contains(&model.provider_id.as_str());
    let mut msg_index = 0usize;

    for msg in &context.messages {
        match msg {
            Message::User { content, .. } => {
                let parts: Vec<Value> = content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text, .. } => {
                            Some(json!({ "type": "input_text", "text": text }))
                        }
                        ContentBlock::Image(img) => Some(json!({
                            "type": "input_image",
                            "detail": "auto",
                            "image_url": format!(
                                "data:{};base64,{}",
                                img.mime_type.as_deref().unwrap_or("application/octet-stream"),
                                img.data
                            ),
                        })),
                        _ => None,
                    })
                    .collect();
                if parts.is_empty() {
                    msg_index += 1;
                    continue;
                }
                messages.push(json!({ "role": "user", "content": parts }));
            }
            Message::Assistant { content, provider, .. } => {
                let same_provider = provider == &model.provider_id;
                let mut out: Vec<Value> = Vec::new();
                let mut text_block_index = 0usize;
                for block in content {
                    match block {
                        ContentBlock::Thinking { signature, .. } => {
                            if let Some(item) = signature
                                .as_deref()
                                .and_then(|sig| serde_json::from_str::<Value>(sig).ok())
                            {
                                out.push(item);
                            }
                        }
                        ContentBlock::Text {
                            text,
                            text_signature,
                        } => {
                            // TS: parseTextSignature → real message id when
                            // available; otherwise positional fallback. Ids
                            // over 64 chars hash down to `msg_<shortHash>`.
                            let parsed = parse_text_signature(text_signature.as_deref());
                            let mut msg_id = parsed
                                .as_ref()
                                .map(|(id, _)| id.clone())
                                .filter(|id| !id.is_empty())
                                .unwrap_or_default();
                            if msg_id.is_empty() {
                                msg_id = if text_block_index == 0 {
                                    format!("msg_pi_{}", msg_index)
                                } else {
                                    format!("msg_pi_{}_{}", msg_index, text_block_index)
                                };
                            } else if msg_id.chars().count() > 64 {
                                msg_id = format!("msg_{}", short_hash(&msg_id));
                            }
                            text_block_index += 1;
                            out.push(json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{ "type": "output_text", "text": text, "annotations": [] }],
                                "status": "completed",
                                "id": msg_id,
                            }));
                        }
                        ContentBlock::ToolCall {
                            id, name, arguments, ..
                        } => {
                            if let Some(property) = grammar_tool_input_properties.get(name) {
                                // TS: custom tool calls replay their grammar
                                // input as a plain string; the item id is
                                // passed through unchanged (ctc_* ids are
                                // valid for custom_tool_call items).
                                let (call_id, item_id) = split_tool_call_id(id);
                                let input = arguments
                                    .get(property)
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let mut item = json!({
                                    "type": "custom_tool_call",
                                    "call_id": call_id,
                                    "name": name,
                                    "input": input,
                                });
                                if let Some(iid) = item_id {
                                    item["id"] = json!(iid);
                                }
                                out.push(item);
                                continue;
                            }
                            let (call_id, item_id) = split_normalized_tool_call_id(id, allowed, same_provider);
                            let mut item = json!({
                                "type": "function_call",
                                "call_id": call_id,
                                "name": name,
                                "arguments": serde_json::to_string(arguments)
                                    .unwrap_or_else(|_| "{}".to_string()),
                            });
                            if let Some(iid) = item_id {
                                item["id"] = json!(iid);
                            }
                            out.push(item);
                        }
                        _ => {}
                    }
                }
                if out.is_empty() {
                    msg_index += 1;
                    continue;
                }
                messages.extend(out);
            }
            Message::ToolResult {
                tool_call_id,
                tool_name,
                content,
                ..
            } => {
                let (call_id, _) =
                    split_normalized_tool_call_id(tool_call_id, allowed, true);
                let text = content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text, .. } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let output_text = if text.is_empty() {
                    "(no tool output)".to_string()
                } else {
                    text
                };
                let item_type = if grammar_tool_input_properties.contains_key(tool_name) {
                    "custom_tool_call_output"
                } else {
                    "function_call_output"
                };
                messages.push(json!({
                    "type": item_type,
                    "call_id": call_id,
                    "output": output_text,
                }));
            }
            Message::System { .. } => {
                // Codex carries the system prompt via `instructions`.
            }
        }
        msg_index += 1;
    }

    messages
}

fn split_tool_call_id(id: &str) -> (String, Option<String>) {
    match id.split_once('|') {
        Some((call, item)) => (call.to_string(), Some(item.to_string())),
        None => (id.to_string(), None),
    }
}

/// TS: `normalizeToolCallId` + the codex `call_id|item_id` split. Returns the
/// normalized `(call_id, item_id)` pair; the item id is only kept for
/// same-provider (replayable) `fc_*` ids.
fn split_normalized_tool_call_id(
    id: &str,
    allowed: bool,
    same_provider: bool,
) -> (String, Option<String>) {
    if !allowed {
        return (normalize_id_part(id), None);
    }
    let (call_id, item_id) = split_tool_call_id(id);
    let normalized_call_id = normalize_id_part(&call_id);
    let normalized_item_id = item_id
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(|iid| {
            if same_provider {
                if iid.starts_with("fc_") {
                    normalize_id_part(iid)
                } else {
                    normalize_id_part(&format!("fc_{}", iid))
                }
            } else {
                // Foreign tool call: rewrite the item id as fc_<shortHash> so
                // the backend accepts it (TS: buildForeignResponsesItemId).
                let hashed = format!("fc_{}", short_hash(iid));
                if hashed.len() > 64 {
                    hashed[..64].to_string()
                } else {
                    hashed
                }
            }
        });
    (normalized_call_id, normalized_item_id)
}

fn normalize_id_part(part: &str) -> String {
    let sanitized: String = part
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let clamped: String = sanitized.chars().take(64).collect();
    clamped.trim_end_matches('_').to_string()
}

/// Port of pi-ai `shortHash` (two-lane multiply hash, base-36).
fn short_hash(s: &str) -> String {
    fn imul(a: i32, b: i32) -> i32 {
        a.wrapping_mul(b)
    }
    let mut h1: i32 = 0xdeadbeefu32 as i32;
    let mut h2: i32 = 0x41c6ce57u32 as i32;
    for ch in s.chars() {
        let c = ch as u32 as i32;
        h1 = imul(h1 ^ c, 2654435761u32 as i32);
        h2 = imul(h2 ^ c, 1597334677u32 as i32);
    }
    h1 = imul(h1 ^ ((h1 as u32) >> 16) as i32, 2246822507u32 as i32)
        ^ imul(h2 ^ ((h2 as u32) >> 13) as i32, 3266489909u32 as i32);
    h2 = imul(h2 ^ ((h2 as u32) >> 16) as i32, 2246822507u32 as i32)
        ^ imul(h1 ^ ((h1 as u32) >> 13) as i32, 3266489909u32 as i32);
    let h2u = h2 as u32;
    let h1u = h1 as u32;
    format!("{}{}", radix36(h2u), radix36(h1u))
}

fn radix36(mut v: u32) -> String {
    if v == 0 {
        return "0".to_string();
    }
    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut buf = Vec::new();
    while v > 0 {
        buf.push(DIGITS[(v % 36) as usize]);
        v /= 36;
    }
    buf.into_iter().rev().map(|b| b as char).collect()
}

// ---------------------------------------------------------------------------
// URL / header helpers
// ---------------------------------------------------------------------------

pub fn resolve_codex_url(base_url: Option<&str>) -> String {
    let raw = base_url
        .filter(|b| !b.trim().is_empty())
        .unwrap_or(DEFAULT_CODEX_BASE_URL);
    let normalized = raw.trim_end_matches('/');
    if normalized.ends_with("/codex/responses") {
        normalized.to_string()
    } else if normalized.ends_with("/codex") {
        format!("{}/responses", normalized)
    } else {
        format!("{}/codex/responses", normalized)
    }
}

pub fn resolve_codex_websocket_url(base_url: Option<&str>) -> String {
    let url = resolve_codex_url(base_url);
    if let Some(rest) = url.strip_prefix("https://") {
        format!("wss://{}", rest)
    } else if let Some(rest) = url.strip_prefix("http://") {
        format!("ws://{}", rest)
    } else {
        url
    }
}

fn build_sse_headers(
    extra_headers: &ProviderHeaders,
    account_id: &str,
    token: &str,
    session_id: Option<&str>,
) -> reqwest::header::HeaderMap {
    let mut map = reqwest::header::HeaderMap::new();
    let insert = |map: &mut reqwest::header::HeaderMap, k: &str, v: &str| {
        if let (Ok(name), Ok(value)) = (
            reqwest::header::HeaderName::from_bytes(k.as_bytes()),
            reqwest::header::HeaderValue::from_str(v),
        ) {
            map.insert(name, value);
        }
    };
    for (k, v) in extra_headers {
        insert(&mut map, k, v);
    }
    insert(&mut map, "Authorization", &format!("Bearer {}", token));
    insert(&mut map, "chatgpt-account-id", account_id);
    insert(&mut map, "OpenAI-Beta", OPENAI_BETA_RESPONSES_EXPERIMENTAL);
    insert(&mut map, "accept", "text/event-stream");
    insert(&mut map, "content-type", "application/json");
    if let Some(sid) = session_id {
        insert(&mut map, "session-id", sid);
        insert(&mut map, "x-client-request-id", sid);
    }
    map
}

pub fn build_websocket_headers(
    extra_headers: &ProviderHeaders,
    account_id: &str,
    token: &str,
    request_id: &str,
) -> reqwest::header::HeaderMap {
    let mut map = reqwest::header::HeaderMap::new();
    let insert = |map: &mut reqwest::header::HeaderMap, k: &str, v: &str| {
        if let (Ok(name), Ok(value)) = (
            reqwest::header::HeaderName::from_bytes(k.as_bytes()),
            reqwest::header::HeaderValue::from_str(v),
        ) {
            map.insert(name, value);
        }
    };
    for (k, v) in extra_headers {
        insert(&mut map, k, v);
    }
    insert(&mut map, "Authorization", &format!("Bearer {}", token));
    insert(&mut map, "chatgpt-account-id", account_id);
    insert(&mut map, "OpenAI-Beta", OPENAI_BETA_RESPONSES_WEBSOCKETS);
    insert(&mut map, "session-id", request_id);
    insert(&mut map, "x-client-request-id", request_id);
    map
}

pub fn clamp_prompt_cache_key(key: &str) -> String {
    let chars: Vec<char> = key.chars().collect();
    if chars.len() <= PROMPT_CACHE_KEY_MAX_CHARS {
        key.to_string()
    } else {
        chars[..PROMPT_CACHE_KEY_MAX_CHARS].iter().collect()
    }
}

// ---------------------------------------------------------------------------
// JWT account-id extraction
// ---------------------------------------------------------------------------

pub fn extract_account_id(token: &str) -> Result<String, String> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return Err("invalid token".to_string());
    }
    let payload = decode_base64url(parts[1]).map_err(|e| format!("invalid base64: {e}"))?;
    let value: Value =
        serde_json::from_slice(&payload).map_err(|e| format!("invalid payload JSON: {e}"))?;
    let account_id = value
        .get(JWT_CLAIM_PATH)
        .and_then(|auth| auth.get("chatgpt_account_id"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "no account ID in token".to_string())?;
    Ok(account_id.to_string())
}

fn decode_base64url(input: &str) -> Result<Vec<u8>, String> {
    let mut buf: Vec<u8> = Vec::with_capacity(input.len() * 3 / 4);
    let mut acc: u32 = 0;
    let mut bits: u32 = 0;
    for c in input.chars() {
        let v = match c {
            'A'..='Z' => c as u32 - 'A' as u32,
            'a'..='z' => c as u32 - 'a' as u32 + 26,
            '0'..='9' => c as u32 - '0' as u32 + 52,
            '-' | '+' => 62,
            '_' | '/' => 63,
            '=' => continue,
            _ => return Err(format!("invalid base64 char: {c}")),
        };
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            buf.push(((acc >> bits) & 0xff) as u8);
        }
    }
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Retry helpers (TS: isRetryableError / getRetryAfterDelayMs / …)
// ---------------------------------------------------------------------------

fn is_terminal_rate_limit_error(error_text: &str) -> bool {
    let lower = error_text.to_lowercase();
    [
        "gousagelimiterror",
        "freeusagelimiterror",
        "monthly usage limit reached",
        "available balance",
        "insufficient_quota",
        "out of budget",
        "quota exceeded",
        "billing",
    ]
    .iter()
    .any(|p| lower.contains(p))
}

fn is_retryable_error(status: u16, error_text: &str) -> bool {
    if status == 429 && is_terminal_rate_limit_error(error_text) {
        return false;
    }
    if matches!(status, 429 | 500 | 502 | 503 | 504) {
        return true;
    }
    let lower = error_text.to_lowercase().replace([' ', '-'], "_");
    lower.contains("rate_limit")
        || lower.contains("ratelimit")
        || lower.contains("overloaded")
        || lower.contains("service_unavailable")
        || lower.contains("upstream_connect")
        || lower.contains("connection_refused")
}

fn get_retry_after_delay_ms(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    if let Some(v) = headers.get("retry-after-ms").and_then(|h| h.to_str().ok()) {
        if let Ok(millis) = v.trim().parse::<f64>() {
            if millis.is_finite() {
                return Some(millis.max(0.0) as u64);
            }
        }
    }
    if let Some(v) = headers.get("retry-after").and_then(|h| h.to_str().ok()) {
        let v = v.trim();
        if let Ok(seconds) = v.parse::<f64>() {
            return Some((seconds.max(0.0) * 1000.0) as u64);
        }
        if let Ok(date) = chrono::DateTime::parse_from_rfc2822(v) {
            let now = chrono::Utc::now().timestamp_millis();
            return Some((date.timestamp_millis() - now).max(0) as u64);
        }
    }
    None
}

fn validate_retry_delay_ms(delay_ms: u64, max_retry_delay_ms: Option<u64>) -> Result<u64, String> {
    let max = max_retry_delay_ms.unwrap_or(DEFAULT_MAX_RETRY_DELAY_MS);
    if max > 0 && delay_ms > max {
        return Err(format!(
            "Server requested {}s retry delay (max: {}s)",
            delay_ms.div_ceil(1000),
            max.div_ceil(1000)
        ));
    }
    Ok(delay_ms)
}

async fn wait_for_cancel(signal: Option<&CancellationToken>) {
    match signal {
        Some(s) => {
            while !s.is_cancelled() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
        None => {
            std::future::pending::<()>().await;
        }
    }
}

async fn check_cancelled(signal: Option<&CancellationToken>) -> Result<(), String> {
    if let Some(s) = signal {
        if s.is_cancelled() {
            return Err("Request was aborted".to_string());
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Error response parsing (TS: parseErrorResponse)
// ---------------------------------------------------------------------------

fn parse_error_response_body(status: u16, raw: &str) -> String {
    let mut message = if raw.is_empty() {
        "Request failed".to_string()
    } else {
        raw.to_string()
    };
    let mut friendly: Option<String> = None;

    if let Ok(parsed) = serde_json::from_str::<Value>(raw) {
        if let Some(err) = parsed.get("error") {
            let code = err
                .get("code")
                .and_then(|v| v.as_str())
                .or_else(|| err.get("type").and_then(|v| v.as_str()))
                .unwrap_or("");
            let is_rate = code.contains("usage_limit_reached")
                || code.contains("usage_not_included")
                || code.contains("rate_limit_exceeded")
                || status == 429;
            if is_rate {
                let plan = err
                    .get("plan_type")
                    .and_then(|v| v.as_str())
                    .map(|p| format!(" ({} plan)", p.to_lowercase()))
                    .unwrap_or_default();
                let mins = err
                    .get("resets_at")
                    .and_then(|v| v.as_i64())
                    .map(|ts| {
                        let reset_ms = ts.saturating_mul(1000);
                        let now = chrono::Utc::now().timestamp_millis();
                        ((reset_ms.saturating_sub(now)).max(0) as f64 / 60000.0).round() as i64
                    });
                let when = mins
                    .map(|m| format!(" Try again in ~{} min.", m))
                    .unwrap_or_default();
                friendly = Some(format!(
                    "You have hit your ChatGPT usage limit{}.{}",
                    plan, when
                ));
            }
            let msg = err.get("message").and_then(|v| v.as_str());
            if let Some(m) = msg.filter(|m| !m.is_empty()) {
                message = m.to_string();
            } else if let Some(f) = &friendly {
                message = f.clone();
            }
        }
    }
    message
}

// ---------------------------------------------------------------------------
// SSE request loop with retry
// ---------------------------------------------------------------------------

async fn process_sse_with_retry(
    model: &Model,
    opts: &CodexStreamOptions,
    headers: &reqwest::header::HeaderMap,
    body: &Value,
    output: &mut AssistantMessage,
    es: &AssistantMessageEventStream,
    grammar_tool_input_properties: &std::collections::HashMap<String, String>,
) -> Result<(), String> {
    let url = resolve_codex_url(opts.base_url.as_deref());
    let client = reqwest::Client::new();
    let max_retries = opts.max_retries.unwrap_or(DEFAULT_MAX_RETRIES);
    let body_json = serde_json::to_string(body).map_err(|e| e.to_string())?;

    // TS: `compressRequestBodyZstd` — the Codex backend accepts
    // `content-encoding: zstd` bodies. WebSocket frames stay uncompressed.
    #[cfg(feature = "codex-zstd")]
    let (sse_body, compressed) = match zstd::bulk::compress(body_json.as_bytes(), 3) {
        Ok(c) => (c, true),
        Err(_) => (body_json.clone().into_bytes(), false),
    };
    #[cfg(not(feature = "codex-zstd"))]
    let (sse_body, compressed) = (body_json.clone().into_bytes(), false);

    let mut attempt: u32 = 0;
    let response = loop {
        check_cancelled(opts.signal.as_ref()).await?;

        let mut request_headers = headers.clone();
        if compressed {
            request_headers.insert(
                reqwest::header::CONTENT_ENCODING,
                reqwest::header::HeaderValue::from_static("zstd"),
            );
        }
        let request = client
            .post(&url)
            .headers(request_headers)
            .body(sse_body.clone());

        let fetch_result = match opts.timeout_ms {
            Some(ms) if ms > 0 => {
                match tokio::time::timeout(Duration::from_millis(ms), request.send()).await {
                    Err(_) => {
                        return Err(format!(
                            "Codex SSE response headers timed out after {}ms",
                            ms
                        ));
                    }
                    Ok(r) => r,
                }
            }
            _ => request.send().await,
        };

        match fetch_result {
            Ok(resp) => {
                if resp.status().is_success() {
                    break resp;
                }
                let status = resp.status().as_u16();
                let headers_snapshot = resp.headers().clone();
                let error_text = resp.text().await.unwrap_or_default();
                if attempt < max_retries && is_retryable_error(status, &error_text) {
                    let retry_after = get_retry_after_delay_ms(&headers_snapshot);
                    let delay_ms = match retry_after {
                        Some(d) => validate_retry_delay_ms(d, opts.max_retry_delay_ms)?,
                        None => BASE_DELAY_MS * (1u64 << attempt),
                    };
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    attempt += 1;
                    continue;
                }
                return Err(parse_error_response_body(status, &error_text));
            }
            Err(e) => {
                // Network errors are retryable.
                if attempt < max_retries && !e.to_string().contains("usage limit") {
                    let delay_ms = BASE_DELAY_MS * (1u64 << attempt);
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    attempt += 1;
                    continue;
                }
                return Err(e.to_string());
            }
        }
    };

    es.push(AssistantMessageEvent::Start {
        partial: output.clone(),
    });

    let events = process_sse_bytes(response.bytes_stream(), opts.signal.clone());
    let started = std::sync::atomic::AtomicBool::new(true);
    process_codex_events(events, model, opts, output, es, &started, grammar_tool_input_properties)
        .await
        .map_err(|e| e.message)?;

    assert_successful_output(output).map_err(|e| e.message)
}

// ---------------------------------------------------------------------------
// SSE byte-stream parsing (TS: parseSSE — note pi-ai also only reads `data:`
// lines; `event:` lines are intentionally ignored, exactly like upstream)
// ---------------------------------------------------------------------------

fn process_sse_bytes<S, B>(
    byte_stream: S,
    signal: Option<CancellationToken>,
) -> std::pin::Pin<Box<dyn futures::Stream<Item = Result<Value, CodexStreamError>> + Send + 'static>>
where
    S: futures::Stream<Item = reqwest::Result<B>> + Send + 'static,
    B: AsRef<[u8]> + Send,
{
    Box::pin(async_stream::stream! {
        let mut byte_stream = std::pin::pin!(byte_stream);
        let mut carry: Vec<u8> = Vec::new();
        let mut pending = String::new();
        loop {
            let item = tokio::select! {
                i = byte_stream.next() => i,
                _ = wait_for_cancel(signal.as_ref()), if signal.is_some() => {
                    yield Err(CodexStreamError::other("Request was aborted"));
                    return;
                }
            };
            match item {
                None => break,
                Some(Err(e)) => {
                    yield Err(CodexStreamError::other(e.to_string()));
                    return;
                }
                Some(Ok(chunk)) => {
                    carry.extend_from_slice(chunk.as_ref());
                    let valid_up_to = valid_utf8_prefix_len(&carry);
                    pending.push_str(&String::from_utf8_lossy(&carry[..valid_up_to]));
                    carry.drain(..valid_up_to);
                    while let Some(idx) = pending.find("\n\n") {
                        let block: String = pending[..idx].to_string();
                        pending.drain(..idx + 2);
                        let data_lines: Vec<String> = block
                            .split('\n')
                            .filter(|l| l.starts_with("data:"))
                            .map(|l| l[5..].trim().to_string())
                            .collect();
                        if data_lines.is_empty() {
                            continue;
                        }
                        let data = data_lines.join("\n").trim().to_string();
                        if data.is_empty() || data == "[DONE]" {
                            continue;
                        }
                        match serde_json::from_str::<Value>(&data) {
                            Ok(v) => yield Ok(v),
                            Err(e) => {
                                yield Err(CodexStreamError::protocol(format!(
                                    "Invalid Codex SSE JSON: {e}"
                                )));
                                return;
                            }
                        }
                    }
                }
            }
        }
    })
}

fn valid_utf8_prefix_len(buf: &[u8]) -> usize {
    match std::str::from_utf8(buf) {
        Ok(_) => buf.len(),
        Err(e) => e.valid_up_to(),
    }
}

// ---------------------------------------------------------------------------
// WebSocket transport (TS: processWebSocketStream + cached continuation)
// ---------------------------------------------------------------------------

#[cfg(feature = "codex-websocket")]
pub use websocket::{
    close_codex_ws_sessions, get_codex_ws_debug_stats, reset_codex_ws_debug_stats,
    CodexWsDebugStats,
};

#[cfg(feature = "codex-websocket")]
pub(crate) mod websocket {
    use super::{
        assert_successful_output, build_websocket_headers, clamp_prompt_cache_key,
        convert_responses_messages, process_codex_events, resolve_codex_websocket_url,
        CodexErrorKind, CodexStreamError, CodexStreamOptions, CodexTransport,
        CODEX_TOOL_CALL_PROVIDERS,
    };
    use crate::types::{AssistantMessage, Message, Model};
    use crate::utils::event_stream::AssistantMessageEventStream;
    use futures::{SinkExt, StreamExt};
    use parking_lot::Mutex;
    use serde_json::{json, Value};
    use std::collections::{HashMap, HashSet};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, OnceLock};
    use std::time::{Duration, Instant};

    type WsStream =
        tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;
    type SocketSlot = Arc<Mutex<Option<WsStream>>>;

    const DEFAULT_CONNECT_TIMEOUT_MS: u64 = 15_000;
    const DEFAULT_CACHE_TTL_MS: u64 = 5 * 60 * 1000;
    const DEFAULT_MAX_AGE_MS: u64 = 55 * 60 * 1000;
    pub const WEBSOCKET_CONNECTION_LIMIT_REACHED: &str = "websocket_connection_limit_reached";
    pub const PREVIOUS_RESPONSE_NOT_FOUND: &str = "previous_response_not_found";

    // ---- session cache (TS: websocketSessionCache) ----
    struct ContinuationState {
        last_request_body: Value,
        last_response_id: String,
        last_response_items: Vec<Value>,
    }

    struct CachedEntry {
        socket: Option<WsStream>,
        created_at: Instant,
        continuation: Option<ContinuationState>,
        idle_timer: Option<tokio::task::JoinHandle<()>>,
    }

    type Entry = Arc<Mutex<CachedEntry>>;
    type SessionCache = HashMap<String, HashMap<String, Entry>>;

    fn session_cache() -> &'static Mutex<SessionCache> {
        static CACHE: OnceLock<Mutex<SessionCache>> = OnceLock::new();
        CACHE.get_or_init(|| Mutex::new(HashMap::new()))
    }

    fn sse_fallback_sessions() -> &'static Mutex<HashSet<String>> {
        static SET: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
        SET.get_or_init(|| Mutex::new(HashSet::new()))
    }

    // ---- debug stats (TS: OpenAICodexWebSocketDebugStats) ----
    #[derive(Debug, Clone, Default, PartialEq)]
    pub struct CodexWsDebugStats {
        pub requests: u64,
        pub connections_created: u64,
        pub connections_reused: u64,
        pub cached_context_requests: u64,
        pub store_true_requests: u64,
        pub full_context_requests: u64,
        pub delta_requests: u64,
        pub last_input_items: u64,
        pub last_delta_input_items: Option<u64>,
        pub last_previous_response_id: Option<String>,
        pub websocket_failures: u64,
        pub sse_fallbacks: u64,
        pub websocket_fallback_active: Option<bool>,
        pub last_websocket_error: Option<String>,
    }

    fn debug_stats_map() -> &'static Mutex<HashMap<String, CodexWsDebugStats>> {
        static MAP: OnceLock<Mutex<HashMap<String, CodexWsDebugStats>>> = OnceLock::new();
        MAP.get_or_init(|| Mutex::new(HashMap::new()))
    }

    fn mutate_stats<F: FnOnce(&mut CodexWsDebugStats)>(session: Option<&str>, f: F) {
        let Some(sid) = session else { return };
        let mut map = debug_stats_map().lock();
        f(map.entry(sid.to_string()).or_default());
    }

    /// TS: `getOpenAICodexWebSocketDebugStats`.
    pub fn get_codex_ws_debug_stats(session: &str) -> Option<CodexWsDebugStats> {
        let present = debug_stats_map().lock().contains_key(session);
        if !present {
            return None;
        }
        let mut stats = debug_stats_map().lock().get(session).cloned().unwrap_or_default();
        stats.websocket_fallback_active = Some(is_sse_fallback_active(Some(session)));
        Some(stats)
    }

    /// TS: `resetOpenAICodexWebSocketDebugStats`.
    pub fn reset_codex_ws_debug_stats(session: Option<&str>) {
        match session {
            Some(sid) => {
                debug_stats_map().lock().remove(sid);
                sse_fallback_sessions().lock().remove(sid);
            }
            None => {
                debug_stats_map().lock().clear();
                sse_fallback_sessions().lock().clear();
            }
        }
    }

    /// TS: `closeOpenAICodexWebSocketSessions` — drops cached sockets
    /// (closing the TCP connections) and clears continuation state.
    pub fn close_codex_ws_sessions(session: Option<&str>) {
        let mut cache = session_cache().lock();
        let retire = |entry: &Entry| {
            let mut e = entry.lock();
            if let Some(timer) = e.idle_timer.take() {
                timer.abort();
            }
            e.socket.take();
            e.continuation = None;
        };
        match session {
            Some(sid) => {
                if let Some(entries) = cache.get_mut(sid) {
                    for entry in entries.values() {
                        retire(entry);
                    }
                }
                cache.remove(sid);
            }
            None => {
                for entries in cache.values_mut() {
                    for entry in entries.values() {
                        retire(entry);
                    }
                }
                cache.clear();
            }
        }
    }

    pub(crate) fn is_sse_fallback_active(session: Option<&str>) -> bool {
        session
            .map(|sid| sse_fallback_sessions().lock().contains(sid))
            .unwrap_or(false)
    }

    pub(crate) fn record_sse_fallback(session: Option<&str>) {
        mutate_stats(session, |s| {
            s.sse_fallbacks += 1;
            s.websocket_fallback_active = Some(true);
        });
    }

    pub(crate) fn record_websocket_failure(session: Option<&str>, error: &str) {
        let Some(sid) = session else { return };
        sse_fallback_sessions().lock().insert(sid.to_string());
        mutate_stats(session, |s| {
            s.websocket_failures += 1;
            s.last_websocket_error = Some(error.to_string());
            s.websocket_fallback_active = Some(true);
        });
    }

    // ---- errors ----
    pub struct WebSocketRunError {
        pub message: String,
        /// True once the `start` event was emitted on the output stream;
        /// failures after this point must not silently retry (pi-ai
        /// `phase: after_message_stream_start`).
        pub started: bool,
        pub kind: CodexErrorKind,
        pub code: Option<String>,
    }

    fn transport_err(message: impl Into<String>) -> WebSocketRunError {
        WebSocketRunError {
            message: message.into(),
            started: false,
            kind: CodexErrorKind::Other,
            code: None,
        }
    }

    // ---- connect ----
    async fn connect(
        url: &str,
        headers: &reqwest::header::HeaderMap,
        connect_timeout_ms: u64,
    ) -> Result<WsStream, WebSocketRunError> {
        let sec_websocket_key = {
            use rand::RngCore;
            let mut bytes = [0u8; 16];
            rand::thread_rng().fill_bytes(&mut bytes);
            const TABLE: &[u8] =
                b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
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
            out
        };
        let authority = url
            .trim_start_matches("wss://")
            .trim_start_matches("ws://")
            .split('/')
            .next()
            .unwrap_or_default()
            .to_string();
        let mut request = http::Request::builder()
            .method("GET")
            .uri(url)
            .header("host", authority)
            .header("connection", "Upgrade")
            .header("upgrade", "websocket")
            .header("sec-websocket-version", "13")
            .header("sec-websocket-key", sec_websocket_key);
        for (k, v) in headers.iter() {
            request = request.header(k.as_str(), v.to_str().unwrap_or_default());
        }
        let request = request
            .body(())
            .map_err(|e| transport_err(format!("WebSocket request build failed: {e}")))?;

        let connect_fut = tokio_tungstenite::connect_async(request);
        match tokio::time::timeout(Duration::from_millis(connect_timeout_ms), connect_fut).await {
            Err(_) => Err(transport_err(format!(
                "WebSocket connect timeout after {}ms",
                connect_timeout_ms
            ))),
            Ok(Err(e)) => Err(transport_err(format!("WebSocket connect failed: {e}"))),
            Ok(Ok((ws, _))) => Ok(ws),
        }
    }

    struct Acquired {
        socket_slot: SocketSlot,
        /// Present when this connection is tracked in the session cache.
        entry: Option<Entry>,
        entry_key: Option<(String, String)>,
        reused: bool,
    }

    async fn acquire(
        url: &str,
        headers: &reqwest::header::HeaderMap,
        session: Option<&str>,
        account_id: &str,
        opts: &CodexStreamOptions,
    ) -> Result<Acquired, WebSocketRunError> {
        let connect_timeout = opts
            .websocket_connect_timeout_ms
            .unwrap_or(DEFAULT_CONNECT_TIMEOUT_MS);

        let Some(sid) = session else {
            // One-shot connection (never cached).
            let socket = connect(url, headers, connect_timeout).await?;
            return Ok(Acquired {
                socket_slot: Arc::new(Mutex::new(Some(socket))),
                entry: None,
                entry_key: None,
                reused: false,
            });
        };

        // Reuse path.
        enum CacheDecision {
            Reuse(Entry, Box<WsStream>),
            Busy,
            Retire(Entry),
            Miss,
        }
        let decision = {
            let mut cache = session_cache().lock();
            let entry = cache
                .get_mut(sid)
                .and_then(|entries| entries.get(account_id).cloned());
            match entry {
                None => CacheDecision::Miss,
                Some(entry) => {
                    let mut e = entry.lock();
                    if let Some(timer) = e.idle_timer.take() {
                        timer.abort();
                    }
                    let max_age = opts.websocket_max_age_ms.unwrap_or(DEFAULT_MAX_AGE_MS);
                    let expired = Instant::now().duration_since(e.created_at)
                        >= Duration::from_millis(max_age);
                    // Take the socket out of the entry inside the lock so the
                    // Reuse arm below cannot race with another acquirer.
                    let socket = e.socket.take();
                    match socket {
                        Some(sock) if !expired => {
                            drop(e);
                            CacheDecision::Reuse(entry, Box::new(sock))
                        }
                        _ if expired => {
                            drop(e);
                            CacheDecision::Retire(entry)
                        }
                        _ => {
                            drop(e);
                            CacheDecision::Busy
                        }
                    }
                }
            }
        };

        match decision {
            CacheDecision::Reuse(entry, socket) => {
                return Ok(Acquired {
                    socket_slot: Arc::new(Mutex::new(Some(*socket))),
                    entry: Some(entry),
                    entry_key: Some((sid.to_string(), account_id.to_string())),
                    reused: true,
                });
            }
            CacheDecision::Retire(entry) => {
                retire_entry(&(sid.to_string(), account_id.to_string()), &entry);
                // Fall through to a fresh cached connection.
            }
            CacheDecision::Busy => {
                // Serve this request on a one-shot connection.
                let socket = connect(url, headers, connect_timeout).await?;
                return Ok(Acquired {
                    socket_slot: Arc::new(Mutex::new(Some(socket))),
                    entry: None,
                    entry_key: None,
                    reused: false,
                });
            }
            CacheDecision::Miss => {}
        }

        // Fresh cached connection.
        let socket = connect(url, headers, connect_timeout).await?;
        let entry: Entry = Arc::new(Mutex::new(CachedEntry {
            socket: None,
            created_at: Instant::now(),
            continuation: None,
            idle_timer: None,
        }));
        session_cache()
            .lock()
            .entry(sid.to_string())
            .or_default()
            .insert(account_id.to_string(), entry.clone());
        Ok(Acquired {
            socket_slot: Arc::new(Mutex::new(Some(socket))),
            entry: Some(entry),
            entry_key: Some((sid.to_string(), account_id.to_string())),
            reused: false,
        })
    }

    fn retire_entry(key: &(String, String), entry: &Entry) {
        let mut cache = session_cache().lock();
        if let Some(entries) = cache.get_mut(&key.0) {
            if let Some(current) = entries.get(&key.1) {
                if Arc::ptr_eq(current, entry) {
                    let mut e = entry.lock();
                    if let Some(timer) = e.idle_timer.take() {
                        timer.abort();
                    }
                    e.socket.take();
                    e.continuation = None;
                    drop(e);
                    entries.remove(&key.1);
                    if entries.is_empty() {
                        cache.remove(&key.0);
                    }
                }
            }
        }
    }

    /// Return the socket to the cache entry and schedule the idle expiry
    /// (TS: `release({keep: true})`).
    fn release_keep(acquired: &Acquired, ttl_ms: u64) {
        let Some(key) = &acquired.entry_key else {
            return; // one-shot: dropping the slot closes the connection
        };
        let Some(entry) = &acquired.entry else {
            return;
        };
        {
            let mut e = entry.lock();
            if let Some(timer) = e.idle_timer.take() {
                timer.abort();
            }
            e.socket = acquired.socket_slot.lock().take();
        }
        let key = key.clone();
        let entry_clone = entry.clone();
        let timer = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(ttl_ms)).await;
            let mut cache = session_cache().lock();
            if let Some(entries) = cache.get_mut(&key.0) {
                if let Some(current) = entries.get(&key.1) {
                    if Arc::ptr_eq(current, &entry_clone) {
                        let mut e = entry_clone.lock();
                        // Only retire when idle (the socket is home).
                        if e.socket.is_some() {
                            e.socket.take();
                            e.continuation = None;
                            drop(e);
                            entries.remove(&key.1);
                            if entries.is_empty() {
                                cache.remove(&key.0);
                            }
                        }
                    }
                }
            }
        });
        entry.lock().idle_timer = Some(timer);
    }

    fn release(acquired: &Acquired, keep: bool, opts: &CodexStreamOptions) {
        if !keep {
            // Drop the socket (closing the connection) and drop the entry.
            acquired.socket_slot.lock().take();
            if let (Some(key), Some(entry)) = (&acquired.entry_key, &acquired.entry) {
                retire_entry(key, entry);
            }
            return;
        }
        let ttl = opts.websocket_cache_ttl_ms.unwrap_or(DEFAULT_CACHE_TTL_MS);
        release_keep(acquired, ttl);
    }

    // ---- cached-context request bodies (TS: buildCachedWebSocketRequestBody) ----
    fn request_body_without_input(body: &Value) -> Value {
        let mut clone = body.clone();
        if let Some(obj) = clone.as_object_mut() {
            obj.remove("input");
            obj.remove("previous_response_id");
        }
        clone
    }

    fn request_bodies_match_except_input(a: &Value, b: &Value) -> bool {
        request_body_without_input(a) == request_body_without_input(b)
    }

    fn get_cached_input_delta(body: &Value, cont: &ContinuationState) -> Option<Vec<Value>> {
        if !request_bodies_match_except_input(body, &cont.last_request_body) {
            return None;
        }
        let current = body.get("input")?.as_array()?;
        let mut baseline = cont
            .last_request_body
            .get("input")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        baseline.extend(cont.last_response_items.iter().cloned());
        if current.len() < baseline.len() {
            return None;
        }
        if serde_json::to_string(&current[..baseline.len()]).ok()
            != serde_json::to_string(&baseline).ok()
        {
            return None;
        }
        Some(current[baseline.len()..].to_vec())
    }

    fn build_cached_request_body(entry: &Entry, body: &Value) -> Value {
        let mut e = entry.lock();
        let Some(cont) = &e.continuation else {
            return body.clone();
        };
        let Some(delta) = get_cached_input_delta(body, cont) else {
            e.continuation = None;
            return body.clone();
        };
        let mut out = body.clone();
        out["previous_response_id"] = json!(&cont.last_response_id);
        out["input"] = Value::Array(delta);
        out
    }

    fn is_terminal_event(v: &Value) -> bool {
        matches!(
            v.get("type").and_then(|t| t.as_str()),
            Some("response.completed") | Some("response.done") | Some("response.incomplete")
        )
    }

    // ---- main ----
    #[allow(clippy::too_many_arguments)]
    pub async fn run_websocket(
        model: &Model,
        opts: &CodexStreamOptions,
        body: &Value,
        account_id: &str,
        cache_session_id: Option<&str>,
        es: &AssistantMessageEventStream,
        output: &mut AssistantMessage,
        grammar_tool_input_properties: &std::collections::HashMap<String, String>,
    ) -> Result<(), WebSocketRunError> {
        let url = resolve_codex_websocket_url(opts.base_url.as_deref());
        let request_id = cache_session_id
            .map(clamp_prompt_cache_key)
            .unwrap_or_else(crate::types::uuidv7);
        let headers = build_websocket_headers(
            &opts.extra_headers,
            account_id,
            &opts.api_key,
            &request_id,
        );

        if let Some(s) = &opts.signal {
            if s.is_cancelled() {
                return Err(transport_err("Request was aborted"));
            }
        }

        let acquired = acquire(&url, &headers, cache_session_id, account_id, opts).await?;
        let use_cached_context = matches!(
            opts.transport,
            CodexTransport::Auto | CodexTransport::WebsocketCached
        );

        let request_body = match (&acquired.entry, use_cached_context) {
            (Some(entry), true) => build_cached_request_body(entry, body),
            _ => body.clone(),
        };

        // Debug stats (TS parity).
        mutate_stats(cache_session_id, |s| {
            s.requests += 1;
            if acquired.reused {
                s.connections_reused += 1;
            } else {
                s.connections_created += 1;
            }
            if use_cached_context {
                s.cached_context_requests += 1;
            }
            if request_body.get("store").and_then(|v| v.as_bool()) == Some(true) {
                s.store_true_requests += 1;
            }
            let input_len = request_body
                .get("input")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0) as u64;
            s.last_input_items = input_len;
            if let Some(prev) = request_body.get("previous_response_id").and_then(|v| v.as_str()) {
                s.delta_requests += 1;
                s.last_delta_input_items = Some(input_len);
                s.last_previous_response_id = Some(prev.to_string());
            } else {
                s.full_context_requests += 1;
                s.last_delta_input_items = None;
                s.last_previous_response_id = None;
            }
        });

        // Frame: `{type: "response.create", …requestBody}` (uncompressed
        // JSON, matching the official Codex client).
        let mut frame_map = serde_json::Map::new();
        frame_map.insert("type".to_string(), json!("response.create"));
        if let Some(obj) = request_body.as_object() {
            frame_map.extend(obj.clone());
        }
        let frame_text = Value::Object(frame_map).to_string();

        // Reader task: owns the socket, feeds parsed events into the channel,
        // re-homes the socket in the slot after a clean terminal event.
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Value, CodexStreamError>>(64);
        let started_flag = Arc::new(AtomicBool::new(false));
        let socket_slot = acquired.socket_slot.clone();
        let idle_timeout_ms = opts.timeout_ms.unwrap_or(0);
        let signal = opts.signal.clone();
        let reader = tokio::spawn(async move {
            let Some(mut sock) = socket_slot.lock().take() else {
                return;
            };
            if sock
                .send(tokio_tungstenite::tungstenite::Message::Text(frame_text))
                .await
                .is_err()
            {
                let _ = tx
                    .send(Err(CodexStreamError::other("WebSocket send failed")))
                    .await;
                return;
            }
            loop {
                if let Some(s) = &signal {
                    if s.is_cancelled() {
                        let _ = tx
                            .send(Err(CodexStreamError::other("Request was aborted")))
                            .await;
                        return; // socket dropped → closed
                    }
                }
                let next = sock.next();
                let msg = if idle_timeout_ms > 0 {
                    match tokio::time::timeout(Duration::from_millis(idle_timeout_ms), next).await {
                        Err(_) => {
                            let _ = tx
                                .send(Err(CodexStreamError::other(format!(
                                    "WebSocket idle timeout after {}ms",
                                    idle_timeout_ms
                                ))))
                                .await;
                            return;
                        }
                        Ok(m) => m,
                    }
                } else {
                    next.await
                };
                match msg {
                    None => {
                        let _ = tx
                            .send(Err(CodexStreamError::other(
                                "WebSocket stream closed before response.completed",
                            )))
                            .await;
                        return;
                    }
                    Some(Ok(tokio_tungstenite::tungstenite::Message::Text(text))) => {
                        let parsed = match serde_json::from_str::<Value>(&text) {
                            Ok(v) => v,
                            Err(e) => {
                                let _ = tx
                                    .send(Err(CodexStreamError::protocol(format!(
                                        "Invalid Codex WebSocket JSON: {e}"
                                    ))))
                                    .await;
                                return;
                            }
                        };
                        let terminal = is_terminal_event(&parsed);
                        if terminal {
                            // Re-home BEFORE forwarding the terminal event so
                            // the consumer's `release` always finds the socket
                            // in the slot (no race with the reader).
                            *socket_slot.lock() = Some(sock);
                            let _ = tx.send(Ok(parsed)).await;
                            break;
                        }
                        if tx.send(Ok(parsed)).await.is_err() {
                            // Consumer gone; re-home so the connection can be
                            // reused rather than dropped.
                            *socket_slot.lock() = Some(sock);
                            break;
                        }
                    }
                    Some(Ok(tokio_tungstenite::tungstenite::Message::Close(_))) => {
                        let _ = tx
                            .send(Err(CodexStreamError::other(
                                "WebSocket stream closed before response.completed",
                            )))
                            .await;
                        return;
                    }
                    Some(Ok(_)) => continue,
                    Some(Err(e)) => {
                        let _ = tx
                            .send(Err(CodexStreamError::other(format!(
                                "WebSocket error: {e}"
                            ))))
                            .await;
                        return;
                    }
                }
            }
        });

        let events = tokio_stream::wrappers::ReceiverStream::new(rx);
        let result = process_codex_events(
            events,
            model,
            opts,
            output,
            es,
            &started_flag,
            grammar_tool_input_properties,
        )
        .await;
        let _ = &reader; // keep the reader alive through processing
        let started = started_flag.load(Ordering::SeqCst);

        match result {
            Ok(()) => {
                let cancelled = opts
                    .signal
                    .as_ref()
                    .map(|s| s.is_cancelled())
                    .unwrap_or(false);
                let keep = !cancelled;
                if keep && use_cached_context {
                    let recorded = match output {
                        Message::Assistant {
                            response_id: Some(rid),
                            ..
                        } => Some((rid.clone(), output.clone())),
                        _ => None,
                    };
                    if let Some((rid, snapshot)) = recorded {
                        if let Some(entry) = &acquired.entry {
                            // TS: convertResponsesMessages(model, {messages:
                            // [output]}, …) minus call outputs.
                            let ctx = crate::types::Context {
                                messages: vec![snapshot],
                                ..Default::default()
                            };
                            let items: Vec<Value> = convert_responses_messages(
                                model,
                                &ctx,
                                CODEX_TOOL_CALL_PROVIDERS,
                                false,
                                grammar_tool_input_properties,
                            )
                            .into_iter()
                            .filter(|item| {
                                let t = item.get("type").and_then(|v| v.as_str());
                                t != Some("function_call_output")
                                    && t != Some("custom_tool_call_output")
                            })
                            .collect();
                            entry.lock().continuation = Some(ContinuationState {
                                last_request_body: body.clone(),
                                last_response_id: rid,
                                last_response_items: items,
                            });
                        }
                    }
                }
                release(&acquired, keep, opts);
                assert_successful_output(output).map_err(|e| WebSocketRunError {
                    message: e.message,
                    started,
                    kind: e.kind,
                    code: e.code,
                })
            }
            Err(err) => {
                reader.abort();
                if let Some(entry) = &acquired.entry {
                    entry.lock().continuation = None;
                }
                release(&acquired, false, opts);
                Err(WebSocketRunError {
                    message: err.message,
                    started,
                    kind: err.kind,
                    code: err.code,
                })
            }
        }
    }

}



// ---------------------------------------------------------------------------
// Codex event mapping (TS: mapCodexEvents) + projection (processResponsesStream)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotKind {
    Thinking,
    Text,
    ToolCall,
    Other,
}

#[derive(Debug, Clone)]
struct Slot {
    kind: SlotKind,
    content_index: usize,
    /// Streaming scratch buffer for tool-call arguments (TS: `partialJson`).
    partial_json: Option<String>,
    /// True for `custom_tool_call` items; their input streams through the
    /// `response.custom_tool_call_input.*` events instead of arguments.
    custom_input: bool,
}

async fn process_codex_events<S>(
    mut events: S,
    model: &Model,
    opts: &CodexStreamOptions,
    output: &mut AssistantMessage,
    es: &AssistantMessageEventStream,
    started: &std::sync::atomic::AtomicBool,
    grammar_tool_input_properties: &std::collections::HashMap<String, String>,
) -> Result<(), CodexStreamError>
where
    S: futures::Stream<Item = Result<Value, CodexStreamError>> + Unpin,
{
    let mut saw_tool_call = false;
    let mut slots: HashMap<usize, Slot> = HashMap::new();

    while let Some(item) = events.next().await {
        let event = item?;
        let evt_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if evt_type.is_empty() {
            continue;
        }

        // TS: `startWebSocketOutputOnFirstEvent` sits *after* mapCodexEvents,
        // so events that throw before being yielded (error / response.failed
        // frames, bad JSON) never trigger the `start` event.
        if evt_type != "error"
            && evt_type != "response.failed"
            && !started.swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            es.push(AssistantMessageEvent::Start {
                partial: output.clone(),
            });
        }

        match evt_type {
            "error" => {
                let nested = event.get("error").filter(|v| v.is_object());
                let code = event
                    .get("code")
                    .and_then(|v| v.as_str())
                    .or_else(|| {
                        nested
                            .and_then(|n| n.get("code"))
                            .and_then(|v| v.as_str())
                    });
                let message = event
                    .get("message")
                    .and_then(|v| v.as_str())
                    .or_else(|| {
                        nested
                            .and_then(|n| n.get("message"))
                            .and_then(|v| v.as_str())
                    });
                let fallback = event.to_string();
                return Err(CodexStreamError::api(
                    format!(
                        "Codex error: {}",
                        message.or(code).unwrap_or(&fallback)
                    ),
                    code.map(|c| c.to_string()),
                ));
            }
            "response.failed" => {
                let response = event.get("response");
                let code = response
                    .and_then(|r| r.get("error"))
                    .and_then(|e| e.get("code"))
                    .and_then(|v| v.as_str());
                let message = response
                    .and_then(|r| r.get("error"))
                    .and_then(|e| e.get("message"))
                    .and_then(|v| v.as_str());
                return Err(CodexStreamError::api(
                    message
                        .map(|m| m.to_string())
                        .or_else(|| code.map(|c| c.to_string()))
                        .unwrap_or_else(|| "Codex response failed".to_string()),
                    code.map(|c| c.to_string()),
                ));
            }
            "response.done" | "response.completed" | "response.incomplete" => {
                // TS: normalize status + capture end_turn.
                let mut response = event.get("response").cloned().unwrap_or(Value::Null);
                if let Some(end_turn) = response.get("end_turn").and_then(|v| v.as_bool()) {
                    if let Message::Assistant { end_turn: et, .. } = output {
                        *et = Some(end_turn);
                    }
                }
                match normalize_codex_status(response.get("status").and_then(|v| v.as_str())) {
                    Some(status) => response["status"] = json!(status),
                    None => {
                        response
                            .as_object_mut()
                            .map(|m| m.remove("status"));
                    }
                }
                finalize_response(&response, model, opts, output, saw_tool_call, es)?;
                return Ok(());
            }
            "response.created" => {
                if let Some(id) = event.pointer("/response/id").and_then(|v| v.as_str()) {
                    if let Message::Assistant { response_id, .. } = output {
                        *response_id = Some(id.to_string());
                    }
                }
            }
            "response.output_item.added" => {
                let output_index =
                    event.get("output_index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let item = event.get("item").cloned().unwrap_or(Value::Null);
                create_slot(
                    output,
                    &mut slots,
                    es,
                    output_index,
                    &item,
                    &mut saw_tool_call,
                    grammar_tool_input_properties,
                );
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                let output_index =
                    event.get("output_index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let delta = event.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                if let Some(slot) = slots.get(&output_index).filter(|s| s.kind == SlotKind::Thinking) {
                    append_thinking(output, slot.content_index, delta);
                    es.push(AssistantMessageEvent::ThinkingDelta {
                        delta: delta.to_string(),
                    });
                }
            }
            "response.reasoning_summary_part.done" => {
                let output_index =
                    event.get("output_index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                if let Some(slot) = slots.get(&output_index).filter(|s| s.kind == SlotKind::Thinking) {
                    append_thinking(output, slot.content_index, "\n\n");
                    es.push(AssistantMessageEvent::ThinkingDelta {
                        delta: "\n\n".to_string(),
                    });
                }
            }
            "response.output_text.delta" | "response.refusal.delta" => {
                let output_index =
                    event.get("output_index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let delta = event.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                if let Some(slot) = slots.get(&output_index).filter(|s| s.kind == SlotKind::Text) {
                    append_text(output, slot.content_index, delta);
                    es.push(AssistantMessageEvent::TextDelta {
                        delta: delta.to_string(),
                    });
                }
            }
            "response.function_call_arguments.delta" => {
                let output_index =
                    event.get("output_index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let delta = event.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                if let Some(slot) = slots
                    .get_mut(&output_index)
                    .filter(|s| s.kind == SlotKind::ToolCall && !s.custom_input)
                {
                    let ci = slot.content_index;
                    let mut partial = slot.partial_json.clone().unwrap_or_default();
                    partial.push_str(delta);
                    if let Ok(Some(parsed)) =
                        crate::utils::json_parse::parse_partial_json(&partial)
                    {
                        set_tool_arguments(output, ci, parsed);
                    }
                    slot.partial_json = Some(partial);
                    es.push(AssistantMessageEvent::ToolcallDelta {
                        content_index: ci,
                        delta: delta.to_string(),
                    });
                }
            }
            "response.function_call_arguments.done" => {
                let output_index =
                    event.get("output_index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let arguments = event.get("arguments").and_then(|v| v.as_str()).unwrap_or("");
                if let Some(slot) = slots
                    .get_mut(&output_index)
                    .filter(|s| s.kind == SlotKind::ToolCall && !s.custom_input)
                {
                    let ci = slot.content_index;
                    let prev = slot.partial_json.clone().unwrap_or_default();
                    if let Ok(Some(parsed)) =
                        crate::utils::json_parse::parse_partial_json(arguments)
                    {
                        set_tool_arguments(output, ci, parsed);
                    }
                    if let Some(extra) = arguments.strip_prefix(&prev) {
                        if !extra.is_empty() {
                            es.push(AssistantMessageEvent::ToolcallDelta {
                                content_index: ci,
                                delta: extra.to_string(),
                            });
                        }
                    }
                    slot.partial_json = Some(arguments.to_string());
                }
            }
            "response.custom_tool_call_input.delta" => {
                let output_index =
                    event.get("output_index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let delta = event.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                if let Some(slot) = slots
                    .get_mut(&output_index)
                    .filter(|s| s.kind == SlotKind::ToolCall && s.custom_input)
                {
                    let ci = slot.content_index;
                    let mut partial = slot.partial_json.clone().unwrap_or_default();
                    partial.push_str(delta);
                    set_custom_tool_input(output, ci, &partial);
                    slot.partial_json = Some(partial);
                    es.push(AssistantMessageEvent::ToolcallDelta {
                        content_index: ci,
                        delta: delta.to_string(),
                    });
                }
            }
            "response.custom_tool_call_input.done" => {
                let output_index =
                    event.get("output_index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let input = event.get("input").and_then(|v| v.as_str()).unwrap_or("");
                if let Some(slot) = slots
                    .get_mut(&output_index)
                    .filter(|s| s.kind == SlotKind::ToolCall && s.custom_input)
                {
                    let ci = slot.content_index;
                    let prev = slot.partial_json.clone().unwrap_or_default();
                    set_custom_tool_input(output, ci, input);
                    if let Some(extra) = input.strip_prefix(&prev) {
                        if !extra.is_empty() {
                            es.push(AssistantMessageEvent::ToolcallDelta {
                                content_index: ci,
                                delta: extra.to_string(),
                            });
                        }
                    }
                    slot.partial_json = Some(input.to_string());
                }
            }
            "response.output_item.done" => {
                let output_index =
                    event.get("output_index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let item = event.get("item").cloned().unwrap_or(Value::Null);
                let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                let existing = slots.get(&output_index).cloned();
                match item_type {
                    "reasoning" => {
                        if let Some(slot) = existing.filter(|s| s.kind == SlotKind::Thinking) {
                            let summary = string_array_join(&item, "summary");
                            let content_text = string_array_join(&item, "content");
                            let signature = serde_json::to_string(&item).unwrap_or_default();
                            if let Some(ContentBlock::Thinking {
                                ref mut thinking,
                                signature: ref mut sig,
                            }) = content_block_mut(output, slot.content_index)
                            {
                                if !summary.is_empty() {
                                    *thinking = summary.clone();
                                } else if !content_text.is_empty() {
                                    *thinking = content_text;
                                }
                                *sig = Some(signature.clone());
                            }
                            es.push(AssistantMessageEvent::ThinkingEnd {
                                signature: Some(signature),
                            });
                            slots.remove(&output_index);
                        }
                    }
                    "message" => {
                        let slot = match existing.filter(|s| s.kind == SlotKind::Text) {
                            Some(s) => s,
                            None => {
                                let mut saw = saw_tool_call;
                                let created = create_slot(
                                    output,
                                    &mut slots,
                                    es,
                                    output_index,
                                    &item,
                                    &mut saw,
                                    grammar_tool_input_properties,
                                );
                                saw_tool_call = saw;
                                created
                            }
                        };
                        if slot.kind == SlotKind::Text {
                            let text = item
                                .get("content")
                                .and_then(|v| v.as_array())
                                .map(|arr| {
                                    arr.iter()
                                        .filter_map(|c| {
                                            c.get("text")
                                                .and_then(|t| t.as_str())
                                                .or_else(|| {
                                                    c.get("refusal").and_then(|t| t.as_str())
                                                })
                                        })
                                        .collect::<Vec<_>>()
                                        .join("")
                                })
                                .unwrap_or_default();
                            if let Some(ContentBlock::Text {
                                text: ref mut t,
                                text_signature: ref mut sig,
                            }) = content_block_mut(output, slot.content_index)
                            {
                                *t = text;
                                // TS: encodeTextSignatureV1(item.id, phase) —
                                // keeps the server-assigned message id stable
                                // across replays.
                                let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
                                let phase = item.get("phase").and_then(|v| v.as_str());
                                if !id.is_empty() {
                                    *sig = Some(encode_text_signature_v1(id, phase));
                                }
                            }
                            es.push(AssistantMessageEvent::TextEnd);
                            slots.remove(&output_index);
                        }
                    }
                    "function_call" => {
                        if let Some(slot) = existing.filter(|s| s.kind == SlotKind::ToolCall && !s.custom_input) {
                            let arguments_str = item
                                .get("arguments")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string())
                                .or_else(|| slot.partial_json.clone())
                                .unwrap_or_else(|| "{}".to_string());
                            if let Ok(Some(parsed)) =
                                crate::utils::json_parse::parse_partial_json(&arguments_str)
                            {
                                set_tool_arguments(output, slot.content_index, parsed);
                            }
                            let block = content_block(output, slot.content_index)
                                .unwrap_or(ContentBlock::Text { text: String::new(), text_signature: None});
                            es.push(AssistantMessageEvent::ToolcallEnd {
                                content_index: slot.content_index,
                                tool_call: block,
                            });
                            slots.remove(&output_index);
                        }
                    }
                    "custom_tool_call" => {
                        if let Some(slot) =
                            existing.filter(|s| s.kind == SlotKind::ToolCall && s.custom_input)
                        {
                            let input_str = item
                                .get("input")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string())
                                .or_else(|| slot.partial_json.clone())
                                .unwrap_or_default();
                            set_custom_tool_input(output, slot.content_index, &input_str);
                            let block = content_block(output, slot.content_index)
                                .unwrap_or(ContentBlock::Text { text: String::new(), text_signature: None});
                            es.push(AssistantMessageEvent::ToolcallEnd {
                                content_index: slot.content_index,
                                tool_call: block,
                            });
                            slots.remove(&output_index);
                        }
                    }
                    _ => {}
                }
            }
            _ => {
                // Unknown event types are ignored (forward compatibility).
            }
        }
    }

    Err(CodexStreamError::other(
        "OpenAI Responses stream ended before a terminal response event",
    ))
}

fn string_array_join(item: &Value, key: &str) -> String {
    item.get(key)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|s| s.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n\n")
        })
        .unwrap_or_default()
}

fn normalize_codex_status(status: Option<&str>) -> Option<String> {
    const KNOWN: &[&str] = &[
        "completed",
        "incomplete",
        "failed",
        "cancelled",
        "queued",
        "in_progress",
    ];
    status.filter(|s| KNOWN.contains(s)).map(|s| s.to_string())
}

fn create_slot(
    output: &mut AssistantMessage,
    slots: &mut HashMap<usize, Slot>,
    es: &AssistantMessageEventStream,
    output_index: usize,
    item: &Value,
    saw_tool_call: &mut bool,
    grammar_tool_input_properties: &std::collections::HashMap<String, String>,
) -> Slot {
    let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let slot = match item_type {
        "reasoning" => {
            if let Message::Assistant { content, .. } = output {
                content.push(ContentBlock::Thinking {
                    thinking: String::new(),
                    signature: None,
                });
            }
            es.push(AssistantMessageEvent::ThinkingStart);
            Slot {
                kind: SlotKind::Thinking,
                content_index: content_len(output).saturating_sub(1),
                partial_json: None,
                custom_input: false,
            }
        }
        "message" => {
            if let Message::Assistant { content, .. } = output {
                content.push(ContentBlock::Text { text: String::new(), text_signature: None});
            }
            es.push(AssistantMessageEvent::TextStart);
            Slot {
                kind: SlotKind::Text,
                content_index: content_len(output).saturating_sub(1),
                partial_json: None,
                custom_input: false,
            }
        }
        "function_call" => {
            let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
            let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let combined_id = format!("{}|{}", call_id, id);
            if let Message::Assistant { content, .. } = output {
                content.push(ContentBlock::ToolCall {
                    id: combined_id.clone(),
                    name: name.to_string(),
                    arguments: json!({}),
                });
            }
            *saw_tool_call = true;
            let content_index = content_len(output).saturating_sub(1);
            // Seed the scratch buffer with any arguments already present on
            // the `added` event (TS: `partialJson: item.arguments || ""`).
            let initial = item
                .get("arguments")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            es.push(AssistantMessageEvent::ToolcallStart {
                content_index,
                id: Some(combined_id),
                name: Some(name.to_string()),
            });
            Slot {
                kind: SlotKind::ToolCall,
                content_index,
                partial_json: Some(initial),
                custom_input: false,
            }
        }
        "custom_tool_call" => {
            let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
            let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let combined_id = format!("{}|{}", call_id, id);
            let property = grammar_tool_input_properties
                .get(name)
                .cloned()
                .unwrap_or_else(|| "input".to_string());
            let initial_input = item
                .get("input")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if let Message::Assistant { content, .. } = output {
                content.push(ContentBlock::ToolCall {
                    id: combined_id.clone(),
                    name: name.to_string(),
                    arguments: json!({ property: initial_input }),
                });
            }
            *saw_tool_call = true;
            let content_index = content_len(output).saturating_sub(1);
            es.push(AssistantMessageEvent::ToolcallStart {
                content_index,
                id: Some(combined_id),
                name: Some(name.to_string()),
            });
            Slot {
                kind: SlotKind::ToolCall,
                content_index,
                partial_json: Some(initial_input),
                custom_input: true,
            }
        }
        _ => Slot {
            kind: SlotKind::Other,
            content_index: usize::MAX,
            partial_json: None,
            custom_input: false,
        },
    };
    slots.insert(output_index, slot.clone());
    slot
}

fn content_len(output: &AssistantMessage) -> usize {
    match output {
        Message::Assistant { content, .. } => content.len(),
        _ => 0,
    }
}

fn content_block_mut(
    output: &mut AssistantMessage,
    index: usize,
) -> Option<&mut ContentBlock> {
    match output {
        Message::Assistant { content, .. } => content.get_mut(index),
        _ => None,
    }
}

fn content_block(output: &AssistantMessage, index: usize) -> Option<ContentBlock> {
    match output {
        Message::Assistant { content, .. } => content.get(index).cloned(),
        _ => None,
    }
}

fn append_thinking(output: &mut AssistantMessage, index: usize, delta: &str) {
    if let Some(ContentBlock::Thinking { thinking, .. }) = content_block_mut(output, index) {
        thinking.push_str(delta);
    }
}

fn append_text(output: &mut AssistantMessage, index: usize, delta: &str) {
    if let Some(ContentBlock::Text { text, .. }) = content_block_mut(output, index) {
        text.push_str(delta);
    }
}

fn set_tool_arguments(output: &mut AssistantMessage, index: usize, arguments: Value) {
    if let Some(ContentBlock::ToolCall {
        arguments: ref mut a,
        ..
    }) = content_block_mut(output, index)
    {
        *a = arguments;
    }
}

/// TS: `appendCustomToolCallInput` — the custom tool input is kept as a raw
/// string under `{ <property>: <raw> }` (grammar JSON buffering is
/// approximated by keeping the raw string; streaming diffs still flow through
/// ToolcallDelta). `property` mirrors the tool's inferred input property.
fn set_custom_tool_input(output: &mut AssistantMessage, index: usize, input: &str) {
    if let Some(ContentBlock::ToolCall {
        arguments: ref mut a,
        ..
    }) = content_block_mut(output, index)
    {
        // Keep the stored property: the block was created with the inferred
        // input property, so reuse the existing key when present.
        let property = a
            .as_object()
            .and_then(|m| m.keys().next().cloned())
            .unwrap_or_else(|| "input".to_string());
        *a = json!({ property: input });
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize_response(
    response: &Value,
    model: &Model,
    opts: &CodexStreamOptions,
    output: &mut AssistantMessage,
    saw_tool_call: bool,
    es: &AssistantMessageEventStream,
) -> Result<(), CodexStreamError> {
    if let Some(id) = response.get("id").and_then(|v| v.as_str()) {
        if let Message::Assistant {
            ref mut response_id, ..
        } = output
        {
            *response_id = Some(id.to_string());
        }
    }

    let mut usage = match output {
        Message::Assistant { usage, .. } => usage.clone(),
        _ => Usage::default(),
    };

    if let Some(u) = response.get("usage") {
        let input_tokens = u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        let output_tokens = u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        let total_tokens = u.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        let cached_tokens = u
            .pointer("/input_tokens_details/cached_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cache_write_tokens = u
            .pointer("/input_tokens_details/cache_write_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let reasoning_tokens = u
            .pointer("/output_tokens_details/reasoning_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        // OpenAI includes cached and cache-write tokens in input_tokens.
        usage.input = input_tokens
            .saturating_sub(cached_tokens)
            .saturating_sub(cache_write_tokens);
        usage.output = output_tokens;
        usage.cache_read = cached_tokens;
        usage.cache_write = cache_write_tokens;
        usage.reasoning = reasoning_tokens;
        usage.total_tokens = total_tokens;
    }

    usage.cost = crate::utils::validation::calculate_cost(model, &usage);

    // Service-tier pricing (TS: applyServiceTierPricing).
    let response_tier = response
        .get("service_tier")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let effective_tier =
        resolve_codex_service_tier(response_tier.as_deref(), opts.service_tier.as_deref());
    let multiplier = service_tier_cost_multiplier(&model.id, effective_tier.as_deref());
    if multiplier != 1.0 {
        usage.cost.input *= multiplier;
        usage.cost.output *= multiplier;
        usage.cost.cache_read *= multiplier;
        usage.cost.cache_write *= multiplier;
        usage.cost.total = usage.cost.input
            + usage.cost.output
            + usage.cost.cache_read
            + usage.cost.cache_write;
    }

    if let Message::Assistant { usage: ref mut u, .. } = output {
        *u = usage.clone();
    }
    es.push(AssistantMessageEvent::Usage { usage });

    // Status → stop reason (TS: mapStopReason).
    let status = response.get("status").and_then(|v| v.as_str());
    let incomplete_reason = response
        .pointer("/incomplete_details/reason")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let (stop_reason, error_message) = match status {
        None | Some("completed") => ("stop".to_string(), None),
        Some("incomplete") => {
            if incomplete_reason.as_deref() == Some("max_output_tokens") {
                ("length".to_string(), None)
            } else {
                (
                    "error".to_string(),
                    Some(match &incomplete_reason {
                        Some(r) => format!("Response incomplete: {}", r),
                        None => "Response incomplete without a provider reason".to_string(),
                    }),
                )
            }
        }
        Some("failed") | Some("cancelled") => ("error".to_string(), None),
        Some("in_progress") | Some("queued") => ("stop".to_string(), None),
        Some(other) => {
            return Err(CodexStreamError::other(format!(
                "Unhandled stop reason: {}",
                other
            )))
        }
    };
    let stop_reason = if stop_reason == "stop" && saw_tool_call {
        "toolUse".to_string()
    } else {
        stop_reason
    };

    if let Message::Assistant {
        stop_reason: ref mut sr,
        error_message: ref mut em,
        ..
    } = output
    {
        *sr = Some(stop_reason);
        *em = error_message;
    }
    Ok(())
}

fn resolve_codex_service_tier(
    response_tier: Option<&str>,
    request_tier: Option<&str>,
) -> Option<String> {
    if response_tier == Some("default")
        && (request_tier == Some("flex") || request_tier == Some("priority"))
    {
        return request_tier.map(|s| s.to_string());
    }
    response_tier.or(request_tier).map(|s| s.to_string())
}

fn service_tier_cost_multiplier(model_id: &str, service_tier: Option<&str>) -> f64 {
    match service_tier {
        Some("flex") => 0.5,
        Some("priority") => {
            if model_id == "gpt-5.5" {
                2.5
            } else {
                2.0
            }
        }
        _ => 1.0,
    }
}

// ---------------------------------------------------------------------------
// Simple-options bridging (TS: streamSimple)
// ---------------------------------------------------------------------------

/// Translate a thinking level into a raw effort string; the model's level map
/// is applied later at request-build time (mirrors `clampThinkingLevel`).
pub fn thinking_level_to_effort(model: &Model, level: ModelThinkingLevel) -> Option<String> {
    let raw = match level {
        ModelThinkingLevel::None => return None,
        ModelThinkingLevel::Low => "low",
        ModelThinkingLevel::Medium => "medium",
        ModelThinkingLevel::High => "high",
    };
    Some(
        model
            .thinking_level_map
            .as_ref()
            .and_then(|m| m.get(raw).cloned())
            .unwrap_or_else(|| raw.to_string()),
    )
}

/// Bridge an [`crate::types::ApiStreamOptions`] (auth-merged, with the
/// serialized simple-stream knobs in `extra_body`) into full Codex options.
pub fn codex_options_from_api(
    _model: &Model,
    config: &crate::providers::oauth_compat::OAuthProviderConfig,
    options: &crate::types::ApiStreamOptions,
) -> CodexStreamOptions {
    let ro = &options.request_options;
    let api_key = ro
        .headers
        .get("Authorization")
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|s| s.to_string())
        .or_else(|| ro.headers.get("x-api-key").cloned())
        .unwrap_or_default();

    let mut extra_headers = config.extra_headers.clone().unwrap_or_default();
    for (k, v) in &ro.headers {
        if k != "Authorization" {
            extra_headers.insert(k.clone(), v.clone());
        }
    }

    let mut opts = CodexStreamOptions {
        api_key,
        base_url: ro.base_url.clone().or(Some(config.base_url.clone())),
        extra_headers,
        signal: options.signal.clone(),
        max_retries: options.max_retries,
        max_retry_delay_ms: options.max_retry_delay_ms,
        timeout_ms: options.timeout_ms,
        ..Default::default()
    };

    if let Some(extra) = &ro.extra_body {
        if let Some(sid) = extra.get("session_id").and_then(|v| v.as_str()) {
            opts.session_id = Some(sid.to_string());
        }
        if let Some(cr) = extra.get("cache_retention").and_then(|v| v.as_str()) {
            opts.cache_retention = match cr {
                "short" => CacheRetention::Short,
                "long" => CacheRetention::Long,
                _ => CacheRetention::None,
            };
        }
        if let Some(temp) = extra.get("temperature").and_then(|v| v.as_f64()) {
            opts.temperature = Some(temp);
        }
        if let Some(reasoning) = extra.get("reasoning").and_then(|v| v.as_str()) {
            opts.reasoning_effort = Some(reasoning.to_string());
        }
        if let Some(transport) = extra.get("transport").and_then(|v| v.as_str()) {
            opts.transport = CodexTransport::parse(transport);
        }
        if let Some(base) = extra.get("base_url_override").and_then(|v| v.as_str()) {
            opts.base_url = Some(base.to_string());
        }
        if let Some(tc) = extra.get("tool_choice") {
            match tc.as_str() {
                Some("auto") => opts.tool_choice = Some(ToolChoice::Auto),
                Some("none") => opts.tool_choice = Some(ToolChoice::None),
                Some("required") => opts.tool_choice = Some(ToolChoice::Required),
                _ => {}
            }
        }
    }

    opts
}
