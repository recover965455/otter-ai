//! Codex e2e tests — Rust port of pi-ai's
//! `test/openai-codex-stream.test.ts` (SSE + retry subset) and
//! `test/openai-codex-oauth.test.ts` (refresh subset), driven against a
//! local mock server instead of a stubbed global `fetch`.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use otter_ai::auth::{
    resolve_provider_auth, AuthOperationOptions, AuthResolutionOverrides, Credential,
    CredentialStore, InMemoryCredentialStore, ModifyFnOutput, OAuthAuth,
};
use otter_ai::models::Models;
use otter_ai::providers::oauth_compat::{
    build_oauth_provider, GenericOAuthAuth, OAuthProviderConfig, OAuthProviderSpec,
    OAuthTokenRequestEncoding,
};
use otter_ai::providers::openai_responses::{
    build_request_body, stream_codex, CodexStreamOptions, CodexTransport,
};
use otter_ai::types::{
    AssistantMessageEvent, CacheRetention, ContentBlock, Context, Message, Model, ModelCostRates,
    ModelThinkingLevel, SimpleStreamOptions, Tool, ToolChoice, ToolConstrainedSampling,
    ToolGrammarVariants, Usage,
};

use common::{
    basic_completion_events, MockResponse, MockServer, RecordedRequest, RequestHandler, SseChunk,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn b64(data: &str) -> String {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes = data.as_bytes();
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
}

fn mock_token(account_id: &str) -> String {
    let payload = format!(
        "{{\"https://api.openai.com/auth\":{{\"chatgpt_account_id\":\"{}\"}}}}",
        account_id
    );
    format!("aaa.{}.bbb", b64(&payload))
}

fn codex_model(id: &str) -> Model {
    codex_model_with_map(id, None)
}

fn codex_model_with_map(
    id: &str,
    thinking_level_map: Option<std::collections::HashMap<String, String>>,
) -> Model {
    Model {
        id: id.to_string(),
        provider_id: "chatgpt-plus".to_string(),
        name: id.to_string(),
        api: "openai-codex-responses".to_string(),
        max_input_tokens: Some(400_000),
        max_output_tokens: Some(128_000),
        supports_images: true,
        supports_audio: false,
        supports_video: false,
        supports_pdf: false,
        supports_tool_calling: true,
        supports_structured_output: true,
        supports_system_prompt: true,
        thinking: ModelThinkingLevel::High,
        reasoning: true,
        cost_rates: ModelCostRates::default(),
        context_window: Some(400_000),
        default_temperature: Some(1.0),
        thinking_level_map,
    }
}

fn level_map(entries: &[(&str, &str)]) -> std::collections::HashMap<String, String> {
    entries
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn user_context(text: &str) -> Context {
    Context {
        system_prompt: Some("You are a helpful assistant.".to_string()),
        messages: vec![Message::user_from_string(text)],
        ..Default::default()
    }
}

async fn collect_events(
    stream: otter_ai::AssistantMessageEventStream,
) -> (Vec<AssistantMessageEvent>, Message) {
    let result = stream.result_future();
    let mut events = Vec::new();
    let mut s = stream;
    while let Some(evt) = s.next().await {
        events.push(evt);
    }
    let msg = result.await;
    (events, msg)
}

fn event_types(events: &[AssistantMessageEvent]) -> Vec<&'static str> {
    events
        .iter()
        .map(|e| match e {
            AssistantMessageEvent::Start { .. } => "start",
            AssistantMessageEvent::TextStart => "text_start",
            AssistantMessageEvent::TextDelta { .. } => "text_delta",
            AssistantMessageEvent::TextEnd => "text_end",
            AssistantMessageEvent::ThinkingStart => "thinking_start",
            AssistantMessageEvent::ThinkingDelta { .. } => "thinking_delta",
            AssistantMessageEvent::ThinkingEnd { .. } => "thinking_end",
            AssistantMessageEvent::ToolcallStart { .. } => "toolcall_start",
            AssistantMessageEvent::ToolcallDelta { .. } => "toolcall_delta",
            AssistantMessageEvent::ToolcallEnd { .. } => "toolcall_end",
            AssistantMessageEvent::Usage { .. } => "usage",
            AssistantMessageEvent::Done { .. } => "done",
            AssistantMessageEvent::Error { .. } => "error",
        })
        .collect()
}

fn text_of(msg: &Message) -> String {
    otter_ai::content_text(match msg {
        Message::Assistant { content, .. } => content,
        _ => &[],
    })
}

fn stop_reason_of(msg: &Message) -> String {
    msg.stop_reason().unwrap_or_default().to_string()
}

fn usage_of(msg: &Message) -> Usage {
    match msg {
        Message::Assistant { usage, .. } => usage.clone(),
        _ => Usage::default(),
    }
}

fn static_handler(resp: MockResponse) -> RequestHandler {
    Arc::new(move |_| resp.clone())
}

// ---------------------------------------------------------------------------
// SSE streaming (ports of openai-codex-stream.test.ts)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn streams_sse_responses_into_event_stream_with_wire_headers() {
    let token = mock_token("acc_test");
    let server = MockServer::spawn(static_handler(MockResponse::sse(basic_completion_events(
        "completed",
        None,
    ))))
    .await;

    let opts = CodexStreamOptions {
        api_key: token.clone(),
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        ..Default::default()
    };
    let (events, msg) = collect_events(stream_codex(
        &codex_model("gpt-5.4"),
        user_context("Say hello"),
        opts,
    ))
    .await;

    let types = event_types(&events);
    assert!(types.contains(&"text_delta"));
    assert!(types.contains(&"done"));
    assert_eq!(text_of(&msg), "Hello");
    assert_eq!(stop_reason_of(&msg), "stop");

    let req = &server.recorded()[0];
    assert_eq!(req.method, "POST");
    assert_eq!(req.path, "/codex/responses");
    assert_eq!(
        req.header("authorization"),
        Some(format!("Bearer {}", token).as_str())
    );
    assert_eq!(req.header("chatgpt-account-id"), Some("acc_test"));
    assert_eq!(req.header("openai-beta"), Some("responses=experimental"));
    assert_eq!(req.header("accept"), Some("text/event-stream"));
    assert_eq!(req.header("x-api-key"), None);

    server.shutdown();
}

#[tokio::test]
async fn completes_after_response_completed_even_when_body_stays_open() {
    let token = mock_token("acc_test");
    let mut events = basic_completion_events("completed", Some(false));
    events.push("data: [DONE]".to_string());
    let payload = events
        .iter()
        .map(|e| format!("{}\n\n", e))
        .collect::<Vec<_>>()
        .join("");
    let server = MockServer::spawn(Arc::new(move |_| {
        MockResponse::sse_stream(
            vec![SseChunk::now(payload.clone())],
            true, // body never closes
        )
    }))
    .await;

    let opts = CodexStreamOptions {
        api_key: token,
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        ..Default::default()
    };
    let (events, msg) = tokio::time::timeout(
        Duration::from_secs(5),
        collect_events(stream_codex(
            &codex_model("gpt-5.4"),
            user_context("Say hello"),
            opts,
        )),
    )
    .await
    .expect("completes without waiting for body close");

    assert_eq!(text_of(&msg), "Hello");
    assert_eq!(stop_reason_of(&msg), "stop");
    match &msg {
        Message::Assistant { end_turn, .. } => assert_eq!(end_turn, &Some(false)),
        _ => panic!("expected assistant message"),
    }
    assert!(event_types(&events).contains(&"usage"));
    server.shutdown();
}

#[tokio::test]
async fn maps_response_incomplete_to_stop_reason_length() {
    let token = mock_token("acc_test");
    let server = MockServer::spawn(static_handler(MockResponse::sse(basic_completion_events(
        "incomplete",
        None,
    ))))
    .await;

    let opts = CodexStreamOptions {
        api_key: token,
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        ..Default::default()
    };
    let (_, msg) = collect_events(stream_codex(
        &codex_model("gpt-5.4"),
        user_context("Say hello"),
        opts,
    ))
    .await;

    assert_eq!(text_of(&msg), "Hello");
    assert_eq!(stop_reason_of(&msg), "length");
    server.shutdown();
}

#[tokio::test]
async fn aborts_sse_fetch_after_the_configured_http_timeout() {
    let token = mock_token("acc_test");
    let server = MockServer::spawn(static_handler(MockResponse::Silent)).await;

    let opts = CodexStreamOptions {
        api_key: token,
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        timeout_ms: Some(50),
        ..Default::default()
    };
    let (_, msg) = collect_events(stream_codex(
        &codex_model("gpt-5.4"),
        user_context("Say hello"),
        opts,
    ))
    .await;

    assert_eq!(stop_reason_of(&msg), "error");
    match &msg {
        Message::Assistant {
            error_message: Some(e),
            ..
        } => assert_eq!(e, "Codex SSE response headers timed out after 50ms"),
        _ => panic!("expected error message"),
    }
    assert_eq!(server.request_count(), 1);
    server.shutdown();
}

#[tokio::test]
async fn aborts_sse_body_reads_after_response_headers_arrive() {
    let token = mock_token("acc_test");
    let delta_one = common::sse_line(serde_json::json!({
        "type": "response.output_item.added",
        "item": { "type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": [] },
    })) + "\n\n"
        + &common::sse_line(serde_json::json!({
            "type": "response.output_text.delta",
            "delta": "one",
        }))
        + "\n\n";
    let delta_two_and_terminal = common::sse_line(serde_json::json!({
        "type": "response.output_text.delta",
        "delta": "two",
    })) + "\n\n"
        + &common::sse_line(serde_json::json!({
            "type": "response.completed",
            "response": { "status": "completed" },
        }))
        + "\n\n";

    let server = MockServer::spawn(Arc::new(move |_| {
        MockResponse::sse_stream(
            vec![
                SseChunk::now(delta_one.clone()),
                SseChunk::after(delta_two_and_terminal.clone(), Duration::from_millis(150)),
            ],
            false,
        )
    }))
    .await;

    let signal = otter_ai::CancellationToken::new();
    let opts = CodexStreamOptions {
        api_key: token,
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        signal: Some(signal.clone()),
        ..Default::default()
    };

    let stream = stream_codex(&codex_model("gpt-5.4"), user_context("Say hello"), opts);
    let result = stream.result_future();
    let mut s = stream;
    let mut saw_one = false;
    while let Some(evt) = s.next().await {
        if let AssistantMessageEvent::TextDelta { delta } = evt {
            if delta == "one" {
                saw_one = true;
                signal.cancel();
            }
            assert_ne!(
                delta, "two",
                "cancelled stream must not deliver later deltas"
            );
        }
    }
    let msg = result.await;
    assert!(saw_one);
    assert_eq!(stop_reason_of(&msg), "aborted");
    match &msg {
        Message::Assistant {
            error_message: Some(e),
            ..
        } => assert_eq!(e, "Request was aborted"),
        _ => panic!("expected error message"),
    }
    server.shutdown();
}

#[tokio::test]
async fn sets_session_headers_and_prompt_cache_key_when_session_id_provided() {
    let token = mock_token("acc_test");
    let captured: Arc<Mutex<Vec<RecordedRequest>>> = Arc::new(Mutex::new(Vec::new()));
    let cap = captured.clone();
    let server = MockServer::spawn(Arc::new(move |req| {
        cap.lock().unwrap().push(req.clone());
        MockResponse::sse(basic_completion_events("completed", None))
    }))
    .await;

    let opts = CodexStreamOptions {
        api_key: token,
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        session_id: Some("test-session-123".to_string()),
        cache_retention: CacheRetention::Short,
        ..Default::default()
    };
    let (_, msg) = collect_events(stream_codex(
        &codex_model("gpt-5.4"),
        user_context("Say hello"),
        opts,
    ))
    .await;
    assert_eq!(stop_reason_of(&msg), "stop");

    let reqs = captured.lock().unwrap().clone();
    let req = &reqs[0];
    assert_eq!(req.header("session-id"), Some("test-session-123"));
    assert_eq!(req.header("x-client-request-id"), Some("test-session-123"));
    assert_eq!(req.json_body()["prompt_cache_key"], "test-session-123");
    server.shutdown();
}

#[tokio::test]
async fn omits_sse_cache_affinity_when_cache_retention_is_none() {
    let token = mock_token("acc_test");
    let server = MockServer::spawn(static_handler(MockResponse::sse(basic_completion_events(
        "completed",
        None,
    ))))
    .await;

    let opts = CodexStreamOptions {
        api_key: token,
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        session_id: Some("one-off-summary".to_string()),
        cache_retention: CacheRetention::None,
        ..Default::default()
    };
    let (_, msg) = collect_events(stream_codex(
        &codex_model("gpt-5.4"),
        user_context("Say hello"),
        opts,
    ))
    .await;
    assert_eq!(stop_reason_of(&msg), "stop");

    let req = &server.recorded()[0];
    assert_eq!(req.header("session-id"), None);
    assert_eq!(req.header("x-client-request-id"), None);
    assert!(req.json_body().get("prompt_cache_key").is_none());
    server.shutdown();
}

#[tokio::test]
async fn clamps_prompt_cache_key_and_session_headers_to_64_chars() {
    let token = mock_token("acc_test");
    let server = MockServer::spawn(static_handler(MockResponse::sse(basic_completion_events(
        "completed",
        None,
    ))))
    .await;

    let long_session = "x".repeat(67);
    let opts = CodexStreamOptions {
        api_key: token,
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        session_id: Some(long_session.clone()),
        cache_retention: CacheRetention::Short,
        ..Default::default()
    };
    let (_, msg) = collect_events(stream_codex(
        &codex_model("gpt-5.4"),
        user_context("Say hello"),
        opts,
    ))
    .await;
    assert_eq!(stop_reason_of(&msg), "stop");

    let clamped = "x".repeat(64);
    let req = &server.recorded()[0];
    assert_eq!(req.header("session-id"), Some(clamped.as_str()));
    assert_eq!(req.header("x-client-request-id"), Some(clamped.as_str()));
    assert_eq!(req.json_body()["prompt_cache_key"], clamped);
    server.shutdown();
}

#[tokio::test]
async fn does_not_set_session_headers_when_session_id_missing() {
    let token = mock_token("acc_test");
    let server = MockServer::spawn(static_handler(MockResponse::sse(basic_completion_events(
        "completed",
        None,
    ))))
    .await;

    let opts = CodexStreamOptions {
        api_key: token,
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        ..Default::default()
    };
    let (_, msg) = collect_events(stream_codex(
        &codex_model("gpt-5.4"),
        user_context("Say hello"),
        opts,
    ))
    .await;
    assert_eq!(stop_reason_of(&msg), "stop");

    let req = &server.recorded()[0];
    assert_eq!(req.header("session-id"), None);
    assert_eq!(req.header("x-client-request-id"), None);
    assert!(req.json_body().get("prompt_cache_key").is_none());
    server.shutdown();
}

// ---------------------------------------------------------------------------
// Request body shape
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sends_the_expected_codex_request_body_shape() {
    let token = mock_token("acc_test");
    let server = MockServer::spawn(static_handler(MockResponse::sse(basic_completion_events(
        "completed",
        None,
    ))))
    .await;

    let mut context = user_context("Use a tool");
    context.tools = vec![Tool {
        name: "ping".to_string(),
        description: Some("Ping".to_string()),
        parameters: serde_json::json!({ "type": "object", "properties": {} }),
        constrained_sampling: None,
    }];

    let opts = CodexStreamOptions {
        api_key: token,
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        ..Default::default()
    };
    let (_, msg) = collect_events(stream_codex(&codex_model("gpt-5.4"), context, opts)).await;
    assert_eq!(stop_reason_of(&msg), "stop");

    let body = server.recorded()[0].json_body();
    assert_eq!(body["model"], "gpt-5.4");
    assert_eq!(body["store"], false);
    assert_eq!(body["stream"], true);
    assert_eq!(body["instructions"], "You are a helpful assistant.");
    assert_eq!(
        body["include"],
        serde_json::json!(["reasoning.encrypted_content"])
    );
    assert_eq!(body["parallel_tool_calls"], true);
    assert_eq!(body["tool_choice"], "auto");
    assert_eq!(body["text"]["verbosity"], "low");
    // Codex rejects max_output_tokens with 400 "Unsupported parameter".
    assert!(body.get("max_output_tokens").is_none());
    assert_eq!(body["tools"][0]["type"], "function");
    assert_eq!(body["tools"][0]["name"], "ping");
    assert_eq!(body["tools"][0]["strict"], serde_json::Value::Null);
    assert_eq!(
        body["input"],
        serde_json::json!([
            { "role": "user", "content": [{ "type": "input_text", "text": "Use a tool" }] }
        ])
    );
    server.shutdown();
}

#[test]
fn build_request_body_supports_system_prompt_override_and_custom_instructions() {
    let model = codex_model("gpt-5.4");
    let mut ctx = user_context("hi");
    ctx.system_prompt = Some("Custom system prompt.".to_string());
    let opts = CodexStreamOptions::default();
    let body = build_request_body(&model, &ctx, &opts, None).unwrap();
    assert_eq!(body["instructions"], "Custom system prompt.");
}

#[tokio::test]
async fn preserves_xhigh_reasoning_effort_through_the_level_map() {
    let token = mock_token("acc_test");
    let server = MockServer::spawn(static_handler(MockResponse::sse(basic_completion_events(
        "completed",
        None,
    ))))
    .await;

    let model = codex_model_with_map(
        "gpt-5.5",
        Some(level_map(&[("xhigh", "xhigh"), ("minimal", "low")])),
    );
    let opts = CodexStreamOptions {
        api_key: token,
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        reasoning_effort: Some("xhigh".to_string()),
        ..Default::default()
    };
    let (_, msg) = collect_events(stream_codex(&model, user_context("Say hello"), opts)).await;
    assert_eq!(stop_reason_of(&msg), "stop");

    assert_eq!(
        server.recorded()[0].json_body()["reasoning"],
        serde_json::json!({ "effort": "xhigh", "summary": "auto" })
    );
    server.shutdown();
}

#[tokio::test]
async fn clamps_minimal_reasoning_effort_to_low() {
    let token = mock_token("acc_test");
    let server = MockServer::spawn(static_handler(MockResponse::sse(basic_completion_events(
        "completed",
        None,
    ))))
    .await;

    for model_id in ["gpt-5.3-codex-spark", "gpt-5.4", "gpt-5.5"] {
        let model = codex_model_with_map(model_id, Some(level_map(&[("minimal", "low")])));
        let opts = CodexStreamOptions {
            api_key: token.clone(),
            base_url: Some(server.url()),
            transport: CodexTransport::Sse,
            reasoning_effort: Some("minimal".to_string()),
            ..Default::default()
        };
        let (_, msg) = collect_events(stream_codex(&model, user_context("Say hello"), opts)).await;
        assert_eq!(stop_reason_of(&msg), "stop");
        assert_eq!(
            server.recorded().last().unwrap().json_body()["reasoning"],
            serde_json::json!({ "effort": "low", "summary": "auto" })
        );
    }
    server.shutdown();
}

#[tokio::test]
async fn forwards_required_tool_choice() {
    let token = mock_token("acc_test");
    let server = MockServer::spawn(static_handler(MockResponse::sse(basic_completion_events(
        "completed",
        None,
    ))))
    .await;

    let opts = CodexStreamOptions {
        api_key: token,
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        tool_choice: Some(ToolChoice::Required),
        ..Default::default()
    };
    let mut ctx = user_context("Do not call ping. Respond with text instead.");
    ctx.tools = vec![Tool {
        name: "ping".to_string(),
        description: Some("Ping".to_string()),
        parameters: serde_json::json!({ "type": "object", "properties": {} }),
        constrained_sampling: None,
    }];
    let (_, msg) = collect_events(stream_codex(&codex_model("gpt-5.5"), ctx, opts)).await;
    assert_eq!(stop_reason_of(&msg), "stop");
    assert_eq!(server.recorded()[0].json_body()["tool_choice"], "required");
    server.shutdown();
}

// ---------------------------------------------------------------------------
// Service tier pricing (ports of the it.each service tier tests)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn applies_service_tier_cost_multipliers_when_backend_echoes_default() {
    let cases: &[(&str, &str, f64, u64, u64)] = &[
        ("gpt-5.1-codex", "flex", 0.5, 1_000_000, 1_000_000),
        ("gpt-5.1-codex", "priority", 2.0, 1_000_000, 1_000_000),
        ("gpt-5.5", "flex", 0.5, 1_000_000, 1_000_000),
        ("gpt-5.5", "priority", 2.5, 1_000_000, 1_000_000),
    ];
    for (model_id, tier, multiplier, input_tokens, output_tokens) in cases {
        let token = mock_token("acc_test");
        let server = MockServer::spawn(static_handler(MockResponse::sse(
            common::usage_events_with_tier(*input_tokens, *output_tokens, Some("default")),
        )))
        .await;

        let mut model = codex_model(model_id);
        model.cost_rates = ModelCostRates {
            input_per_million: Some(1.0),
            output_per_million: Some(2.0),
            ..Default::default()
        };
        let opts = CodexStreamOptions {
            api_key: token,
            base_url: Some(server.url()),
            transport: CodexTransport::Sse,
            service_tier: Some(tier.to_string()),
            ..Default::default()
        };
        let (_, msg) = collect_events(stream_codex(&model, user_context("Say hello"), opts)).await;

        let usage = usage_of(&msg);
        let expected_input = 1.0 * *input_tokens as f64 / 1_000_000.0 * multiplier;
        let expected_output = 2.0 * *output_tokens as f64 / 1_000_000.0 * multiplier;
        assert!(
            (usage.cost.input - expected_input).abs() < 1e-6,
            "{} {} input cost {} != {}",
            model_id,
            tier,
            usage.cost.input,
            expected_input
        );
        assert!(
            (usage.cost.output - expected_output).abs() < 1e-6,
            "{} {} output cost mismatch",
            model_id,
            tier
        );
        assert!((usage.cost.total - (expected_input + expected_output)).abs() < 1e-6);
        server.shutdown();
    }
}

// ---------------------------------------------------------------------------
// Retry behaviour
// ---------------------------------------------------------------------------

fn rate_limited_response(extra_headers: Vec<(String, String)>) -> MockResponse {
    MockResponse::json_with_headers(
        429,
        serde_json::json!({ "error": { "code": "rate_limit_exceeded", "message": "rate limited" } }),
        extra_headers,
    )
}

#[tokio::test]
async fn uses_retry_after_ms_for_sse_retries() {
    let token = mock_token("acc_test");
    let counter = Arc::new(Mutex::new(0u32));
    let cnt = counter.clone();
    let server = MockServer::spawn(Arc::new(move |_| {
        let mut n = cnt.lock().unwrap();
        *n += 1;
        if *n == 1 {
            rate_limited_response(vec![("retry-after-ms".to_string(), "50".to_string())])
        } else {
            MockResponse::sse(basic_completion_events("completed", None))
        }
    }))
    .await;

    let opts = CodexStreamOptions {
        api_key: token,
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        max_retries: Some(1),
        ..Default::default()
    };
    let (_, msg) = tokio::time::timeout(
        Duration::from_secs(5),
        collect_events(stream_codex(
            &codex_model("gpt-5.4"),
            user_context("Say hello"),
            opts,
        )),
    )
    .await
    .expect("retry succeeds");
    assert_eq!(text_of(&msg), "Hello");
    assert_eq!(server.request_count(), 2);
    server.shutdown();
}

#[tokio::test]
async fn uses_retry_after_seconds_for_sse_retries() {
    let token = mock_token("acc_test");
    let counter = Arc::new(Mutex::new(0u32));
    let cnt = counter.clone();
    let server = MockServer::spawn(Arc::new(move |_| {
        let mut n = cnt.lock().unwrap();
        *n += 1;
        if *n == 1 {
            rate_limited_response(vec![("retry-after".to_string(), "1".to_string())])
        } else {
            MockResponse::sse(basic_completion_events("completed", None))
        }
    }))
    .await;

    let opts = CodexStreamOptions {
        api_key: token,
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        max_retries: Some(1),
        ..Default::default()
    };
    let (_, msg) = tokio::time::timeout(
        Duration::from_secs(5),
        collect_events(stream_codex(
            &codex_model("gpt-5.4"),
            user_context("Say hello"),
            opts,
        )),
    )
    .await
    .expect("retry succeeds");
    assert_eq!(text_of(&msg), "Hello");
    assert_eq!(server.request_count(), 2);
    server.shutdown();
}

#[tokio::test]
async fn fails_immediately_when_retry_delay_exceeds_the_limit() {
    for status in [429u16, 503] {
        let token = mock_token("acc_test");
        let server = MockServer::spawn(static_handler(MockResponse::json_with_headers(
            status,
            serde_json::json!({ "error": { "code": "temporarily_unavailable", "message": "retry later" } }),
            vec![("retry-after".to_string(), "2".to_string())],
        )))
        .await;

        let opts = CodexStreamOptions {
            api_key: token,
            base_url: Some(server.url()),
            transport: CodexTransport::Sse,
            max_retries: Some(3),
            max_retry_delay_ms: Some(1000),
            ..Default::default()
        };
        let (_, msg) = collect_events(stream_codex(
            &codex_model("gpt-5.4"),
            user_context("Say hello"),
            opts,
        ))
        .await;
        assert_eq!(stop_reason_of(&msg), "error", "status {}", status);
        match &msg {
            Message::Assistant {
                error_message: Some(e),
                ..
            } => assert_eq!(e, "Server requested 2s retry delay (max: 1s)"),
            _ => panic!("expected error message"),
        }
        assert_eq!(server.request_count(), 1, "no retry after over-limit delay");
        server.shutdown();
    }
}

#[tokio::test]
async fn uses_exponential_backoff_across_repeated_retries() {
    let token = mock_token("acc_test");
    let counter = Arc::new(Mutex::new(0u32));
    let cnt = counter.clone();
    let server = MockServer::spawn(Arc::new(move |_| {
        let mut n = cnt.lock().unwrap();
        *n += 1;
        if *n <= 2 {
            rate_limited_response(vec![])
        } else {
            MockResponse::sse(basic_completion_events("completed", None))
        }
    }))
    .await;

    let opts = CodexStreamOptions {
        api_key: token,
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        max_retries: Some(2),
        ..Default::default()
    };
    // Backoff delays: 1s + 2s = 3s total.
    let (_, msg) = tokio::time::timeout(
        Duration::from_secs(15),
        collect_events(stream_codex(
            &codex_model("gpt-5.4"),
            user_context("Say hello"),
            opts,
        )),
    )
    .await
    .expect("eventual success after backoff");
    assert_eq!(text_of(&msg), "Hello");
    assert_eq!(server.request_count(), 3);
    server.shutdown();
}

#[tokio::test]
async fn does_not_retry_terminal_usage_limit_errors() {
    let token = mock_token("acc_test");
    let server = MockServer::spawn(static_handler(MockResponse::json(
        429,
        serde_json::json!({
            "error": {
                "code": "insufficient_quota",
                "message": "You have hit your ChatGPT usage limit (plus plan).",
                "plan_type": "plus",
            }
        }),
    )))
    .await;

    let opts = CodexStreamOptions {
        api_key: token,
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        max_retries: Some(3),
        ..Default::default()
    };
    let (_, msg) = collect_events(stream_codex(
        &codex_model("gpt-5.4"),
        user_context("Say hello"),
        opts,
    ))
    .await;
    assert_eq!(stop_reason_of(&msg), "error");
    match &msg {
        Message::Assistant {
            error_message: Some(e),
            ..
        } => assert!(
            e.contains("You have hit your ChatGPT usage limit"),
            "unexpected message: {}",
            e
        ),
        _ => panic!("expected error message"),
    }
    assert_eq!(
        server.request_count(),
        1,
        "terminal rate limits are not retried"
    );
    server.shutdown();
}

#[tokio::test]
async fn surfaces_unauthorized_error_bodies() {
    let token = mock_token("acc_test");
    let server = MockServer::spawn(static_handler(MockResponse::json(
        401,
        serde_json::json!({
            "error": { "message": "Could not validate your token. Please try signing in again." }
        }),
    )))
    .await;

    let opts = CodexStreamOptions {
        api_key: token,
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        ..Default::default()
    };
    let (_, msg) = collect_events(stream_codex(
        &codex_model("gpt-5.4"),
        user_context("Say hello"),
        opts,
    ))
    .await;
    assert_eq!(stop_reason_of(&msg), "error");
    match &msg {
        Message::Assistant {
            error_message: Some(e),
            ..
        } => assert!(e.contains("Could not validate your token")),
        _ => panic!("expected error message"),
    }
    server.shutdown();
}

#[tokio::test]
async fn maps_stream_error_events_to_errors() {
    let token = mock_token("acc_test");
    let events = vec![common::sse_line(serde_json::json!({
        "type": "error",
        "code": "server_error",
        "message": "boom",
    }))];
    let server = MockServer::spawn(static_handler(MockResponse::sse(events))).await;

    let opts = CodexStreamOptions {
        api_key: token,
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        ..Default::default()
    };
    let (_, msg) = collect_events(stream_codex(
        &codex_model("gpt-5.4"),
        user_context("Say hello"),
        opts,
    ))
    .await;
    assert_eq!(stop_reason_of(&msg), "error");
    match &msg {
        Message::Assistant {
            error_message: Some(e),
            ..
        } => assert!(e.contains("boom"), "message was: {}", e),
        _ => panic!("expected error message"),
    }
    server.shutdown();
}

#[tokio::test]
async fn maps_response_failed_events_to_errors() {
    let token = mock_token("acc_test");
    let events = vec![common::sse_line(serde_json::json!({
        "type": "response.failed",
        "response": { "status": "failed", "error": { "code": "model_error", "message": "upstream exploded" } },
    }))];
    let server = MockServer::spawn(static_handler(MockResponse::sse(events))).await;

    let opts = CodexStreamOptions {
        api_key: token,
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        ..Default::default()
    };
    let (_, msg) = collect_events(stream_codex(
        &codex_model("gpt-5.4"),
        user_context("Say hello"),
        opts,
    ))
    .await;
    assert_eq!(stop_reason_of(&msg), "error");
    match &msg {
        Message::Assistant {
            error_message: Some(e),
            ..
        } => assert_eq!(e, "upstream exploded"),
        _ => panic!("expected error message"),
    }
    server.shutdown();
}

// ---------------------------------------------------------------------------
// Rich streaming: tool calls, thinking, usage mapping, multi-turn replay
// ---------------------------------------------------------------------------

#[tokio::test]
async fn streams_tool_calls_and_maps_tool_use_stop_reason() {
    let token = mock_token("acc_test");
    let events = vec![
        common::sse_line(serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": { "type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather", "arguments": "" },
        })),
        common::sse_line(serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": "{\"ci",
        })),
        common::sse_line(serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": "ty\":\"Paris\"}",
        })),
        common::sse_line(serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": { "type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" },
        })),
        common::sse_line(serde_json::json!({
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": { "input_tokens": 5, "output_tokens": 3, "total_tokens": 8 },
            },
        })),
    ];
    let server = MockServer::spawn(static_handler(MockResponse::sse(events))).await;

    let opts = CodexStreamOptions {
        api_key: token,
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        ..Default::default()
    };
    let (events, msg) = collect_events(stream_codex(
        &codex_model("gpt-5.4"),
        user_context("Weather in Paris?"),
        opts,
    ))
    .await;

    let types = event_types(&events);
    assert!(types.contains(&"toolcall_start"));
    assert_eq!(types.iter().filter(|t| **t == "toolcall_delta").count(), 2);
    assert!(types.contains(&"toolcall_end"));
    assert_eq!(stop_reason_of(&msg), "toolUse");

    match &msg {
        Message::Assistant { content, .. } => match &content[0] {
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => {
                assert_eq!(id, "call_1|fc_1");
                assert_eq!(name, "get_weather");
                assert_eq!(arguments, &serde_json::json!({ "city": "Paris" }));
            }
            other => panic!("expected tool call, got {:?}", other),
        },
        _ => panic!("expected assistant message"),
    }
    server.shutdown();
}

#[tokio::test]
async fn streams_thinking_with_encrypted_signature_for_replay() {
    let token = mock_token("acc_test");
    let events = vec![
        common::sse_line(serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": { "type": "reasoning", "id": "rs_1" },
        })),
        common::sse_line(serde_json::json!({
            "type": "response.reasoning_summary_text.delta",
            "output_index": 0,
            "delta": "thinking hard",
        })),
        common::sse_line(serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "reasoning",
                "id": "rs_1",
                "encrypted_content": "ENC",
                "summary": [{ "type": "summary_text", "text": "thinking hard" }],
            },
        })),
        common::sse_line(serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 1,
            "item": { "type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": [] },
        })),
        common::sse_line(serde_json::json!({
            "type": "response.output_text.delta",
            "output_index": 1,
            "delta": "Answer",
        })),
        common::sse_line(serde_json::json!({
            "type": "response.completed",
            "response": { "status": "completed", "usage": { "input_tokens": 5, "output_tokens": 3, "total_tokens": 8 } },
        })),
    ];
    let server = MockServer::spawn(static_handler(MockResponse::sse(events))).await;

    let opts = CodexStreamOptions {
        api_key: token.clone(),
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        ..Default::default()
    };
    let context = user_context("Think then answer");
    let (events, msg) =
        collect_events(stream_codex(&codex_model("gpt-5.4"), context.clone(), opts)).await;

    let types = event_types(&events);
    assert!(types.contains(&"thinking_start"));
    assert!(types.contains(&"thinking_delta"));
    assert!(types.contains(&"thinking_end"));

    match &msg {
        Message::Assistant { content, .. } => {
            match &content[0] {
                ContentBlock::Thinking {
                    thinking,
                    signature,
                } => {
                    assert_eq!(thinking, "thinking hard");
                    let sig = signature.as_deref().expect("signature present");
                    assert!(sig.contains("ENC"), "signature carries encrypted_content");
                    // The signature must round-trip as the replayed item.
                    let item: serde_json::Value = serde_json::from_str(sig).unwrap();
                    assert_eq!(item["type"], "reasoning");
                    assert_eq!(item["encrypted_content"], "ENC");
                }
                other => panic!("expected thinking block, got {:?}", other),
            }
            match &content[1] {
                ContentBlock::Text { text, .. } => assert_eq!(text, "Answer"),
                other => panic!("expected text block, got {:?}", other),
            }
        }
        _ => panic!("expected assistant message"),
    }

    // Multi-turn replay: the thinking signature must be sent as a reasoning
    // item, assistant text as a message item.
    let server2 = MockServer::spawn(static_handler(MockResponse::sse(basic_completion_events(
        "completed",
        None,
    ))))
    .await;
    let mut ctx2 = context;
    ctx2.messages.push(msg);
    ctx2.messages.push(Message::user_from_string("continue"));
    let opts2 = CodexStreamOptions {
        api_key: token,
        base_url: Some(server2.url()),
        transport: CodexTransport::Sse,
        ..Default::default()
    };
    let (_, msg2) = collect_events(stream_codex(&codex_model("gpt-5.4"), ctx2, opts2)).await;
    assert_eq!(stop_reason_of(&msg2), "stop");
    let input = server2.recorded()[0].json_body()["input"]
        .as_array()
        .unwrap()
        .clone();
    // [user, reasoning(replayed signature), assistant message, user]
    assert_eq!(input[1]["type"], "reasoning");
    assert_eq!(input[1]["encrypted_content"], "ENC");
    assert_eq!(input[2]["type"], "message");
    assert_eq!(input[2]["content"][0]["text"], "Answer");
    server2.shutdown();
    server.shutdown();
}

#[tokio::test]
async fn replays_multi_turn_history_with_tool_calls_and_results() {
    let token = mock_token("acc_test");
    let server = MockServer::spawn(static_handler(MockResponse::sse(basic_completion_events(
        "completed",
        None,
    ))))
    .await;

    let assistant =
        Message::assistant_default("openai-codex-responses".into(), "chatgpt-plus".into())
            .with_content(vec![
                ContentBlock::Text {
                    text: "Let me check.".into(),
                    text_signature: None,
                },
                ContentBlock::ToolCall {
                    id: "call_1|fc_1".into(),
                    name: "sample_tool".into(),
                    arguments: serde_json::json!({ "payload": "x" }),
                },
            ])
            .with_model(Some("gpt-5.4".into()))
            .with_stop_reason(Some("toolUse".into()));
    let tool_result = Message::ToolResult {
        tool_call_id: "call_1|fc_1".into(),
        tool_name: "sample_tool".into(),
        content: vec![ContentBlock::Text {
            text: "real result".into(),
            text_signature: None,
        }],
        is_error: false,
        timestamp: 2,
    };
    let mut context = user_context("Use the tool");
    context.messages.push(assistant);
    context.messages.push(tool_result);
    context
        .messages
        .push(Message::user_from_string("Now finish"));

    let opts = CodexStreamOptions {
        api_key: token,
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        ..Default::default()
    };
    let (_, msg) = collect_events(stream_codex(&codex_model("gpt-5.4"), context, opts)).await;
    assert_eq!(stop_reason_of(&msg), "stop");

    let input = server.recorded()[0].json_body()["input"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(
        input[0],
        serde_json::json!({ "role": "user", "content": [{ "type": "input_text", "text": "Use the tool" }] })
    );
    assert_eq!(input[1]["type"], "message");
    assert_eq!(input[1]["role"], "assistant");
    assert_eq!(input[1]["content"][0]["text"], "Let me check.");
    assert_eq!(input[2]["type"], "function_call");
    assert_eq!(input[2]["call_id"], "call_1");
    assert_eq!(input[2]["id"], "fc_1");
    assert_eq!(input[2]["name"], "sample_tool");
    assert_eq!(
        input[2]["arguments"],
        serde_json::json!("{\"payload\":\"x\"}")
    );
    assert_eq!(
        input[3],
        serde_json::json!({ "type": "function_call_output", "call_id": "call_1", "output": "real result" })
    );
    assert_eq!(
        input[4],
        serde_json::json!({ "role": "user", "content": [{ "type": "input_text", "text": "Now finish" }] })
    );
    server.shutdown();
}

#[tokio::test]
async fn maps_usage_details_into_cache_and_reasoning_buckets() {
    let token = mock_token("acc_test");
    let events = vec![common::sse_line(serde_json::json!({
        "type": "response.completed",
        "response": {
            "status": "completed",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 160,
                "input_tokens_details": { "cached_tokens": 40, "cache_write_tokens": 10 },
                "output_tokens_details": { "reasoning_tokens": 7 },
            },
        },
    }))];
    let server = MockServer::spawn(static_handler(MockResponse::sse(events))).await;

    let mut model = codex_model("gpt-5.4");
    model.cost_rates = ModelCostRates {
        input_per_million: Some(1.0),
        output_per_million: Some(2.0),
        input_cache_read_per_million: Some(0.1),
        input_cache_write_per_million: Some(1.25),
        ..Default::default()
    };

    let opts = CodexStreamOptions {
        api_key: token,
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        ..Default::default()
    };
    let (_, msg) = collect_events(stream_codex(&model, user_context("hi"), opts)).await;

    let usage = usage_of(&msg);
    // input = 100 - 40 (cached) - 10 (cache write) = 50
    assert_eq!(usage.input, 50);
    assert_eq!(usage.output, 50);
    assert_eq!(usage.cache_read, 40);
    assert_eq!(usage.cache_write, 10);
    assert_eq!(usage.reasoning, 7);
    assert_eq!(usage.total_tokens, 160);

    let expected_input_cost = 50_f64 / 1_000_000.0 * 1.0;
    let expected_output_cost = 50_f64 / 1_000_000.0 * 2.0;
    let expected_cache_read = 40_f64 / 1_000_000.0 * 0.1;
    let expected_cache_write = 10_f64 / 1_000_000.0 * 1.25;
    assert!((usage.cost.input - expected_input_cost).abs() < 1e-9);
    assert!((usage.cost.output - expected_output_cost).abs() < 1e-9);
    assert!((usage.cost.cache_read - expected_cache_read).abs() < 1e-9);
    assert!((usage.cost.cache_write - expected_cache_write).abs() < 1e-9);
    assert!(
        (usage.cost.total
            - (expected_input_cost
                + expected_output_cost
                + expected_cache_read
                + expected_cache_write))
            .abs()
            < 1e-9
    );
    server.shutdown();
}

// ---------------------------------------------------------------------------
// OAuth (ports of openai-codex-oauth.test.ts refresh subset)
// ---------------------------------------------------------------------------

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

fn test_spec(token_url: String, base_url: String) -> OAuthProviderSpec<'static> {
    OAuthProviderSpec {
        id: "chatgpt-plus",
        display_name: "ChatGPT Plus/Pro (Codex)",
        base_url: leak(base_url),
        client_id: "app_EMoamEEZ73f0CkXaXp7hrann",
        scopes: &["openid", "profile", "email", "offline_access"],
        auth_url: Some("https://auth.openai.com/oauth/authorize"),
        token_url: Some(leak(token_url)),
        device_auth_url: None,
        redirect_uri: Some("http://localhost:1455/auth/callback"),
        is_subscription: true,
        login_label: Some("ChatGPT Plus/Pro"),
        api_label: "openai-codex-responses",
        default_models_fn: || vec![],
        extra_headers: None,
        token_request_encoding: OAuthTokenRequestEncoding::FormUrlEncoded,
        include_state_in_token_exchange: false,
    }
}

fn config_from_spec(spec: &OAuthProviderSpec<'static>) -> OAuthProviderConfig {
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
        extra_headers: None,
        token_request_encoding: spec.token_request_encoding,
        include_state_in_token_exchange: spec.include_state_in_token_exchange,
        default_models: vec![],
    }
}

fn expired_oauth_credential() -> otter_ai::auth::OAuthCredential {
    otter_ai::auth::OAuthCredential {
        inner: otter_ai::auth::OAuthCredentials {
            access: mock_token("acc_old"),
            refresh: "refresh-token-1".to_string(),
            expires: 0,
            extra: Default::default(),
        },
    }
}

async fn put_credential(store: &InMemoryCredentialStore, provider_id: &str, cred: Credential) {
    store
        .modify_fn(
            provider_id,
            Box::new(move |_| Box::pin(async move { Ok(Some(cred)) }) as ModifyFnOutput),
            AuthOperationOptions::default(),
        )
        .await
        .expect("write credential");
}

async fn get_credential(store: &InMemoryCredentialStore, provider_id: &str) -> Option<Credential> {
    store
        .read(provider_id, AuthOperationOptions::default())
        .await
        .expect("read credential")
}

#[tokio::test]
async fn refreshes_expired_tokens_through_the_token_endpoint() {
    let new_access = mock_token("acc_new");
    let server = MockServer::spawn(Arc::new(move |req| {
        assert_eq!(req.path, "/oauth/token");
        let form = req.form_body();
        assert_eq!(
            form.get("grant_type").map(String::as_str),
            Some("refresh_token")
        );
        assert_eq!(
            form.get("client_id").map(String::as_str),
            Some("app_EMoamEEZ73f0CkXaXp7hrann")
        );
        assert_eq!(
            form.get("refresh_token").map(String::as_str),
            Some("refresh-token-1")
        );
        MockResponse::json(
            200,
            serde_json::json!({
                "access_token": new_access,
                "refresh_token": "refresh-token-2",
                "expires_in": 3600,
            }),
        )
    }))
    .await;

    let auth = GenericOAuthAuth::new(config_from_spec(&test_spec(
        format!("{}/oauth/token", server.url()),
        server.url(),
    )));

    let before = chrono::Utc::now().timestamp_millis();
    let refreshed = auth
        .refresh(
            &expired_oauth_credential(),
            &otter_ai::CancellationToken::new(),
        )
        .await
        .expect("refresh succeeds");
    let after = chrono::Utc::now().timestamp_millis();

    assert_eq!(refreshed.inner.access, mock_token("acc_new"));
    assert_eq!(refreshed.inner.refresh, "refresh-token-2");
    assert!(refreshed.inner.expires >= before + 3600 * 1000);
    assert!(refreshed.inner.expires <= after + 3600 * 1000);
    assert_eq!(
        refreshed
            .inner
            .extra
            .get("account_id")
            .and_then(|v| v.as_str()),
        Some("acc_new")
    );
    server.shutdown();
}

#[tokio::test]
async fn refreshes_expired_tokens_with_json_token_requests_when_configured() {
    let new_access = mock_token("acc_claude_new");
    let server = MockServer::spawn(Arc::new(move |req| {
        assert_eq!(req.method, "POST");
        assert_eq!(req.header("content-type"), Some("application/json"));
        let body = req.json_body();
        assert_eq!(body["grant_type"], "refresh_token");
        assert_eq!(body["client_id"], "9d1c250a-e61b-44d9-88ed-5944d1962f5e");
        assert_eq!(body["refresh_token"], "refresh-token-1");
        MockResponse::json(
            200,
            serde_json::json!({
                "access_token": new_access,
                "refresh_token": "refresh-token-2",
                "expires_in": 3600,
            }),
        )
    }))
    .await;

    let auth = GenericOAuthAuth::new(OAuthProviderConfig {
        base_url: "https://api.anthropic.com/v1".to_string(),
        client_id: "9d1c250a-e61b-44d9-88ed-5944d1962f5e".to_string(),
        scopes: vec![
            "org:create_api_key".to_string(),
            "user:profile".to_string(),
            "user:inference".to_string(),
        ],
        auth_url: Some("https://claude.ai/oauth/authorize".to_string()),
        token_url: Some(server.url()),
        device_auth_url: None,
        redirect_uri: Some("https://console.anthropic.com/oauth/code/callback".to_string()),
        display_name: "Claude Pro/Max".to_string(),
        is_subscription: true,
        login_label: Some("Claude Pro/Max".to_string()),
        api_label: "anthropic-messages".to_string(),
        extra_headers: None,
        token_request_encoding: OAuthTokenRequestEncoding::Json,
        include_state_in_token_exchange: true,
        default_models: vec![],
    });

    let refreshed = auth
        .refresh(
            &expired_oauth_credential(),
            &otter_ai::CancellationToken::new(),
        )
        .await
        .expect("refresh succeeds");

    assert_eq!(refreshed.inner.access, mock_token("acc_claude_new"));
    assert_eq!(refreshed.inner.refresh, "refresh-token-2");
    server.shutdown();
}

#[tokio::test]
async fn refresh_failures_include_the_response_body() {
    let server = MockServer::spawn(static_handler(MockResponse::json(
        401,
        serde_json::json!({
            "error": {
                "message": "Could not validate your token. Please try signing in again.",
                "type": "invalid_request_error",
            }
        }),
    )))
    .await;

    let auth = GenericOAuthAuth::new(config_from_spec(&test_spec(
        format!("{}/oauth/token", server.url()),
        server.url(),
    )));

    let err = auth
        .refresh(
            &expired_oauth_credential(),
            &otter_ai::CancellationToken::new(),
        )
        .await
        .expect_err("refresh must fail");
    assert!(err.to_string().contains("failed (401)"), "err: {}", err);
    assert!(
        err.to_string().contains("Could not validate your token"),
        "err: {}",
        err
    );
    server.shutdown();
}

#[tokio::test]
async fn resolve_provider_auth_refreshes_and_persists_expired_credentials() {
    let new_access = mock_token("acc_refreshed");
    let server = MockServer::spawn(Arc::new(move |_| {
        MockResponse::json(
            200,
            serde_json::json!({
                "access_token": new_access,
                "refresh_token": "refresh-token-2",
                "expires_in": 3600,
            }),
        )
    }))
    .await;

    let provider = build_oauth_provider(test_spec(
        format!("{}/oauth/token", server.url()),
        server.url(),
    ));
    let store = InMemoryCredentialStore::new();
    put_credential(
        &store,
        "chatgpt-plus",
        Credential::OAuth(expired_oauth_credential()),
    )
    .await;

    let signal = otter_ai::CancellationToken::new();
    let result = resolve_provider_auth(
        &provider,
        &otter_ai::auth::default_provider_auth_context(),
        &store,
        AuthResolutionOverrides::default(),
        &signal,
    )
    .await
    .expect("auth resolves");

    assert_eq!(
        result.auth.api_key.as_deref(),
        Some(mock_token("acc_refreshed").as_str())
    );
    assert_eq!(result.source.as_deref(), Some("credential_store"));

    // The refreshed credential must be persisted back into the store.
    match get_credential(&store, "chatgpt-plus").await {
        Some(Credential::OAuth(stored)) => {
            assert_eq!(stored.inner.refresh, "refresh-token-2");
            assert!(stored.inner.expires > chrono::Utc::now().timestamp_millis());
        }
        other => panic!(
            "expected persisted oauth credential, got {:?}",
            other.map(|c| match c {
                Credential::OAuth(_) => "oauth",
                Credential::ApiKey(_) => "api_key",
            })
        ),
    }
    server.shutdown();
}

// ---------------------------------------------------------------------------
// Full-stack: Models → auth merge → Codex adapter (regression for the
// "auth dropped before provider.stream" bug)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn models_stream_merges_auth_and_routes_through_the_codex_adapter() {
    let token = mock_token("acc_full");
    let server = MockServer::spawn(static_handler(MockResponse::sse(basic_completion_events(
        "completed",
        None,
    ))))
    .await;

    let provider = build_oauth_provider(test_spec(
        format!("{}/oauth/token", server.url()),
        "https://chatgpt.com/backend-api".to_string(),
    ));
    let models = Models::new();
    models.set_provider_arc(Arc::new(provider));

    let store = InMemoryCredentialStore::new();
    let unexpired = otter_ai::auth::OAuthCredential {
        inner: otter_ai::auth::OAuthCredentials {
            access: token.clone(),
            refresh: "refresh-token-1".to_string(),
            expires: chrono::Utc::now().timestamp_millis() + 3_600_000,
            extra: Default::default(),
        },
    };
    put_credential(&store, "chatgpt-plus", Credential::OAuth(unexpired)).await;

    let models = models.with_credential_store(Arc::new(store));

    let model = codex_model("gpt-5.4");
    let options = SimpleStreamOptions {
        base_url: Some(server.url()),
        ..Default::default()
    };
    let result = models
        .complete(&model, user_context("Say hello"), options)
        .await
        .expect("completes through the full stack");

    assert_eq!(text_of(&result), "Hello");
    assert_eq!(stop_reason_of(&result), "stop");

    let recorded = server.recorded();
    let req = recorded
        .iter()
        .find(|r| r.method == "POST")
        .expect("SSE POST recorded");
    assert_eq!(req.path, "/codex/responses");
    assert_eq!(
        req.header("authorization"),
        Some(format!("Bearer {}", token).as_str())
    );
    assert_eq!(req.header("chatgpt-account-id"), Some("acc_full"));
    server.shutdown();
}

#[tokio::test]
async fn models_stream_forwards_session_cache_options_to_codex() {
    let token = mock_token("acc_session");
    let server = MockServer::spawn(static_handler(MockResponse::sse(basic_completion_events(
        "completed",
        None,
    ))))
    .await;

    let provider = build_oauth_provider(test_spec(
        format!("{}/oauth/token", server.url()),
        "https://chatgpt.com/backend-api".to_string(),
    ));
    let store = InMemoryCredentialStore::new();
    put_credential(
        &store,
        "chatgpt-plus",
        Credential::OAuth(otter_ai::auth::OAuthCredential {
            inner: otter_ai::auth::OAuthCredentials {
                access: token,
                refresh: "r".to_string(),
                expires: chrono::Utc::now().timestamp_millis() + 3_600_000,
                extra: Default::default(),
            },
        }),
    )
    .await;
    let models = Models::new().with_credential_store(Arc::new(store));
    models.set_provider_arc(Arc::new(provider));

    let model = codex_model("gpt-5.4");
    let options = SimpleStreamOptions {
        base_url: Some(server.url()),
        session_id: Some("session-full-stack".to_string()),
        cache_retention: Some(CacheRetention::Long),
        ..Default::default()
    };
    let result = models
        .complete(&model, user_context("Say hello"), options)
        .await
        .expect("completes");
    assert_eq!(stop_reason_of(&result), "stop");

    // Default transport is `auto`: a websocket upgrade GET may be attempted
    // first against this plain-HTTP mock before the SSE fallback, so pick
    // the actual POST.
    let recorded = server.recorded();
    let req = recorded
        .iter()
        .find(|r| r.method == "POST")
        .expect("SSE POST recorded");
    assert_eq!(req.header("session-id"), Some("session-full-stack"));
    assert_eq!(req.json_body()["prompt_cache_key"], "session-full-stack");
    server.shutdown();
}

// ---------------------------------------------------------------------------
// onPayload hook
// ---------------------------------------------------------------------------

#[tokio::test]
async fn on_payload_hook_observes_the_request_body() {
    let token = mock_token("acc_test");
    let server = MockServer::spawn(static_handler(MockResponse::sse(basic_completion_events(
        "completed",
        None,
    ))))
    .await;

    let seen = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));
    let sink = seen.clone();
    let opts = CodexStreamOptions {
        api_key: token,
        base_url: Some(server.url()),
        transport: CodexTransport::Sse,
        session_id: Some("hook-session".to_string()),
        cache_retention: CacheRetention::Short,
        on_payload: Some(Arc::new(move |body| {
            sink.lock().unwrap().push(body.clone());
        })),
        ..Default::default()
    };
    let (_, msg) = collect_events(stream_codex(
        &codex_model("gpt-5.4"),
        user_context("hi"),
        opts,
    ))
    .await;
    assert_eq!(stop_reason_of(&msg), "stop");

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0]["prompt_cache_key"], "hook-session");
    assert_eq!(seen[0]["model"], "gpt-5.4");
    server.shutdown();
}

// ---------------------------------------------------------------------------
// WebSocket transport (feature-gated; mirrors pi-ai websocket tests)
// ---------------------------------------------------------------------------

#[cfg(feature = "codex-websocket")]
mod websocket_transport {
    use super::*;
    use common::ws::{WsHandler, WsMockServer, WsRecordedRequest, WsReply};

    fn completion_frames() -> Vec<serde_json::Value> {
        vec![
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": { "type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": [] },
            }),
            serde_json::json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "delta": "Hello",
            }),
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{ "type": "output_text", "text": "Hello" }],
                },
            }),
            serde_json::json!({
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "end_turn": false,
                    "usage": { "input_tokens": 5, "output_tokens": 3, "total_tokens": 8 },
                },
            }),
        ]
    }

    fn ws_request_line() -> &'static str {
        "/codex/responses"
    }

    #[tokio::test]
    async fn streams_over_websocket_with_codex_beta_headers_and_frame() {
        let token = mock_token("acc_ws");
        let server = WsMockServer::spawn(Arc::new(|_conn, _idx, _path, _headers, frame| {
            assert_eq!(frame["type"], "response.create");
            assert_eq!(frame["store"], false);
            assert_eq!(frame["stream"], true);
            WsReply::Frames(completion_frames())
        }))
        .await;

        let opts = CodexStreamOptions {
            api_key: token.clone(),
            base_url: Some(server.url()),
            transport: CodexTransport::Websocket,
            session_id: Some("ws-session".to_string()),
            cache_retention: CacheRetention::Short,
            ..Default::default()
        };
        let (events, msg) = collect_events(stream_codex(
            &codex_model("gpt-5.4"),
            user_context("Say hello"),
            opts,
        ))
        .await;

        let types = event_types(&events);
        assert_eq!(types.iter().filter(|t| **t == "start").count(), 1);
        assert!(types.contains(&"text_delta"));
        assert!(types.contains(&"done"));
        assert_eq!(text_of(&msg), "Hello");
        assert_eq!(stop_reason_of(&msg), "stop");
        match &msg {
            Message::Assistant { end_turn, .. } => assert_eq!(end_turn, &Some(false)),
            _ => panic!("expected assistant message"),
        }

        let req: WsRecordedRequest = server.recorded()[0].clone();
        assert_eq!(req.path, ws_request_line());
        assert_eq!(
            req.header("authorization"),
            Some(format!("Bearer {}", token).as_str())
        );
        assert_eq!(req.header("chatgpt-account-id"), Some("acc_ws"));
        assert_eq!(
            req.header("openai-beta"),
            Some("responses_websockets=2026-02-06")
        );
        assert_eq!(req.header("session-id"), Some("ws-session"));
        assert_eq!(req.header("x-client-request-id"), Some("ws-session"));
        server.shutdown();
    }

    #[cfg(feature = "codex-zstd")]
    #[tokio::test]
    async fn zstd_compresses_sse_request_bodies() {
        use otter_ai::providers::openai_responses::CodexStreamOptions;
        let token = mock_token("acc_zstd");
        let server = MockServer::spawn(static_handler(MockResponse::sse(basic_completion_events(
            "completed",
            None,
        ))))
        .await;

        let large_text = "compress me ".repeat(400);
        let opts = CodexStreamOptions {
            api_key: token,
            base_url: Some(server.url()),
            transport: CodexTransport::Sse,
            ..Default::default()
        };
        let mut ctx = Context {
            system_prompt: Some("You are a helpful assistant.".to_string()),
            messages: vec![Message::user_from_string(&large_text)],
            ..Default::default()
        };
        let _ = &mut ctx;
        let (_, msg) = collect_events(stream_codex(&codex_model("gpt-5.4"), ctx, opts)).await;
        assert_eq!(stop_reason_of(&msg), "stop");

        let req: RecordedRequest = server.recorded()[0].clone();
        assert_eq!(req.header("content-encoding"), Some("zstd"));
        // The raw wire bytes are zstd-compressed and decompress back to the
        // request JSON containing the large text.
        #[cfg(feature = "codex-zstd")]
        {
            let decoded: Vec<u8> = zstd::decode_all(std::io::Cursor::new(&req.raw_body))
                .expect("raw body decompresses");
            let json: serde_json::Value =
                serde_json::from_slice(&decoded).expect("decoded body is JSON");
            assert_eq!(
                json["input"][0]["content"][0]["text"],
                serde_json::json!(large_text)
            );
        }
        assert_eq!(
            req.json_body()["input"][0]["content"][0]["text"],
            large_text
        );
        server.shutdown();
    }

    #[tokio::test]
    async fn auto_transport_falls_back_to_sse_when_websocket_connect_fails() {
        let token = mock_token("acc_fallback");
        // Plain HTTP server: the WS upgrade handshake will fail (no 101),
        // so `auto` must retry the same request over SSE.
        let server = MockServer::spawn(static_handler(MockResponse::sse(basic_completion_events(
            "completed",
            None,
        ))))
        .await;

        let opts = CodexStreamOptions {
            api_key: token,
            base_url: Some(server.url()),
            transport: CodexTransport::Auto,
            ..Default::default()
        };
        let (_, msg) = collect_events(stream_codex(
            &codex_model("gpt-5.4"),
            user_context("Say hello"),
            opts,
        ))
        .await;
        assert_eq!(text_of(&msg), "Hello");
        assert_eq!(stop_reason_of(&msg), "stop");

        // The failed WS handshake (GET) plus the successful SSE POST.
        let recorded = server.recorded();
        assert!(
            recorded.iter().any(|r| r.method == "POST"),
            "SSE POST recorded"
        );
        server.shutdown();
    }

    #[tokio::test]
    async fn explicit_websocket_transport_falls_back_to_sse_when_connect_fails() {
        let token = mock_token("acc_ws_fail");
        let server = MockServer::spawn(static_handler(MockResponse::sse(basic_completion_events(
            "completed",
            None,
        ))))
        .await;

        let opts = CodexStreamOptions {
            api_key: token,
            base_url: Some(server.url()),
            transport: CodexTransport::Websocket,
            ..Default::default()
        };
        let (_, msg) = collect_events(stream_codex(
            &codex_model("gpt-5.4"),
            user_context("Say hello"),
            opts,
        ))
        .await;
        assert_eq!(stop_reason_of(&msg), "stop");
        assert_eq!(text_of(&msg), "Hello");
        server.shutdown();
    }

    fn completed_frame(response_id: &str, message_id: &str, text: &str) -> Vec<serde_json::Value> {
        vec![
            serde_json::json!({
                "type": "response.created",
                "response": { "id": response_id },
            }),
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": { "type": "message", "id": message_id, "role": "assistant", "status": "in_progress", "content": [] },
            }),
            serde_json::json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "delta": text,
            }),
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": message_id,
                    "role": "assistant",
                    "status": "completed",
                    "content": [{ "type": "output_text", "text": text }],
                },
            }),
            serde_json::json!({
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "status": "completed",
                    "end_turn": false,
                    "usage": { "input_tokens": 5, "output_tokens": 3, "total_tokens": 8 },
                },
            }),
        ]
    }

    fn append_message(context: &mut Context, msg: Message) {
        context.messages.push(msg);
    }

    #[tokio::test]
    async fn uses_cached_websocket_context_with_auto_transport() {
        otter_ai::providers::openai_responses::close_codex_ws_sessions(Some("session-auto"));
        otter_ai::providers::openai_responses::reset_codex_ws_debug_stats(Some("session-auto"));
        let token = mock_token("acc_auto");
        let headers_seen: Arc<Mutex<Option<std::collections::HashMap<String, String>>>> =
            Arc::new(Mutex::new(None));
        let seen = headers_seen.clone();
        let server = WsMockServer::spawn(Arc::new(move |_conn, _idx, _path, headers, frame| {
            assert_eq!(frame["type"], "response.create");
            assert_eq!(frame["store"], false);
            *seen.lock().unwrap() = Some(headers.clone());
            WsReply::Frames(completed_frame("resp_1", "msg_1", "Hello"))
        }))
        .await;

        let opts = CodexStreamOptions {
            api_key: token,
            base_url: Some(server.url()),
            session_id: Some("session-auto".to_string()),
            cache_retention: CacheRetention::Short,
            transport: CodexTransport::Auto,
            ..Default::default()
        };
        let (events, msg) = collect_events(stream_codex(
            &codex_model("gpt-5.4"),
            user_context("Say hello"),
            opts,
        ))
        .await;

        assert_eq!(stop_reason_of(&msg), "stop");
        match &msg {
            Message::Assistant { end_turn, .. } => assert_eq!(end_turn, &Some(false)),
            _ => panic!("expected assistant message"),
        }
        assert_eq!(
            event_types(&events)
                .iter()
                .filter(|t| **t == "start")
                .count(),
            1
        );
        let headers = headers_seen
            .lock()
            .unwrap()
            .clone()
            .expect("ws headers captured");
        assert_eq!(
            headers.get("session-id").map(|s| s.as_str()),
            Some("session-auto")
        );
        assert!(!headers.contains_key("session_id"));
        assert_eq!(
            headers.get("x-client-request-id").map(|s| s.as_str()),
            Some("session-auto")
        );
        let stats = otter_ai::providers::openai_responses::get_codex_ws_debug_stats("session-auto")
            .expect("stats");
        assert_eq!(stats.requests, 1);
        assert_eq!(stats.connections_created, 1);
        assert_eq!(stats.connections_reused, 0);
        assert_eq!(stats.cached_context_requests, 1);
        assert_eq!(stats.full_context_requests, 1);
        assert_eq!(stats.delta_requests, 0);
        server.shutdown();
    }

    #[tokio::test]
    async fn scopes_cached_websockets_to_authenticated_account() {
        otter_ai::providers::openai_responses::close_codex_ws_sessions(Some("shared-session"));
        otter_ai::providers::openai_responses::reset_codex_ws_debug_stats(Some("shared-session"));
        let connected_headers: Arc<Mutex<Vec<std::collections::HashMap<String, String>>>> =
            Arc::new(Mutex::new(Vec::new()));
        let seen = connected_headers.clone();
        let server = WsMockServer::spawn(Arc::new(move |_conn, idx, _path, headers, _frame| {
            if idx == 1 {
                seen.lock().unwrap().push(headers.clone());
            }
            WsReply::Frames(completed_frame("resp_1", "msg_1", "Hi"))
        }))
        .await;

        let make_opts = |account: &str| CodexStreamOptions {
            api_key: mock_token(account),
            base_url: Some(server.url()),
            session_id: Some("shared-session".to_string()),
            cache_retention: CacheRetention::Short,
            transport: CodexTransport::WebsocketCached,
            ..Default::default()
        };
        collect_events(stream_codex(
            &codex_model("gpt-5.4"),
            user_context("one"),
            make_opts("account-a"),
        ))
        .await;
        collect_events(stream_codex(
            &codex_model("gpt-5.4"),
            user_context("two"),
            make_opts("account-b"),
        ))
        .await;
        collect_events(stream_codex(
            &codex_model("gpt-5.4"),
            user_context("three"),
            make_opts("account-a"),
        ))
        .await;

        let headers = connected_headers.lock().unwrap();
        let accounts: Vec<&str> = headers
            .iter()
            .map(|h| {
                h.get("chatgpt-account-id")
                    .map(|s| s.as_str())
                    .unwrap_or("")
            })
            .collect();
        assert_eq!(accounts, vec!["account-a", "account-b"]);
        assert_eq!(
            headers[0].get("authorization"),
            Some(&format!("Bearer {}", mock_token("account-a")))
        );
        assert_eq!(
            headers[1].get("authorization"),
            Some(&format!("Bearer {}", mock_token("account-b")))
        );
        let stats =
            otter_ai::providers::openai_responses::get_codex_ws_debug_stats("shared-session")
                .expect("stats");
        assert_eq!(stats.connections_created, 2);
        assert_eq!(stats.connections_reused, 1);
        assert_eq!(stats.requests, 3);
        server.shutdown();
    }

    #[tokio::test]
    async fn closes_one_shot_websockets_when_cache_retention_none() {
        otter_ai::providers::openai_responses::close_codex_ws_sessions(Some("one-off-summary"));
        otter_ai::providers::openai_responses::reset_codex_ws_debug_stats(Some("one-off-summary"));
        let token = mock_token("acc_oneoff");
        let frames: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
        let collected = frames.clone();
        let collected2 = collected.clone();
        let server = WsMockServer::spawn(Arc::new(move |_conn, _idx, _path, _headers, frame| {
            collected2.lock().unwrap().push(frame.clone());
            WsReply::Frames(completed_frame("resp_1", "msg_1", "Hi"))
        }))
        .await;

        let opts = CodexStreamOptions {
            api_key: token,
            base_url: Some(server.url()),
            session_id: Some("one-off-summary".to_string()),
            cache_retention: CacheRetention::None,
            transport: CodexTransport::Auto,
            ..Default::default()
        };
        collect_events(stream_codex(
            &codex_model("gpt-5.4"),
            user_context("one"),
            opts.clone(),
        ))
        .await;
        collect_events(stream_codex(
            &codex_model("gpt-5.4"),
            user_context("two"),
            opts,
        ))
        .await;

        let requests = server.recorded();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].connection, 1);
        assert_eq!(
            requests[1].connection, 2,
            "one-shot connections are not reused"
        );
        for req in &requests {
            assert!(req.frame.get("prompt_cache_key").is_none());
        }
        assert!(collected
            .lock()
            .unwrap()
            .iter()
            .all(|f| f.get("prompt_cache_key").is_none()));
        assert!(
            otter_ai::providers::openai_responses::get_codex_ws_debug_stats("one-off-summary")
                .is_none(),
            "no stats for uncached sessions"
        );
        server.shutdown();
    }

    #[tokio::test]
    async fn falls_back_to_sse_when_websocket_connect_times_out() {
        otter_ai::providers::openai_responses::close_codex_ws_sessions(Some("ws-connect-timeout"));
        otter_ai::providers::openai_responses::reset_codex_ws_debug_stats(Some(
            "ws-connect-timeout",
        ));
        let token = mock_token("acc_conn_timeout");
        // Combined server: WS upgrades hang (connect timeout), HTTP is SSE.
        let server =
            WsMockServer::spawn_combined_pending_handshake(Arc::new(|req: &RecordedRequest| {
                assert_eq!(req.method, "POST");
                assert_eq!(req.path, "/codex/responses");
                MockResponse::sse(basic_completion_events("completed", None))
            }))
            .await;

        let opts = CodexStreamOptions {
            api_key: token,
            base_url: Some(server.url()),
            session_id: Some("ws-connect-timeout".to_string()),
            cache_retention: CacheRetention::Short,
            transport: CodexTransport::Auto,
            websocket_connect_timeout_ms: Some(50),
            ..Default::default()
        };
        let (_, msg) = collect_events(stream_codex(
            &codex_model("gpt-5.4"),
            user_context("Say hello"),
            opts,
        ))
        .await;
        eprintln!(
            "CONNECT_TIMEOUT_TEST err={:?}",
            match &msg {
                otter_ai::Message::Assistant { error_message, .. } => error_message,
                _ => &None,
            }
        );
        assert_eq!(stop_reason_of(&msg), "stop");
        assert_eq!(text_of(&msg), "Hello");
        assert_eq!(server.recorded_http().len(), 1, "SSE fallback happened");
        let stats =
            otter_ai::providers::openai_responses::get_codex_ws_debug_stats("ws-connect-timeout")
                .expect("stats");
        assert_eq!(stats.websocket_failures, 1);
        assert_eq!(stats.sse_fallbacks, 1);
        assert_eq!(stats.websocket_fallback_active, Some(true));
        server.shutdown();
    }

    #[tokio::test]
    async fn reconnects_once_when_websocket_connection_limit_reached() {
        otter_ai::providers::openai_responses::close_codex_ws_sessions(Some("limit-session"));
        otter_ai::providers::openai_responses::reset_codex_ws_debug_stats(Some("limit-session"));
        let token = mock_token("acc_conn_limit");
        let conn_counter: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
        let count = conn_counter.clone();
        let count2 = count.clone();
        let server = WsMockServer::spawn(Arc::new(move |conn, _idx, _path, _headers, _frame| {
            let mut c = count2.lock().unwrap();
            *c = std::cmp::max(*c, conn);
            if conn == 1 {
                WsReply::Frames(vec![serde_json::json!({
                    "type": "error",
                    "error": { "code": "websocket_connection_limit_reached", "message": "too many" },
                })])
            } else {
                WsReply::Frames(completed_frame("resp_1", "msg_1", "Recovered"))
            }
        }))
        .await;

        let opts = CodexStreamOptions {
            api_key: token,
            base_url: Some(server.url()),
            session_id: Some("limit-session".to_string()),
            cache_retention: CacheRetention::Short,
            transport: CodexTransport::Auto,
            ..Default::default()
        };
        let (_, msg) = collect_events(stream_codex(
            &codex_model("gpt-5.4"),
            user_context("Say hello"),
            opts,
        ))
        .await;
        eprintln!(
            "CONN_LIMIT err={:?}",
            match &msg {
                otter_ai::Message::Assistant { error_message, .. } => error_message,
                _ => &None,
            }
        );
        assert_eq!(stop_reason_of(&msg), "stop");
        assert_eq!(text_of(&msg), "Recovered");
        assert_eq!(
            *count.lock().unwrap(),
            2,
            "reconnected once on a fresh connection"
        );
        server.shutdown();
    }

    #[tokio::test]
    async fn falls_back_to_sse_when_websocket_idle_before_first_event() {
        otter_ai::providers::openai_responses::close_codex_ws_sessions(Some(
            "ws-idle-before-start",
        ));
        otter_ai::providers::openai_responses::reset_codex_ws_debug_stats(Some(
            "ws-idle-before-start",
        ));
        let token = mock_token("acc_idle_before");
        // Combined server: WS handshake completes, frame arrives, no reply
        // (Hang) → client idles for timeout_ms → SSE fallback on the same URL.
        let server = WsMockServer::spawn_combined(
            Arc::new(|_conn, _idx, _path, _headers, _frame| WsReply::Hang),
            Arc::new(|req: &RecordedRequest| {
                assert_eq!(req.method, "POST");
                MockResponse::sse(basic_completion_events("completed", None))
            }),
        )
        .await;

        let opts = CodexStreamOptions {
            api_key: token,
            base_url: Some(server.url()),
            session_id: Some("ws-idle-before-start".to_string()),
            cache_retention: CacheRetention::Short,
            transport: CodexTransport::Auto,
            timeout_ms: Some(50),
            ..Default::default()
        };
        let (_, msg) = collect_events(stream_codex(
            &codex_model("gpt-5.4"),
            user_context("Say hello"),
            opts,
        ))
        .await;
        assert_eq!(stop_reason_of(&msg), "stop");
        assert_eq!(text_of(&msg), "Hello");
        assert_eq!(server.recorded_http().len(), 1, "SSE fallback happened");
        let stats =
            otter_ai::providers::openai_responses::get_codex_ws_debug_stats("ws-idle-before-start")
                .expect("stats");
        assert_eq!(stats.websocket_failures, 1);
        assert_eq!(stats.sse_fallbacks, 1);
        assert_eq!(stats.websocket_fallback_active, Some(true));
        server.shutdown();
    }

    #[tokio::test]
    async fn errors_when_websocket_idle_after_stream_started() {
        otter_ai::providers::openai_responses::close_codex_ws_sessions(Some("idle-after-start"));
        otter_ai::providers::openai_responses::reset_codex_ws_debug_stats(Some("idle-after-start"));
        let token = mock_token("acc_idle_after");
        let server = WsMockServer::spawn(Arc::new(|_conn, _idx, _path, _headers, _frame| {
            // Send a single event (start is emitted), then go silent.
            WsReply::Frames(vec![serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": { "type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": [] },
            })])
        }))
        .await;

        let opts = CodexStreamOptions {
            api_key: token,
            base_url: Some(server.url()),
            session_id: Some("idle-after-start".to_string()),
            cache_retention: CacheRetention::Short,
            transport: CodexTransport::Auto,
            timeout_ms: Some(50),
            ..Default::default()
        };
        let (events, msg) = collect_events(stream_codex(
            &codex_model("gpt-5.4"),
            user_context("Say hello"),
            opts,
        ))
        .await;
        assert_eq!(stop_reason_of(&msg), "error");
        assert_eq!(
            match &msg {
                Message::Assistant {
                    error_message: Some(e),
                    ..
                } => e.clone(),
                _ => String::new(),
            },
            "WebSocket idle timeout after 50ms"
        );
        // start was emitted (stream started) → no SSE fallback.
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, AssistantMessageEvent::Start { .. }))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, AssistantMessageEvent::Error { .. }))
                .count(),
            1
        );
        server.shutdown();
    }

    #[tokio::test]
    async fn opens_fresh_cached_websocket_before_backend_connection_age_limit() {
        otter_ai::providers::openai_responses::close_codex_ws_sessions(Some("aged-ws-session"));
        otter_ai::providers::openai_responses::reset_codex_ws_debug_stats(Some("aged-ws-session"));
        let token = mock_token("acc_age");
        let sent_conns: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
        let conns = sent_conns.clone();
        let conns2 = conns.clone();
        let server = WsMockServer::spawn(Arc::new(move |conn, _idx, _path, _headers, _frame| {
            conns2.lock().unwrap().push(conn);
            let rid = format!("resp_{}", conn);
            let mid = format!("msg_{}", conn);
            WsReply::Frames(completed_frame(&rid, &mid, "Hi"))
        }))
        .await;

        let opts = CodexStreamOptions {
            api_key: token,
            base_url: Some(server.url()),
            session_id: Some("aged-ws-session".to_string()),
            cache_retention: CacheRetention::Short,
            transport: CodexTransport::WebsocketCached,
            websocket_max_age_ms: Some(100),
            ..Default::default()
        };
        collect_events(stream_codex(
            &codex_model("gpt-5.4"),
            user_context("one"),
            opts.clone(),
        ))
        .await;
        tokio::time::sleep(Duration::from_millis(200)).await;
        collect_events(stream_codex(
            &codex_model("gpt-5.4"),
            user_context("two"),
            opts,
        ))
        .await;

        let conns = conns.lock().unwrap();
        assert_eq!(
            conns.as_slice(),
            &[1, 2],
            "age-limited connection is replaced"
        );
        let stats =
            otter_ai::providers::openai_responses::get_codex_ws_debug_stats("aged-ws-session")
                .expect("stats");
        assert_eq!(stats.connections_created, 2);
        assert_eq!(stats.connections_reused, 0);
        server.shutdown();
    }

    #[tokio::test]
    async fn sends_only_response_input_deltas_in_websocket_cached_mode() {
        otter_ai::providers::openai_responses::close_codex_ws_sessions(Some("session-1"));
        otter_ai::providers::openai_responses::reset_codex_ws_debug_stats(Some("session-1"));
        let token = mock_token("acc_delta");
        let sent_bodies: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
        let bodies = sent_bodies.clone();
        let bodies2 = bodies.clone();
        let server = WsMockServer::spawn(Arc::new(move |_conn, idx, _path, _headers, frame| {
            bodies2.lock().unwrap().push(frame.clone());
            if idx == 1 {
                WsReply::Frames(vec![
                    serde_json::json!({ "type": "response.created", "response": { "id": "resp_1" } }),
                    serde_json::json!({
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": { "type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "sample_tool", "input": "" },
                    }),
                    serde_json::json!({ "type": "response.custom_tool_call_input.delta", "output_index": 0, "item_id": "ctc_1", "delta": "abc" }),
                    serde_json::json!({ "type": "response.custom_tool_call_input.done", "output_index": 0, "item_id": "ctc_1", "input": "abc" }),
                    serde_json::json!({
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": { "type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "sample_tool", "input": "abc" },
                    }),
                    serde_json::json!({
                        "type": "response.completed",
                        "response": {
                            "id": "resp_1",
                            "status": "completed",
                            "end_turn": false,
                            "usage": { "input_tokens": 5, "output_tokens": 3, "total_tokens": 8 },
                        },
                    }),
                ])
            } else {
                WsReply::Frames(completed_frame("resp_2", "msg_2", "done"))
            }
        }))
        .await;

        let mut ctx = Context {
            system_prompt: Some("You are a helpful assistant.".to_string()),
            messages: vec![Message::user_from_string("Use the tool")],
            tools: vec![Tool {
                name: "sample_tool".to_string(),
                description: Some("Sample tool".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": { "payload": { "type": "string" } },
                    "required": ["payload"],
                    "additionalProperties": false,
                }),
                constrained_sampling: Some(ToolConstrainedSampling {
                    sampling_type: "grammar".to_string(),
                    variants: Some(ToolGrammarVariants {
                        openai_lark: Some("start: /[a-z]+/".to_string()),
                        openai_regex: None,
                    }),
                }),
            }],
            ..Default::default()
        };
        let opts = CodexStreamOptions {
            api_key: token,
            base_url: Some(server.url()),
            session_id: Some("session-1".to_string()),
            cache_retention: CacheRetention::Short,
            transport: CodexTransport::WebsocketCached,
            ..Default::default()
        };
        let (_, first) = collect_events(stream_codex(
            &codex_model("gpt-5.4"),
            ctx.clone(),
            opts.clone(),
        ))
        .await;
        // Assistant tool call + tool result + follow-up user message.
        append_message(&mut ctx, first);
        ctx.messages.push(Message::ToolResult {
            tool_call_id: "call_1|ctc_1".to_string(),
            tool_name: "sample_tool".to_string(),
            content: vec![ContentBlock::Text {
                text: "real result".to_string(),
                text_signature: None,
            }],
            is_error: false,
            timestamp: 0,
        });
        ctx.messages.push(Message::user_from_string("Now finish"));
        let (_, _second) = collect_events(stream_codex(&codex_model("gpt-5.4"), ctx, opts)).await;

        let bodies = bodies.lock().unwrap();
        assert_eq!(bodies.len(), 2);
        let first_body = &bodies[0];
        let second_body = &bodies[1];
        assert_eq!(first_body["store"], false);
        assert!(first_body.get("previous_response_id").is_none());
        assert_eq!(
            first_body["input"][0],
            serde_json::json!({ "role": "user", "content": [{ "type": "input_text", "text": "Use the tool" }] })
        );
        assert_eq!(second_body["store"], false);
        assert_eq!(second_body["previous_response_id"], "resp_1");
        assert_eq!(
            second_body["input"],
            serde_json::json!([
                { "type": "custom_tool_call_output", "call_id": "call_1", "output": "real result" },
                { "role": "user", "content": [{ "type": "input_text", "text": "Now finish" }] },
            ])
        );
        let stats = otter_ai::providers::openai_responses::get_codex_ws_debug_stats("session-1")
            .expect("stats");
        assert_eq!(stats.requests, 2);
        assert_eq!(stats.connections_created, 1);
        assert_eq!(stats.connections_reused, 1);
        assert_eq!(stats.cached_context_requests, 2);
        assert_eq!(stats.store_true_requests, 0);
        assert_eq!(stats.full_context_requests, 1);
        assert_eq!(stats.delta_requests, 1);
        assert_eq!(stats.last_delta_input_items, Some(2));
        assert_eq!(stats.last_previous_response_id.as_deref(), Some("resp_1"));
        server.shutdown();
    }

    #[tokio::test]
    async fn recovers_missing_cached_websocket_continuation_via_websocket() {
        missing_continuation_recovery("websocket").await;
    }

    #[tokio::test]
    async fn recovers_missing_cached_websocket_continuation_via_sse() {
        missing_continuation_recovery("sse").await;
    }

    async fn missing_continuation_recovery(recovery_transport: &str) {
        let recovery_transport = recovery_transport.to_string();
        let token = mock_token("acc_missing_cont");
        let session = format!("missing-continuation-{}", recovery_transport);
        otter_ai::providers::openai_responses::close_codex_ws_sessions(Some(&session));
        otter_ai::providers::openai_responses::reset_codex_ws_debug_stats(Some(&session));
        let sent_bodies: Arc<Mutex<Vec<(usize, serde_json::Value)>>> =
            Arc::new(Mutex::new(Vec::new()));
        let bodies = sent_bodies.clone();
        let bodies2 = bodies.clone();
        let rt = recovery_transport.clone();
        // Global request counter across connections (pi-ai counts sentBodies
        // globally, not per socket).
        let request_counter = Arc::new(AtomicUsize::new(0));
        let counter = request_counter.clone();
        let ws_handler: WsHandler = Arc::new(move |conn, _idx, _path, _headers, frame| {
            let request_num = counter.fetch_add(1, Ordering::SeqCst) + 1;
            bodies.lock().unwrap().push((conn, frame.clone()));
            if request_num == 2 {
                // Delta request → previous response not found.
                WsReply::Frames(vec![serde_json::json!({
                    "type": "error",
                    "error": { "code": "previous_response_not_found", "message": "Previous response with id 'resp_1' not found." },
                })])
            } else if request_num == 3 && rt == "sse" {
                // Websocket retry fails at the transport level → SSE fallback.
                WsReply::Close
            } else {
                let (rid, mid, text) = if request_num == 1 {
                    (
                        "resp_1".to_string(),
                        "msg_1".to_string(),
                        "Hello".to_string(),
                    )
                } else {
                    (
                        "resp_2".to_string(),
                        "msg_2".to_string(),
                        "Recovered".to_string(),
                    )
                };
                WsReply::Frames(completed_frame(&rid, &mid, &text))
            }
        });
        let server = WsMockServer::spawn_combined(
            ws_handler,
            Arc::new(|_req: &RecordedRequest| {
                MockResponse::sse(basic_completion_events("completed", None))
            }),
        )
        .await;

        let first_opts = CodexStreamOptions {
            api_key: token.clone(),
            base_url: Some(server.url()),
            session_id: Some(session.clone()),
            cache_retention: CacheRetention::Short,
            transport: CodexTransport::WebsocketCached,
            ..Default::default()
        };
        let mut ctx = Context {
            system_prompt: Some("You are a helpful assistant.".to_string()),
            messages: vec![Message::user_from_string("Say hello")],
            ..Default::default()
        };
        let (_, first) = collect_events(stream_codex(
            &codex_model("gpt-5.4"),
            ctx.clone(),
            first_opts,
        ))
        .await;

        append_message(&mut ctx, first);
        ctx.messages.push(Message::user_from_string("Now finish"));
        let second_opts = CodexStreamOptions {
            api_key: token,
            base_url: Some(server.url()),
            session_id: Some(session.clone()),
            cache_retention: CacheRetention::Short,
            transport: CodexTransport::WebsocketCached,
            ..Default::default()
        };
        let (events, second) =
            collect_events(stream_codex(&codex_model("gpt-5.4"), ctx, second_opts)).await;

        assert_eq!(stop_reason_of(&second), "stop");
        let expected_text = if recovery_transport == "sse" {
            "Hello"
        } else {
            "Recovered"
        };
        assert_eq!(text_of(&second), expected_text);
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, AssistantMessageEvent::Start { .. }))
                .count(),
            1
        );
        assert!(events
            .iter()
            .all(|e| !matches!(e, AssistantMessageEvent::Error { .. })));

        let bodies = bodies2.lock().unwrap();
        assert_eq!(bodies.len(), 3);
        let conn_ids: Vec<usize> = bodies.iter().map(|(c, _)| *c).collect();
        assert_eq!(conn_ids, vec![1, 1, 2]);
        let first_body = &bodies[0].1;
        let delta_body = &bodies[1].1;
        let retry_body = &bodies[2].1;
        assert!(first_body.get("previous_response_id").is_none());
        assert_eq!(delta_body["previous_response_id"], "resp_1");
        assert_eq!(
            delta_body["input"],
            serde_json::json!([{ "role": "user", "content": [{ "type": "input_text", "text": "Now finish" }] }])
        );
        assert!(retry_body.get("previous_response_id").is_none());
        assert_eq!(retry_body["input"].as_array().map(|a| a.len()), Some(3));
        let stats = otter_ai::providers::openai_responses::get_codex_ws_debug_stats(&session)
            .expect("stats");
        assert_eq!(stats.requests, 3);
        assert_eq!(stats.connections_created, 2);
        assert_eq!(stats.connections_reused, 1);
        assert_eq!(stats.full_context_requests, 2);
        assert_eq!(stats.delta_requests, 1);
        assert_eq!(
            stats.websocket_failures,
            if recovery_transport == "sse" { 1 } else { 0 }
        );
        assert_eq!(
            stats.sse_fallbacks,
            if recovery_transport == "sse" { 1 } else { 0 }
        );
        server.shutdown();
    }
}

// ---------------------------------------------------------------------------
// PKCE browser OAuth login (ports of pi-ai openai-codex-oauth browser flow)
// ---------------------------------------------------------------------------

mod browser_login {
    use super::*;
    use otter_ai::auth::{AuthEvent, AuthInteraction, AuthPrompt, OAuthCredential};

    #[derive(Clone)]
    struct TestInteraction {
        prompt_answer: Arc<Mutex<String>>,
        /// When set, `prompt` blocks until the gate is released (the test
        /// simulates a user who has not typed anything yet).
        prompt_gate: Arc<Mutex<Option<tokio::sync::oneshot::Receiver<()>>>>,
        events: Arc<Mutex<Vec<AuthEvent>>>,
    }

    impl TestInteraction {
        fn new(answer: impl Into<String>) -> Self {
            Self {
                prompt_answer: Arc::new(Mutex::new(answer.into())),
                prompt_gate: Arc::new(Mutex::new(None)),
                events: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn gate(&self) -> tokio::sync::oneshot::Sender<()> {
            let (tx, rx) = tokio::sync::oneshot::channel();
            *self.prompt_gate.lock().unwrap() = Some(rx);
            tx
        }

        fn set_answer(&self, answer: impl Into<String>) {
            *self.prompt_answer.lock().unwrap() = answer.into();
        }

        async fn auth_url(&self) -> String {
            loop {
                let url = {
                    let events = self.events.lock().unwrap();
                    match events.first() {
                        Some(AuthEvent::AuthUrl { url, .. }) => Some(url.clone()),
                        _ => None,
                    }
                };
                if let Some(url) = url {
                    return url;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    }

    #[otter_ai::async_trait]
    impl AuthInteraction for TestInteraction {
        fn signal(&self) -> Option<&otter_ai::CancellationToken> {
            None
        }

        async fn prompt(&self, _prompt: AuthPrompt) -> anyhow::Result<String> {
            let gate = self.prompt_gate.lock().unwrap().take();
            if let Some(rx) = gate {
                let _ = rx.await;
            }
            Ok(self.prompt_answer.lock().unwrap().clone())
        }

        fn notify(&self, event: AuthEvent) {
            self.events.lock().unwrap().push(event);
        }
    }

    fn query_params(url: &str) -> std::collections::HashMap<String, String> {
        url.split_once('?')
            .map(|(_, q)| {
                q.split('&')
                    .filter_map(|p| p.split_once('='))
                    .map(|(k, v)| (url_decode_compat(k), url_decode_compat(v)))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn codex_auth_config(redirect_uri: String, token_url: String) -> OAuthProviderConfig {
        OAuthProviderConfig {
            base_url: "https://chatgpt.com/backend-api".to_string(),
            client_id: "app_EMoamEEZ73f0CkXaXp7hrann".to_string(),
            scopes: vec![
                "openid".into(),
                "profile".into(),
                "email".into(),
                "offline_access".into(),
            ],
            auth_url: Some("https://auth.openai.com/oauth/authorize".to_string()),
            token_url: Some(token_url),
            device_auth_url: None,
            redirect_uri: Some(redirect_uri),
            display_name: "ChatGPT Plus/Pro (Codex)".to_string(),
            is_subscription: true,
            login_label: Some("ChatGPT Plus/Pro".to_string()),
            api_label: "openai-codex-responses".to_string(),
            extra_headers: None,
            token_request_encoding: OAuthTokenRequestEncoding::FormUrlEncoded,
            include_state_in_token_exchange: false,
            default_models: vec![],
        }
    }

    fn claude_auth_config(token_url: String) -> OAuthProviderConfig {
        OAuthProviderConfig {
            base_url: "https://api.anthropic.com/v1".to_string(),
            client_id: "9d1c250a-e61b-44d9-88ed-5944d1962f5e".to_string(),
            scopes: vec![
                "org:create_api_key".into(),
                "user:profile".into(),
                "user:inference".into(),
            ],
            auth_url: Some("https://claude.ai/oauth/authorize".to_string()),
            token_url: Some(token_url),
            device_auth_url: None,
            redirect_uri: Some("https://console.anthropic.com/oauth/code/callback".to_string()),
            display_name: "Claude Pro/Max".to_string(),
            is_subscription: true,
            login_label: Some("Claude Pro/Max".to_string()),
            api_label: "anthropic-messages".to_string(),
            extra_headers: None,
            token_request_encoding: OAuthTokenRequestEncoding::Json,
            include_state_in_token_exchange: true,
            default_models: vec![],
        }
    }

    fn free_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0")
            .expect("bind")
            .local_addr()
            .expect("addr")
            .port()
    }

    fn mock_access_token(account_id: &str) -> String {
        mock_token(account_id)
    }

    fn token_response_json(account_id: &str) -> serde_json::Value {
        serde_json::json!({
            "access_token": mock_access_token(account_id),
            "refresh_token": "refresh-123",
            "expires_in": 3600,
            "token_type": "Bearer",
        })
    }

    #[tokio::test]
    async fn browser_login_exchanges_authorization_code() {
        let token_server = MockServer::spawn(Arc::new(|req: &RecordedRequest| {
            assert_eq!(req.method, "POST");
            let form = req.form_body();
            assert_eq!(
                form.get("grant_type").map(|s| s.as_str()),
                Some("authorization_code")
            );
            assert_eq!(
                form.get("client_id").map(|s| s.as_str()),
                Some("app_EMoamEEZ73f0CkXaXp7hrann")
            );
            assert!(form.contains_key("code"));
            assert!(form.contains_key("code_verifier"));
            assert_eq!(
                form.get("redirect_uri").map(|s| s.as_str()),
                Some("http://localhost:1455/auth/callback")
            );
            MockResponse::json(200, token_response_json("acc_browser"))
        }))
        .await;

        let interaction = TestInteraction::new("");
        let gate = interaction.gate();
        let auth = GenericOAuthAuth::new(codex_auth_config(
            "http://localhost:1455/auth/callback".to_string(),
            token_server.url(),
        ));

        let interaction_for_task = interaction.clone();
        let login_task = tokio::spawn(async move { auth.login(&interaction_for_task).await });

        // The manual prompt pastes a redirect URL that carries the REAL state
        // (the user's browser was redirected with it).
        let auth_url = interaction.auth_url().await;
        let real_state = query_params(&auth_url).get("state").cloned().unwrap();
        interaction.set_answer(format!(
            "http://localhost:1455/auth/callback?code=auth-code-123&state={}",
            urlencode_compat(&real_state)
        ));
        gate.send(()).expect("release manual prompt");

        let credential: OAuthCredential = login_task.await.expect("task").expect("login succeeds");

        assert_eq!(credential.inner.access, mock_access_token("acc_browser"));
        assert_eq!(credential.inner.refresh, "refresh-123");
        assert!(credential.inner.expires > chrono::Utc::now().timestamp_millis());
        assert_eq!(
            credential
                .inner
                .extra
                .get("account_id")
                .and_then(|v| v.as_str()),
            Some("acc_browser")
        );

        let events = interaction.events.lock().unwrap();
        let auth_url = events
            .iter()
            .find_map(|e| match e {
                AuthEvent::AuthUrl { url, .. } => Some(url.clone()),
                _ => None,
            })
            .expect("auth_url event");

        // Authorize URL carries the PKCE + Codex params.
        assert!(auth_url.starts_with("https://auth.openai.com/oauth/authorize?"));
        let query: std::collections::HashMap<String, String> = auth_url
            .split_once('?')
            .map(|(_, q)| {
                q.split('&')
                    .filter_map(|p| p.split_once('='))
                    .map(|(k, v)| (url_decode_compat(k), url_decode_compat(v)))
                    .collect()
            })
            .unwrap();
        assert_eq!(query.get("response_type").map(|s| s.as_str()), Some("code"));
        assert_eq!(
            query.get("client_id").map(|s| s.as_str()),
            Some("app_EMoamEEZ73f0CkXaXp7hrann")
        );
        assert_eq!(
            query.get("redirect_uri").map(|s| s.as_str()),
            Some("http://localhost:1455/auth/callback")
        );
        assert_eq!(
            query.get("code_challenge_method").map(|s| s.as_str()),
            Some("S256")
        );
        assert_eq!(
            query.get("id_token_add_organizations").map(|s| s.as_str()),
            Some("true")
        );
        assert_eq!(
            query.get("codex_cli_simplified_flow").map(|s| s.as_str()),
            Some("true")
        );
        assert_eq!(query.get("originator").map(|s| s.as_str()), Some("pi"));
        let challenge = query.get("code_challenge").expect("code_challenge");
        let state = query.get("state").expect("state");
        assert!(!state.is_empty());

        // The exchanged code_verifier must S256-hash to the code_challenge.
        let exchanged: &RecordedRequest = &token_server.recorded()[0];
        let verifier = exchanged.form_body().get("code_verifier").cloned().unwrap();
        use sha2::Digest;
        let mut hasher = sha2::Sha256::new();
        hasher.update(verifier.as_bytes());
        let expected_challenge = base64url_encode_test(&hasher.finalize());
        assert_eq!(
            &expected_challenge, challenge,
            "S256(code_verifier) == code_challenge"
        );

        // The manual code path wins: the pasted redirect URL's code is used.
        assert_eq!(
            exchanged.form_body().get("code").map(|s| s.as_str()),
            Some("auth-code-123")
        );
        token_server.shutdown();
    }

    #[tokio::test]
    async fn browser_login_receives_callback_from_local_server() {
        let port = free_port();
        let redirect = format!("http://127.0.0.1:{}/auth/callback", port);
        let token_server = MockServer::spawn(Arc::new(|_req: &RecordedRequest| {
            MockResponse::json(200, token_response_json("acc_callback"))
        }))
        .await;

        // The manual prompt never resolves (no gate release): the browser
        // callback wins the race.
        let interaction = TestInteraction::new("");
        let _gate = interaction.gate();
        let auth = GenericOAuthAuth::new(codex_auth_config(redirect.clone(), token_server.url()));

        // The login task races the local callback server against the prompt.
        let interaction_for_task = interaction.clone();
        let login_task = tokio::spawn(async move { auth.login(&interaction_for_task).await });

        // Wait until the authorize URL is published, then act as the browser.
        let auth_url = interaction.auth_url().await;
        let state = query_params(&auth_url)
            .get("state")
            .cloned()
            .expect("state");

        // Browser redirects to the local callback with the code + state.
        let callback_url = format!(
            "http://127.0.0.1:{}/auth/callback?code=cb-code-1&state={}",
            port,
            urlencode_compat(&state)
        );
        let response = reqwest::get(&callback_url)
            .await
            .expect("callback reaches local server");
        assert!(response.status().is_success());

        let credential = login_task.await.expect("task").expect("login succeeds");
        assert_eq!(credential.inner.access, mock_access_token("acc_callback"));
        assert_eq!(credential.inner.refresh, "refresh-123");
        // The exchanged code is the one delivered via the callback.
        assert_eq!(
            token_server.recorded()[0]
                .form_body()
                .get("code")
                .map(|s| s.as_str()),
            Some("cb-code-1")
        );
        token_server.shutdown();
    }

    #[tokio::test]
    async fn browser_login_uses_json_token_exchange_when_configured() {
        let token_server = MockServer::spawn(Arc::new(|req: &RecordedRequest| {
            assert_eq!(req.method, "POST");
            assert_eq!(req.header("content-type"), Some("application/json"));
            let body = req.json_body();
            assert_eq!(body["grant_type"], "authorization_code");
            assert_eq!(body["client_id"], "9d1c250a-e61b-44d9-88ed-5944d1962f5e");
            assert_eq!(
                body["redirect_uri"],
                "https://console.anthropic.com/oauth/code/callback"
            );
            assert!(body.get("code").is_some());
            assert!(body.get("code_verifier").is_some());
            assert!(body.get("state").is_some());
            MockResponse::json(200, token_response_json("acc_browser_json"))
        }))
        .await;

        let interaction = TestInteraction::new("");
        let gate = interaction.gate();
        let auth = GenericOAuthAuth::new(claude_auth_config(token_server.url()));

        let interaction_for_task = interaction.clone();
        let login_task = tokio::spawn(async move { auth.login(&interaction_for_task).await });

        let auth_url = interaction.auth_url().await;
        let real_state = query_params(&auth_url).get("state").cloned().unwrap();
        interaction.set_answer(format!("auth-code-json#{}", real_state));
        gate.send(()).expect("release manual prompt");

        let credential: OAuthCredential = login_task.await.expect("task").expect("login succeeds");
        assert_eq!(credential.inner.access, mock_access_token("acc_browser_json"));

        let exchanged: &RecordedRequest = &token_server.recorded()[0];
        assert_eq!(exchanged.json_body()["code"], "auth-code-json");
        assert_eq!(exchanged.json_body()["state"], real_state);
        token_server.shutdown();
    }

    #[tokio::test]
    async fn browser_login_falls_back_to_manual_code_on_state_mismatch() {
        let port = free_port();
        let redirect = format!("http://127.0.0.1:{}/auth/callback", port);
        let token_server = MockServer::spawn(Arc::new(|_req: &RecordedRequest| {
            MockResponse::json(200, token_response_json("acc_manual_fallback"))
        }))
        .await;

        let interaction = TestInteraction::new("manual-code-9");
        let gate = interaction.gate();
        let auth = GenericOAuthAuth::new(codex_auth_config(redirect.clone(), token_server.url()));

        let interaction_for_task = interaction.clone();
        let login_task = tokio::spawn(async move { auth.login(&interaction_for_task).await });

        // Browser hits the callback with a WRONG state → 400 + server waits.
        let auth_url = interaction.auth_url().await;
        let _real_state = query_params(&auth_url).get("state").cloned().unwrap();

        let wrong_state_url = format!(
            "http://127.0.0.1:{}/auth/callback?code=evil-code&state=wrong-state",
            port
        );
        let response = reqwest::get(&wrong_state_url)
            .await
            .expect("callback reaches local server");
        assert_eq!(response.status().as_u16(), 400);

        // The manual prompt answers with a bare code → login completes via it.
        gate.send(()).expect("release manual prompt");
        let credential = login_task.await.expect("task").expect("login succeeds");
        assert_eq!(
            credential.inner.access,
            mock_access_token("acc_manual_fallback")
        );
        assert_eq!(
            token_server.recorded()[0]
                .form_body()
                .get("code")
                .map(|s| s.as_str()),
            Some("manual-code-9")
        );
        token_server.shutdown();
    }

    #[tokio::test]
    async fn browser_login_surfaces_token_exchange_failure() {
        let token_server = MockServer::spawn(Arc::new(|_req: &RecordedRequest| {
            MockResponse::json(400, serde_json::json!({ "error": "invalid_grant" }))
        }))
        .await;

        let interaction = TestInteraction::new("code-from-user");
        let auth = GenericOAuthAuth::new(codex_auth_config(
            "http://localhost:1455/auth/callback".to_string(),
            token_server.url(),
        ));

        let err = auth
            .login(&interaction)
            .await
            .expect_err("exchange failure surfaces");
        assert!(
            err.to_string().contains("token exchange failed"),
            "err: {}",
            err
        );
        token_server.shutdown();
    }
}

fn base64url_encode_test(bytes: &[u8]) -> String {
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
    out.trim_end_matches('=').to_string()
}

fn url_decode_compat(s: &str) -> String {
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

fn urlencode_compat(s: &str) -> String {
    const HEX: &[u8] = b"0123456789ABCDEF";
    let mut out = String::new();
    for &b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => {
                out.push('%');
                out.push(HEX[(b >> 4) as usize] as char);
                out.push(HEX[(b & 0xf) as usize] as char);
            }
        }
    }
    out
}
