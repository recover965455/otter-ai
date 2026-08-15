//! Integration tests for otter-ai — mirrors the TypeScript @earendil-works/pi-ai test suite.
//!
//! Organized into modules matching the TS test file names:
//! - `faux_provider`  (faux-provider.test.ts)        — 23 tests, all run
//! - `validation`     (validation.test.ts)            — 9 tests, all run
//! - `models_runtime` (models-runtime.test.ts)        — core tests run, OAuth/auth-resolution tests #[ignore]
//! - `abort`          (abort.test.ts)                 — 41 tests, all #[ignore] (require real API credentials)

use std::sync::Arc;

use futures::StreamExt;
use otter_ai::auth::{
    ApiKeyCredential, AuthOperationOptions, Credential, CredentialStore, InMemoryCredentialStore,
    OAuthCredential, OAuthCredentials,
};
use otter_ai::auth::types::ModifyFn;
use otter_ai::providers::faux::*;
use otter_ai::types::*;
use otter_ai::*;
use serde_json::json;

// ---------------------------------------------------------------------------
// Helpers shared across modules
// ---------------------------------------------------------------------------

/// Collect all events from an `AssistantMessageEventStream` into a `Vec`.
async fn collect_events(stream: AssistantMessageEventStream) -> Vec<AssistantMessageEvent> {
    let mut events = Vec::new();
    let mut s = stream;
    while let Some(evt) = s.next().await {
        events.push(evt);
    }
    events
}

/// Return a short string label for an `AssistantMessageEvent` variant.
fn event_type(evt: &AssistantMessageEvent) -> &'static str {
    match evt {
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
    }
}

/// Complete using a specific model (not the default) via the registration's core.
async fn complete_with_model(
    registration: &FauxRegistration,
    model: &Model,
    context: Context,
    options: Option<SimpleStreamOptions>,
) -> AssistantMessage {
    let stream = registration.core.stream(model, context, options);
    stream.result_future().await
}

/// Build a minimal user context from a string.
fn user_context(text: &str) -> Context {
    Context {
        messages: vec![Message::User {
            content: vec![ContentBlock::Text { text: text.into() }],
            timestamp: 0,
        }],
        ..Default::default()
    }
}

/// Extract the usage from an assistant message (panics if not assistant).
fn usage_of(msg: &Message) -> Usage {
    match msg {
        Message::Assistant { usage, .. } => usage.clone(),
        _ => panic!("expected assistant message"),
    }
}

/// Build a test `Model` with sensible defaults (mirrors TS `testModel`).
fn test_model(provider: &str, id: &str) -> Model {
    Model {
        id: id.into(),
        provider_id: provider.into(),
        name: id.into(),
        api: "test-api".into(),
        max_input_tokens: None,
        max_output_tokens: Some(1000),
        supports_images: false,
        supports_audio: false,
        supports_video: false,
        supports_pdf: false,
        supports_tool_calling: false,
        supports_structured_output: false,
        supports_system_prompt: false,
        thinking: ModelThinkingLevel::None,
        reasoning: false,
        cost_rates: ModelCostRates::default(),
        context_window: Some(10000),
        default_temperature: None,
    }
}

// ===========================================================================
// faux-provider.test.ts — 23 tests
// ===========================================================================

mod faux_provider {
    use super::*;

    #[tokio::test]
    async fn registers_a_custom_provider_and_estimates_usage() {
        let registration = register_faux_provider(None);
        registration.set_responses(vec![FauxResponseStep::Message(
            faux_assistant_message("hello world", FauxAssistantMessageOptions::default()),
        )]);

        let context = Context {
            system_prompt: Some("Be concise.".into()),
            messages: vec![Message::User {
                content: vec![ContentBlock::Text { text: "hi there".into() }],
                timestamp: 0,
            }],
            ..Default::default()
        };

        let response = complete(&registration, context, None).await.unwrap();
        let usage = usage_of(&response);
        match &response {
            Message::Assistant { content, .. } => {
                assert_eq!(
                    content,
                    &vec![ContentBlock::Text {
                        text: "hello world".into()
                    }]
                );
            }
            _ => panic!("expected assistant message"),
        }
        assert!(usage.input > 0);
        assert!(usage.output > 0);
        assert_eq!(usage.total_tokens, usage.input + usage.output);
        assert_eq!(registration.state().call_count, 1);
    }

    #[tokio::test]
    async fn supports_helper_blocks_for_text_thinking_and_tool_calls() {
        let registration = register_faux_provider(None);
        registration.set_responses(vec![FauxResponseStep::Message(
            faux_assistant_message(
                vec![
                    faux_thinking("think"),
                    faux_tool_call(
                        "echo",
                        json!({ "text": "hi" }),
                        FauxToolCallOptions::default(),
                    ),
                    faux_text("done"),
                ],
                FauxAssistantMessageOptions {
                    stop_reason: Some("toolUse".into()),
                    ..Default::default()
                },
            ),
        )]);

        let response = complete(&registration, user_context("hi"), None)
            .await
            .unwrap();
        match &response {
            Message::Assistant {
                content, stop_reason, ..
            } => {
                assert_eq!(content.len(), 3);
                assert!(matches!(
                    &content[0],
                    ContentBlock::Thinking { thinking, .. } if thinking == "think"
                ));
                assert!(matches!(
                    &content[1],
                    ContentBlock::ToolCall { name, arguments, .. }
                        if name == "echo" && arguments == &json!({ "text": "hi" })
                ));
                assert!(matches!(
                    &content[2],
                    ContentBlock::Text { text } if text == "done"
                ));
                assert_eq!(stop_reason.as_deref(), Some("toolUse"));
            }
            _ => panic!("expected assistant message"),
        }
    }

    #[tokio::test]
    async fn supports_multiple_models_with_per_model_reasoning_and_model_aware_factories() {
        let registration = register_faux_provider(Some(RegisterFauxProviderOptions {
            models: vec![
                FauxModelDefinition {
                    id: "faux-fast".into(),
                    name: Some("Faux Fast".into()),
                    reasoning: false,
                    ..Default::default()
                },
                FauxModelDefinition {
                    id: "faux-thinker".into(),
                    name: Some("Faux Thinker".into()),
                    reasoning: true,
                    ..Default::default()
                },
            ],
            ..Default::default()
        }));

        let factory = || {
            FauxResponseStep::Factory(Arc::new(|_ctx, _opts, _state, model| {
                Box::pin(async move {
                    faux_assistant_message(
                        format!("{}:{}", model.id, model.reasoning),
                        FauxAssistantMessageOptions::default(),
                    )
                })
            }))
        };
        registration.set_responses(vec![factory(), factory()]);

        let ids: Vec<&str> = registration.models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["faux-fast", "faux-thinker"]);

        let default_model = registration.get_model(None).unwrap();
        assert_eq!(default_model.id, "faux-fast");

        assert!(!registration.get_model(Some("faux-fast")).unwrap().reasoning);
        assert!(registration
            .get_model(Some("faux-thinker"))
            .unwrap()
            .reasoning);

        let fast_model = registration.get_model(Some("faux-fast")).unwrap();
        let thinker_model = registration.get_model(Some("faux-thinker")).unwrap();

        let fast = complete_with_model(&registration, &fast_model, user_context("hi"), None).await;
        match &fast {
            Message::Assistant { content, .. } => {
                assert_eq!(
                    content,
                    &vec![ContentBlock::Text {
                        text: "faux-fast:false".into()
                    }]
                );
            }
            _ => panic!("expected assistant message"),
        }

        let thinker =
            complete_with_model(&registration, &thinker_model, user_context("hi"), None).await;
        match &thinker {
            Message::Assistant { content, .. } => {
                assert_eq!(
                    content,
                    &vec![ContentBlock::Text {
                        text: "faux-thinker:true".into()
                    }]
                );
            }
            _ => panic!("expected assistant message"),
        }
    }

    #[tokio::test]
    async fn rewrites_api_provider_and_model_on_returned_messages() {
        let registration = register_faux_provider(Some(RegisterFauxProviderOptions {
            api: Some("faux:test".into()),
            provider: Some("faux-provider".into()),
            models: vec![FauxModelDefinition {
                id: "faux-model".into(),
                ..Default::default()
            }],
            ..Default::default()
        }));
        registration.set_responses(vec![FauxResponseStep::Message(
            faux_assistant_message("hello", FauxAssistantMessageOptions::default()),
        )]);

        let response = complete(&registration, user_context("hi"), None)
            .await
            .unwrap();
        match &response {
            Message::Assistant {
                api,
                provider,
                model,
                ..
            } => {
                assert_eq!(api, "faux:test");
                assert_eq!(provider, "faux-provider");
                assert_eq!(model.as_deref(), Some("faux-model"));
            }
            _ => panic!("expected assistant message"),
        }
    }

    #[tokio::test]
    async fn consumes_queued_responses_in_order_and_errors_when_exhausted() {
        let registration = register_faux_provider(None);
        registration.set_responses(vec![
            FauxResponseStep::Message(faux_assistant_message(
                "first",
                FauxAssistantMessageOptions::default(),
            )),
            FauxResponseStep::Message(faux_assistant_message(
                "second",
                FauxAssistantMessageOptions::default(),
            )),
        ]);

        let context = user_context("hi");

        let first = complete(&registration, context.clone(), None).await.unwrap();
        match &first {
            Message::Assistant { content, .. } => {
                assert_eq!(
                    content,
                    &vec![ContentBlock::Text { text: "first".into() }]
                );
            }
            _ => panic!("expected assistant message"),
        }

        let second = complete(&registration, context.clone(), None).await.unwrap();
        match &second {
            Message::Assistant { content, .. } => {
                assert_eq!(
                    content,
                    &vec![ContentBlock::Text { text: "second".into() }]
                );
            }
            _ => panic!("expected assistant message"),
        }

        let exhausted = complete(&registration, context, None).await.unwrap();
        assert_eq!(exhausted.stop_reason(), Some("error"));
        match &exhausted {
            Message::Assistant { error_message, .. } => {
                assert_eq!(
                    error_message.as_deref(),
                    Some("No more faux responses queued")
                );
            }
            _ => panic!("expected assistant message"),
        }
        assert_eq!(registration.get_pending_response_count(), 0);
        assert_eq!(registration.state().call_count, 3);
    }

    #[tokio::test]
    async fn can_replace_and_append_queued_responses() {
        let registration = register_faux_provider(None);
        registration.set_responses(vec![FauxResponseStep::Message(
            faux_assistant_message("first", FauxAssistantMessageOptions::default()),
        )]);

        let context = user_context("hi");

        let r1 = complete(&registration, context.clone(), None).await.unwrap();
        match &r1 {
            Message::Assistant { content, .. } => {
                assert_eq!(
                    content,
                    &vec![ContentBlock::Text { text: "first".into() }]
                );
            }
            _ => panic!("expected assistant"),
        }
        assert_eq!(registration.get_pending_response_count(), 0);

        registration.set_responses(vec![FauxResponseStep::Message(
            faux_assistant_message("second", FauxAssistantMessageOptions::default()),
        )]);
        assert_eq!(registration.get_pending_response_count(), 1);

        let r2 = complete(&registration, context.clone(), None).await.unwrap();
        match &r2 {
            Message::Assistant { content, .. } => {
                assert_eq!(
                    content,
                    &vec![ContentBlock::Text { text: "second".into() }]
                );
            }
            _ => panic!("expected assistant"),
        }

        registration.append_responses(vec![
            FauxResponseStep::Message(faux_assistant_message(
                "third",
                FauxAssistantMessageOptions::default(),
            )),
            FauxResponseStep::Message(faux_assistant_message(
                "fourth",
                FauxAssistantMessageOptions::default(),
            )),
        ]);
        assert_eq!(registration.get_pending_response_count(), 2);

        let r3 = complete(&registration, context.clone(), None).await.unwrap();
        match &r3 {
            Message::Assistant { content, .. } => {
                assert_eq!(
                    content,
                    &vec![ContentBlock::Text { text: "third".into() }]
                );
            }
            _ => panic!("expected assistant"),
        }

        let r4 = complete(&registration, context, None).await.unwrap();
        match &r4 {
            Message::Assistant { content, .. } => {
                assert_eq!(
                    content,
                    &vec![ContentBlock::Text { text: "fourth".into() }]
                );
            }
            _ => panic!("expected assistant"),
        }
        assert_eq!(registration.get_pending_response_count(), 0);
    }

    #[tokio::test]
    async fn supports_async_response_factories() {
        let registration = register_faux_provider(None);
        registration.set_responses(vec![FauxResponseStep::Factory(Arc::new(
            |ctx, _opts, state, _model| {
                Box::pin(async move {
                    faux_assistant_message(
                        format!("{}:{}", ctx.messages.len(), state.call_count),
                        FauxAssistantMessageOptions::default(),
                    )
                })
            },
        ))]);

        let response = complete(&registration, user_context("hi"), None)
            .await
            .unwrap();
        match &response {
            Message::Assistant { content, .. } => {
                assert_eq!(
                    content,
                    &vec![ContentBlock::Text { text: "1:1".into() }]
                );
            }
            _ => panic!("expected assistant message"),
        }
    }

    #[tokio::test]
    async fn emits_an_error_when_a_response_factory_throws() {
        // In Rust, factories cannot "throw" — they return a Future<Output=AssistantMessage>.
        // The closest equivalent is a factory that returns an error AssistantMessage,
        // which the streaming layer surfaces as a terminal Error event.
        let registration = register_faux_provider(None);
        registration.set_responses(vec![FauxResponseStep::Factory(Arc::new(
            |_ctx, _opts, _state, _model| {
                Box::pin(async move {
                    faux_assistant_message(
                        Vec::<ContentBlock>::new(),
                        FauxAssistantMessageOptions {
                            stop_reason: Some("error".into()),
                            error_message: Some("boom".into()),
                            ..Default::default()
                        },
                    )
                })
            },
        ))]);

        // Use a single stream for both events and the final result so we only
        // consume one queued faux response. `EventStream` is `Clone` and the
        // clones share the same inner state, so draining events on one clone
        // still lets the other observe the terminal result.
        let s = stream(&registration, user_context("hi"), None);
        let s_for_result = s.clone();
        let events = collect_events(s).await;

        // The stream should end with an Error event.
        let terminal = events.last().expect("should have at least one event");
        assert!(matches!(terminal, AssistantMessageEvent::Error { .. }));
        if let AssistantMessageEvent::Error { reason, error } = terminal {
            assert_eq!(reason, "error");
            assert_eq!(error, "boom");
        }

        // The final result message should carry the error fields.
        let result = s_for_result.result_future().await;
        assert_eq!(result.stop_reason(), Some("error"));
        match &result {
            Message::Assistant { error_message, .. } => {
                assert_eq!(error_message.as_deref(), Some("boom"));
            }
            _ => panic!("expected assistant message"),
        }
    }

    #[tokio::test]
    async fn rejects_a_queued_response_without_a_terminal_stop_reason() {
        let registration = register_faux_provider(None);
        registration.set_responses(vec![FauxResponseStep::Message(
            faux_assistant_message(
                "partial",
                FauxAssistantMessageOptions {
                    stop_reason: Some("pending".into()),
                    ..Default::default()
                },
            ),
        )]);

        let events =
            collect_events(stream(&registration, user_context("hi"), None)).await;

        // No "done" event should be emitted.
        assert!(
            !events
                .iter()
                .any(|e| event_type(e) == "done"),
            "should not contain a done event"
        );

        // The terminal event should be an error.
        let terminal = events.last().expect("should have events");
        assert_eq!(event_type(terminal), "error");
        if let AssistantMessageEvent::Error { reason, error } = terminal {
            assert_eq!(reason, "error");
            assert_eq!(error, "Faux response ended without a stop reason");
        }
    }

    #[tokio::test]
    async fn estimates_prompt_and_output_tokens_from_serialized_context() {
        let registration = register_faux_provider(None);
        registration.set_responses(vec![FauxResponseStep::Message(
            faux_assistant_message("done", FauxAssistantMessageOptions::default()),
        )]);

        let tool = Tool {
            name: "echo".into(),
            description: Some("Echo back text".into()),
            parameters: json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"]
            }),
        };

        let context = Context {
            system_prompt: Some("sys".into()),
            messages: vec![
                Message::User {
                    content: vec![
                        ContentBlock::Text { text: "hello".into() },
                        ContentBlock::Image(ImageContent {
                            content_type: "image".into(),
                            data: "abcd".into(),
                            mime_type: Some("image/png".into()),
                        }),
                    ],
                    timestamp: 1,
                },
                faux_assistant_message("prior", FauxAssistantMessageOptions::default()),
                Message::ToolResult {
                    tool_call_id: "tool-1".into(),
                    tool_name: "echo".into(),
                    content: vec![ContentBlock::Text { text: "tool out".into() }],
                    is_error: false,
                    timestamp: 2,
                },
            ],
            tools: vec![tool.clone()],
            ..Default::default()
        };

        let response = complete(&registration, context.clone(), None)
            .await
            .unwrap();
        let usage = usage_of(&response);

        // Reconstruct the same prompt serialization the faux provider uses.
        let prompt_text = otter_ai::providers::faux::serialize_context(&context);
        let expected_prompt_tokens = ((prompt_text.len() as f64) / 4.0).ceil() as u64;
        let expected_output_tokens = ((4.0_f64) / 4.0).ceil() as u64; // "done" = 4 chars

        assert_eq!(usage.input, expected_prompt_tokens);
        assert_eq!(usage.output, expected_output_tokens);
        assert_eq!(usage.cache_read, 0);
        assert_eq!(usage.cache_write, 0);
        assert_eq!(
            usage.total_tokens,
            expected_prompt_tokens + expected_output_tokens
        );
    }

    #[tokio::test]
    async fn does_not_share_cache_across_sessions_or_requests_without_session_id() {
        let registration = register_faux_provider(None);
        registration.set_responses(vec![
            FauxResponseStep::Message(faux_assistant_message(
                "first",
                FauxAssistantMessageOptions::default(),
            )),
            FauxResponseStep::Message(faux_assistant_message(
                "second",
                FauxAssistantMessageOptions::default(),
            )),
            FauxResponseStep::Message(faux_assistant_message(
                "third",
                FauxAssistantMessageOptions::default(),
            )),
        ]);

        let mut context = user_context("hello");

        let opts1 = SimpleStreamOptions {
            session_id: Some("session-1".into()),
            cache_retention: Some(CacheRetention::Short),
            ..Default::default()
        };
        let first = complete(&registration, context.clone(), Some(opts1))
            .await
            .unwrap();
        assert!(usage_of(&first).cache_write > 0);

        context.messages.push(first);
        context.messages.push(Message::User {
            content: vec![ContentBlock::Text { text: "follow up".into() }],
            timestamp: 1,
        });

        let opts2 = SimpleStreamOptions {
            session_id: Some("session-2".into()),
            cache_retention: Some(CacheRetention::Short),
            ..Default::default()
        };
        let second = complete(&registration, context.clone(), Some(opts2))
            .await
            .unwrap();
        let su = usage_of(&second);
        assert_eq!(su.cache_read, 0);
        assert!(su.cache_write > 0);

        let third = complete(&registration, context, None).await.unwrap();
        let tu = usage_of(&third);
        assert_eq!(tu.cache_read, 0);
        assert_eq!(tu.cache_write, 0);
    }

    #[tokio::test]
    async fn simulates_prompt_caching_per_session_id() {
        let registration = register_faux_provider(None);
        registration.set_responses(vec![
            FauxResponseStep::Message(faux_assistant_message(
                "first",
                FauxAssistantMessageOptions::default(),
            )),
            FauxResponseStep::Message(faux_assistant_message(
                "second",
                FauxAssistantMessageOptions::default(),
            )),
        ]);

        let mut context = Context {
            system_prompt: Some("Be concise.".into()),
            messages: vec![Message::User {
                content: vec![ContentBlock::Text { text: "hello".into() }],
                timestamp: 0,
            }],
            ..Default::default()
        };

        let opts = SimpleStreamOptions {
            session_id: Some("session-1".into()),
            cache_retention: Some(CacheRetention::Short),
            ..Default::default()
        };

        let first = complete(&registration, context.clone(), Some(opts.clone()))
            .await
            .unwrap();
        let fu = usage_of(&first);
        assert_eq!(fu.cache_read, 0);
        assert!(fu.cache_write > 0);

        context.messages.push(first);
        context.messages.push(Message::User {
            content: vec![ContentBlock::Text { text: "follow up".into() }],
            timestamp: 1,
        });

        let second = complete(&registration, context.clone(), Some(opts))
            .await
            .unwrap();
        let su = usage_of(&second);
        assert!(su.cache_read > 0);
        assert!(su.input + su.cache_read > su.input);
    }

    #[tokio::test]
    async fn does_not_simulate_caching_when_cache_retention_is_none() {
        let registration = register_faux_provider(None);
        registration.set_responses(vec![
            FauxResponseStep::Message(faux_assistant_message(
                "first",
                FauxAssistantMessageOptions::default(),
            )),
            FauxResponseStep::Message(faux_assistant_message(
                "second",
                FauxAssistantMessageOptions::default(),
            )),
        ]);

        let mut context = user_context("hello");

        let opts = SimpleStreamOptions {
            session_id: Some("session-1".into()),
            cache_retention: Some(CacheRetention::None),
            ..Default::default()
        };
        let _ = complete(&registration, context.clone(), Some(opts.clone()))
            .await
            .unwrap();

        context.messages.push(faux_assistant_message(
            "first",
            FauxAssistantMessageOptions::default(),
        ));
        context.messages.push(Message::User {
            content: vec![ContentBlock::Text { text: "follow up".into() }],
            timestamp: 1,
        });

        let second = complete(&registration, context, Some(opts)).await.unwrap();
        let su = usage_of(&second);
        assert_eq!(su.cache_read, 0);
        assert_eq!(su.cache_write, 0);
    }

    #[tokio::test]
    async fn streams_thinking_text_and_partial_tool_call_deltas() {
        let registration = register_faux_provider(None);
        registration.set_responses(vec![FauxResponseStep::Message(
            faux_assistant_message(
                vec![
                    faux_thinking("thinking text"),
                    faux_text("answer text"),
                    faux_tool_call(
                        "echo",
                        json!({ "text": "hi", "count": 12 }),
                        FauxToolCallOptions {
                            id: Some("tool-1".into()),
                        },
                    ),
                ],
                FauxAssistantMessageOptions {
                    stop_reason: Some("toolUse".into()),
                    ..Default::default()
                },
            ),
        )]);

        let mut event_types = Vec::new();
        let mut toolcall_deltas: Vec<String> = Vec::new();
        let mut s = stream(&registration, user_context("hi"), None);
        while let Some(evt) = s.next().await {
            let t = event_type(&evt);
            event_types.push(t);
            if let AssistantMessageEvent::ToolcallDelta { delta, .. } = &evt {
                toolcall_deltas.push(delta.clone());
            }
        }

        assert!(event_types.contains(&"thinking_start"));
        assert!(event_types.contains(&"thinking_delta"));
        assert!(event_types.contains(&"text_start"));
        assert!(event_types.contains(&"text_delta"));
        assert!(event_types.contains(&"toolcall_start"));
        assert!(event_types.contains(&"toolcall_delta"));
        assert!(event_types.contains(&"toolcall_end"));
        assert!(toolcall_deltas.len() > 1);

        let joined = toolcall_deltas.join("");
        let parsed: serde_json::Value = serde_json::from_str(&joined).unwrap();
        assert_eq!(parsed, json!({ "text": "hi", "count": 12 }));
    }

    #[tokio::test]
    async fn streams_an_exact_event_order_for_fixed_size_chunks() {
        let registration = register_faux_provider(Some(RegisterFauxProviderOptions {
            token_size: Some((1, 1)),
            ..Default::default()
        }));
        registration.set_responses(vec![FauxResponseStep::Message(
            faux_assistant_message(
                vec![
                    faux_thinking("go"),
                    faux_text("ok"),
                    faux_tool_call(
                        "echo",
                        json!({}),
                        FauxToolCallOptions {
                            id: Some("tool-1".into()),
                        },
                    ),
                ],
                FauxAssistantMessageOptions {
                    stop_reason: Some("toolUse".into()),
                    ..Default::default()
                },
            ),
        )]);

        let events =
            collect_events(stream(&registration, user_context("hi"), None)).await;
        let types: Vec<&str> = events.iter().map(event_type).collect();

        // Rust emits a "usage" event before "done" (the TS version does not).
        assert_eq!(
            types,
            vec![
                "start",
                "thinking_start",
                "thinking_delta",
                "thinking_end",
                "text_start",
                "text_delta",
                "text_end",
                "toolcall_start",
                "toolcall_delta",
                "toolcall_end",
                "usage",
                "done",
            ]
        );
    }

    #[tokio::test]
    async fn streams_multiple_tool_calls_in_one_message() {
        let registration = register_faux_provider(None);
        registration.set_responses(vec![FauxResponseStep::Message(
            faux_assistant_message(
                vec![
                    faux_tool_call(
                        "echo",
                        json!({ "text": "one" }),
                        FauxToolCallOptions {
                            id: Some("tool-1".into()),
                        },
                    ),
                    faux_tool_call(
                        "echo",
                        json!({ "text": "two" }),
                        FauxToolCallOptions {
                            id: Some("tool-2".into()),
                        },
                    ),
                ],
                FauxAssistantMessageOptions {
                    stop_reason: Some("toolUse".into()),
                    ..Default::default()
                },
            ),
        )]);

        let events =
            collect_events(stream(&registration, user_context("hi"), None)).await;

        let start_count = events
            .iter()
            .filter(|e| event_type(e) == "toolcall_start")
            .count();
        let end_count = events
            .iter()
            .filter(|e| event_type(e) == "toolcall_end")
            .count();
        assert_eq!(start_count, 2);
        assert_eq!(end_count, 2);
    }

    #[tokio::test]
    async fn streams_an_explicit_assistant_error_message_as_a_terminal_error() {
        let registration = register_faux_provider(Some(RegisterFauxProviderOptions {
            token_size: Some((2, 2)),
            ..Default::default()
        }));
        registration.set_responses(vec![FauxResponseStep::Message(
            faux_assistant_message(
                "partial",
                FauxAssistantMessageOptions {
                    stop_reason: Some("error".into()),
                    error_message: Some("upstream failed".into()),
                    ..Default::default()
                },
            ),
        )]);

        let events =
            collect_events(stream(&registration, user_context("hi"), None)).await;
        let types: Vec<&str> = events.iter().map(event_type).collect();

        assert_eq!(
            types,
            vec!["start", "text_start", "text_delta", "text_end", "error"]
        );

        let terminal = events.last().unwrap();
        if let AssistantMessageEvent::Error { reason, error } = terminal {
            assert_eq!(reason, "error");
            assert_eq!(error, "upstream failed");
        }
    }

    #[tokio::test]
    async fn streams_an_explicit_assistant_aborted_message_as_a_terminal_error() {
        let registration = register_faux_provider(Some(RegisterFauxProviderOptions {
            token_size: Some((2, 2)),
            ..Default::default()
        }));
        registration.set_responses(vec![FauxResponseStep::Message(
            faux_assistant_message(
                "partial",
                FauxAssistantMessageOptions {
                    stop_reason: Some("aborted".into()),
                    error_message: Some("Request was aborted".into()),
                    ..Default::default()
                },
            ),
        )]);

        let events =
            collect_events(stream(&registration, user_context("hi"), None)).await;
        let types: Vec<&str> = events.iter().map(event_type).collect();

        assert_eq!(
            types,
            vec!["start", "text_start", "text_delta", "text_end", "error"]
        );

        let terminal = events.last().unwrap();
        if let AssistantMessageEvent::Error { reason, error } = terminal {
            assert_eq!(reason, "aborted");
            assert_eq!(error, "Request was aborted");
        }
    }

    #[tokio::test]
    async fn supports_aborting_before_the_first_chunk() {
        let registration = register_faux_provider(Some(RegisterFauxProviderOptions {
            tokens_per_second: Some(50),
            token_size: Some((3, 3)),
            ..Default::default()
        }));
        registration.set_responses(vec![FauxResponseStep::Message(
            faux_assistant_message(
                "abcdefghijklmnopqrstuvwxyz",
                FauxAssistantMessageOptions::default(),
            ),
        )]);

        let token = CancellationToken::new();
        token.cancel();

        let events = collect_events(stream(
            &registration,
            user_context("hi"),
            Some(SimpleStreamOptions {
                signal: Some(token),
                ..Default::default()
            }),
        ))
        .await;

        assert_eq!(events.len(), 1);
        assert_eq!(event_type(&events[0]), "error");
        if let AssistantMessageEvent::Error { reason, error } = &events[0] {
            assert_eq!(reason, "aborted");
            // error_str returns the error_message or stop_reason
            assert!(error.contains("aborted") || error.contains("Request was aborted"));
        }
    }

    #[tokio::test]
    async fn supports_aborting_mid_text_stream_when_paced() {
        let registration = register_faux_provider(Some(RegisterFauxProviderOptions {
            tokens_per_second: Some(100),
            token_size: Some((3, 3)),
            ..Default::default()
        }));
        registration.set_responses(vec![FauxResponseStep::Message(
            faux_assistant_message(
                "abcdefghijklmnopqrstuvwxyz",
                FauxAssistantMessageOptions::default(),
            ),
        )]);

        let token = CancellationToken::new();
        let mut event_types = Vec::new();
        let mut text_delta_count = 0u32;

        let mut s = stream(
            &registration,
            user_context("hi"),
            Some(SimpleStreamOptions {
                signal: Some(token.clone()),
                ..Default::default()
            }),
        );
        while let Some(evt) = s.next().await {
            let t = event_type(&evt);
            event_types.push(t);
            if t == "text_delta" {
                text_delta_count += 1;
                token.cancel();
            }
        }

        assert_eq!(text_delta_count, 1);
        assert!(event_types.contains(&"text_start"));
        assert!(event_types.contains(&"text_delta"));
        assert!(event_types.contains(&"error"));
        assert!(!event_types.contains(&"text_end"));
    }

    #[tokio::test]
    async fn supports_aborting_mid_thinking_stream_when_paced() {
        let registration = register_faux_provider(Some(RegisterFauxProviderOptions {
            tokens_per_second: Some(100),
            token_size: Some((3, 3)),
            ..Default::default()
        }));
        registration.set_responses(vec![FauxResponseStep::Message(
            faux_assistant_message(
                vec![faux_thinking("abcdefghijklmnopqrstuvwxyz")],
                FauxAssistantMessageOptions::default(),
            ),
        )]);

        let token = CancellationToken::new();
        let mut event_types = Vec::new();
        let mut thinking_delta_count = 0u32;

        let mut s = stream(
            &registration,
            user_context("hi"),
            Some(SimpleStreamOptions {
                signal: Some(token.clone()),
                ..Default::default()
            }),
        );
        while let Some(evt) = s.next().await {
            let t = event_type(&evt);
            event_types.push(t);
            if t == "thinking_delta" {
                thinking_delta_count += 1;
                token.cancel();
            }
        }

        assert_eq!(thinking_delta_count, 1);
        assert!(event_types.contains(&"thinking_start"));
        assert!(event_types.contains(&"thinking_delta"));
        assert!(event_types.contains(&"error"));
        assert!(!event_types.contains(&"thinking_end"));
    }

    #[tokio::test]
    async fn supports_aborting_mid_toolcall_stream_when_paced() {
        let registration = register_faux_provider(Some(RegisterFauxProviderOptions {
            tokens_per_second: Some(100),
            token_size: Some((3, 3)),
            ..Default::default()
        }));
        registration.set_responses(vec![FauxResponseStep::Message(
            faux_assistant_message(
                vec![faux_tool_call(
                    "echo",
                    json!({ "text": "abcdefghijklmnopqrstuvwxyz", "count": 123456789 }),
                    FauxToolCallOptions {
                        id: Some("tool-1".into()),
                    },
                )],
                FauxAssistantMessageOptions {
                    stop_reason: Some("toolUse".into()),
                    ..Default::default()
                },
            ),
        )]);

        let token = CancellationToken::new();
        let mut event_types = Vec::new();
        let mut toolcall_delta_count = 0u32;

        let mut s = stream(
            &registration,
            user_context("hi"),
            Some(SimpleStreamOptions {
                signal: Some(token.clone()),
                ..Default::default()
            }),
        );
        while let Some(evt) = s.next().await {
            let t = event_type(&evt);
            event_types.push(t);
            if t == "toolcall_delta" {
                toolcall_delta_count += 1;
                token.cancel();
            }
        }

        assert_eq!(toolcall_delta_count, 1);
        assert!(event_types.contains(&"toolcall_start"));
        assert!(event_types.contains(&"toolcall_delta"));
        assert!(event_types.contains(&"error"));
        assert!(!event_types.contains(&"toolcall_end"));
    }

    #[tokio::test]
    async fn unregisters_the_provider() {
        // In Rust standalone mode, unregister() clears the response queue.
        // Subsequent complete() calls return an error message with
        // stop_reason="error" and error_message="No more faux responses queued".
        let registration = register_faux_provider(None);
        registration.set_responses(vec![FauxResponseStep::Message(
            faux_assistant_message("hello", FauxAssistantMessageOptions::default()),
        )]);
        registration.unregister();

        let result = complete(&registration, user_context("hi"), None).await;
        let msg = result.unwrap();
        assert_eq!(msg.stop_reason(), Some("error"));
        match &msg {
            Message::Assistant { error_message, .. } => {
                assert_eq!(
                    error_message.as_deref(),
                    Some("No more faux responses queued")
                );
            }
            _ => panic!("expected assistant message"),
        }
    }
}

// ===========================================================================
// validation.test.ts — 9 tests (all #[ignore])
// ===========================================================================

mod validation {
    use otter_ai::utils::validation::validate_tool_arguments;
    use otter_ai::types::Tool;
    use serde_json::json;

    fn make_tool(parameters: serde_json::Value) -> Tool {
        Tool {
            name: "echo".into(),
            description: Some("Echo tool".into()),
            parameters,
        }
    }

    fn make_tool_with_value_schema(schema: serde_json::Value, value: serde_json::Value) -> (Tool, serde_json::Value) {
        let tool = make_tool(json!({
            "type": "object",
            "properties": { "value": schema },
            "required": ["value"],
        }));
        let args = json!({ "value": value });
        (tool, args)
    }

    #[test]
    fn still_validates_when_function_constructor_is_unavailable() {
        // Rust has no Function constructor concept; this test just verifies
        // that coercion works without any code generation.
        let tool = make_tool(json!({
            "type": "object",
            "properties": { "count": { "type": "number" } },
            "required": ["count"],
        }));
        let args = json!({ "count": "42" });
        let result = validate_tool_arguments(&tool, &args).unwrap();
        assert_eq!(result, json!({ "count": 42 }));
    }

    #[test]
    fn coerces_serialized_plain_json_schemas_with_ajv_compatible_primitive_rules() {
        let cases: Vec<(serde_json::Value, serde_json::Value, serde_json::Value)> = vec![
            (json!({"type":"number"}), json!("42"), json!(42)),
            (json!({"type":"number"}), json!(true), json!(1)),
            (json!({"type":"number"}), json!(null), json!(0)),
            (json!({"type":"integer"}), json!("42"), json!(42)),
            (json!({"type":"boolean"}), json!("true"), json!(true)),
            (json!({"type":"boolean"}), json!("false"), json!(false)),
            (json!({"type":"boolean"}), json!(1), json!(true)),
            (json!({"type":"boolean"}), json!(0), json!(false)),
            (json!({"type":"string"}), json!(null), json!("")),
            (json!({"type":"string"}), json!(true), json!("true")),
            (json!({"type":"null"}), json!(""), json!(null)),
            (json!({"type":"null"}), json!(0), json!(null)),
            (json!({"type":"null"}), json!(false), json!(null)),
            (json!({"type":["number","string"]}), json!("1"), json!("1")),
            (json!({"type":["boolean","number"]}), json!("1"), json!(1)),
        ];

        for (schema, input, expected) in cases {
            let (tool, args) = make_tool_with_value_schema(schema.clone(), input.clone());
            let result = validate_tool_arguments(&tool, &args).unwrap();
            assert_eq!(result, json!({ "value": expected }), "schema: {:?}, input: {:?}", schema, input);
        }
    }

    #[test]
    fn treats_null_as_omission_for_optional_non_nullable_properties() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "offset": { "type": "number" },
                "nullable": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                "metadata": { "type": "object", "properties": { "enabled": { "type": "boolean" } } }
            },
            "required": ["path"],
        }));
        let args = json!({
            "path": "file.txt",
            "offset": null,
            "nullable": null,
            "metadata": { "enabled": null }
        });
        let result = validate_tool_arguments(&tool, &args).unwrap();
        assert_eq!(result, json!({
            "path": "file.txt",
            "nullable": null,
            "metadata": {}
        }));
    }

    #[test]
    fn preserves_optional_nulls_whose_referenced_schema_is_nullable() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {
                "value": { "$ref": "#/$defs/value" }
            },
            "$defs": {
                "value": { "anyOf": [{ "type": "number" }, { "type": "null" }] }
            }
        }));
        let args = json!({ "value": null });
        let result = validate_tool_arguments(&tool, &args).unwrap();
        assert_eq!(result, json!({ "value": null }));
    }

    #[test]
    fn preserves_a_value_that_already_matches_a_nullable_union_arm() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {
                "value": { "anyOf": [{ "type": "number" }, { "type": "null" }] }
            },
            "required": ["value"],
        }));
        let args = json!({ "value": null });
        let result = validate_tool_arguments(&tool, &args).unwrap();
        assert_eq!(result, json!({ "value": null }));
    }

    #[test]
    fn preserves_a_value_that_already_matches_a_oneof_nullable_union_arm() {
        let (tool, args) = make_tool_with_value_schema(
            json!({ "oneOf": [{ "type": "number" }, { "type": "null" }] }),
            json!(null),
        );
        let result = validate_tool_arguments(&tool, &args).unwrap();
        assert_eq!(result, json!({ "value": null }));
    }

    #[test]
    fn still_coerces_nullable_unions_when_the_original_value_does_not_match_any_arm() {
        let (tool, args) = make_tool_with_value_schema(
            json!({ "anyOf": [{ "type": "number" }, { "type": "null" }] }),
            json!("42"),
        );
        let result = validate_tool_arguments(&tool, &args).unwrap();
        assert_eq!(result, json!({ "value": 42 }));
    }

    #[test]
    fn accepts_null_for_nullable_array_schemas_with_items() {
        let (tool, args) = make_tool_with_value_schema(
            json!({ "type": ["array", "null"], "items": { "type": "string" } }),
            json!(null),
        );
        let result = validate_tool_arguments(&tool, &args).unwrap();
        assert_eq!(result, json!({ "value": null }));
    }

    #[test]
    fn rejects_invalid_coercions_for_serialized_plain_json_schemas() {
        let failing_cases: Vec<(serde_json::Value, serde_json::Value)> = vec![
            (json!({"type":"boolean"}), json!("1")),
            (json!({"type":"boolean"}), json!("0")),
            (json!({"type":"null"}), json!("null")),
            (json!({"type":"integer"}), json!("42.1")),
        ];

        for (schema, input) in failing_cases {
            let (tool, args) = make_tool_with_value_schema(schema.clone(), input.clone());
            let result = validate_tool_arguments(&tool, &args);
            assert!(result.is_err(), "should reject: schema={:?}, input={:?}", schema, input);
            assert!(result.unwrap_err().contains("Validation failed"));
        }
    }
}

// ===========================================================================
// models-runtime.test.ts — core tests run, OAuth/credential tests #[ignore]
// ===========================================================================

mod models_runtime {
    use super::*;
    use otter_ai::utils::validation::calculate_cost;

    #[test]
    fn applies_request_wide_pricing_tiers_above_the_configured_input_threshold() {
        let mut model = test_model("openai", "gpt-5.6-sol");
        model.cost_rates = ModelCostRates {
            input_per_million: Some(5.0),
            output_per_million: Some(30.0),
            input_cache_read_per_million: Some(0.5),
            input_cache_write_per_million: Some(6.25),
            tiers: vec![CostTier {
                input_tokens_above: 272_000,
                input_per_million: 10.0,
                output_per_million: 45.0,
                cache_read_per_million: 1.0,
                cache_write_per_million: 12.5,
            }],
        };

        let create_usage = |cache_write: u64| Usage {
            input: 200_000,
            output: 100_000,
            cache_read: 72_000,
            cache_write,
            total_tokens: 372_000 + cache_write,
            cost: UsageCost::default(),
        };

        // non_output_total = 200k + 72k + 0 = 272k, NOT > 272k → base rates
        let short = calculate_cost(&model, &create_usage(0));
        assert!((short.input - 1.0).abs() < 1e-9);
        assert!((short.output - 3.0).abs() < 1e-9);
        assert!((short.cache_read - 0.036).abs() < 1e-9);
        assert!((short.cache_write - 0.0).abs() < 1e-9);

        // non_output_total = 200k + 72k + 1 = 272001, > 272k → tier rates
        let long = calculate_cost(&model, &create_usage(1));
        assert!((long.input - 2.0).abs() < 1e-9);
        assert!((long.output - 4.5).abs() < 1e-9);
        assert!((long.cache_read - 0.072).abs() < 1e-9);
        assert!((long.cache_write - 0.0000125).abs() < 1e-9);
    }

    #[test]
    fn calculates_basic_usage_cost_with_flat_rates() {
        let mut model = test_model("test", "m");
        model.cost_rates = ModelCostRates {
            input_per_million: Some(1.0),
            output_per_million: Some(2.0),
            input_cache_read_per_million: Some(0.5),
            input_cache_write_per_million: Some(0.25),
            ..Default::default()
        };

        // 1M each → 1.0 + 2.0 + 0.5 + 0.25 = 3.75
        let cost = calculate_usage_cost(1_000_000, 1_000_000, 1_000_000, 1_000_000, &model);
        assert!((cost.input - 1.0).abs() < 1e-9);
        assert!((cost.output - 2.0).abs() < 1e-9);
        assert!((cost.cache_read - 0.5).abs() < 1e-9);
        assert!((cost.cache_write - 0.25).abs() < 1e-9);
        assert!((cost.total - 3.75).abs() < 1e-9);
    }

    #[test]
    fn calculates_usage_cost_with_missing_rates_returns_zero() {
        let model = test_model("test", "m");
        let cost = calculate_usage_cost(1_000_000, 1_000_000, 1_000_000, 1_000_000, &model);
        assert_eq!(cost.total, 0.0);
    }

    #[tokio::test]
    async fn registers_replaces_and_deletes_providers() {
        let models = create_models();

        let h1 = register_faux_provider(Some(RegisterFauxProviderOptions {
            provider: Some("p1".into()),
            ..Default::default()
        }));
        let h2 = register_faux_provider(Some(RegisterFauxProviderOptions {
            provider: Some("p2".into()),
            ..Default::default()
        }));
        models.set_provider_arc(h1.provider.clone());
        models.set_provider_arc(h2.provider.clone());

        let provider_ids: Vec<String> =
            models.list_providers().into_iter().map(|(id, _)| id).collect();
        assert!(provider_ids.contains(&"p1".to_string()));
        assert!(provider_ids.contains(&"p2".to_string()));

        // Replace p1 with a new provider instance.
        let replacement = register_faux_provider(Some(RegisterFauxProviderOptions {
            provider: Some("p1".into()),
            ..Default::default()
        }));
        models.set_provider_arc(replacement.provider.clone());
        assert!(models.get_provider("p1").is_some());
        assert_eq!(models.list_providers().len(), 2);

        // Delete p1
        models.delete_provider("p1");
        assert!(models.get_provider("p1").is_none());
        assert_eq!(models.list_providers().len(), 1);

        // Clear all providers
        models.clear_providers();
        assert!(models.list_providers().is_empty());
    }

    #[tokio::test]
    async fn lists_and_finds_models_per_provider() {
        let models = create_models();

        let h1 = register_faux_provider(Some(RegisterFauxProviderOptions {
            provider: Some("p1".into()),
            models: vec![
                FauxModelDefinition {
                    id: "m1".into(),
                    ..Default::default()
                },
                FauxModelDefinition {
                    id: "m2".into(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }));
        let h2 = register_faux_provider(Some(RegisterFauxProviderOptions {
            provider: Some("p2".into()),
            models: vec![FauxModelDefinition {
                id: "m3".into(),
                ..Default::default()
            }],
            ..Default::default()
        }));
        models.set_provider_arc(h1.provider.clone());
        models.set_provider_arc(h2.provider.clone());
        models.refresh_all(false, true).await.unwrap();

        let all_ids: Vec<String> =
            models.list_models(None).iter().map(|m| m.id.clone()).collect();
        assert!(all_ids.contains(&"m1".to_string()));
        assert!(all_ids.contains(&"m2".to_string()));
        assert!(all_ids.contains(&"m3".to_string()));

        let p1_ids: Vec<String> = models
            .list_models(Some("p1"))
            .iter()
            .map(|m| m.id.clone())
            .collect();
        assert!(p1_ids.contains(&"m1".to_string()));
        assert!(p1_ids.contains(&"m2".to_string()));

        assert!(models.list_models(Some("nope")).is_empty());

        assert_eq!(
            models.get_model("p2", "m3").map(|m| m.id),
            Some("m3".to_string())
        );
        assert!(models.get_model("p2", "missing").is_none());
    }

    #[tokio::test]
    async fn produces_an_error_stream_for_unknown_providers_instead_of_throwing() {
        let models = create_models();
        let model = test_model("ghost", "model-a");
        let ctx = user_context("hi");

        let mut stream = models.stream(&model, ctx, SimpleStreamOptions::default());
        let mut events = Vec::new();
        while let Some(evt) = stream.next().await {
            events.push(evt);
        }

        // Should produce an error event mentioning the unknown provider.
        let has_error = events.iter().any(|e| match e {
            AssistantMessageEvent::Error { error, .. } => error.contains("ghost"),
            _ => false,
        });
        assert!(has_error, "should emit an error event for unknown provider");
    }

    #[tokio::test]
    async fn streams_through_the_provider() {
        let models = create_models();
        let handle = register_faux_provider(Some(RegisterFauxProviderOptions {
            provider: Some("p1".into()),
            models: vec![FauxModelDefinition {
                id: "model-a".into(),
                ..Default::default()
            }],
            ..Default::default()
        }));
        handle.set_responses(vec![FauxResponseStep::Message(
            faux_assistant_message("ok", FauxAssistantMessageOptions::default()),
        )]);
        models.set_provider_arc(handle.provider.clone());
        models.refresh_all(false, true).await.unwrap();
        let model = models.get_model("p1", "model-a").expect("model-a should exist");

        let ctx = user_context("hi");
        let mut stream = models.stream(&model, ctx, SimpleStreamOptions::default());
        let mut types = Vec::new();
        while let Some(evt) = stream.next().await {
            types.push(event_type(&evt));
        }

        assert!(types.contains(&"start"));
        assert!(types.contains(&"done"));
    }

    // -----------------------------------------------------------------------
    // Ignored tests: complex OAuth / credential / dynamic-model flows that
    // require enhanced auth infrastructure not yet available in the Rust crate.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn enumerates_credential_metadata_without_exposing_secrets() {
        let credentials = InMemoryCredentialStore::new();

        // api-provider: api_key credential carrying a secret
        let api_modify: ModifyFn = Box::new(|_| {
            Box::pin(async move {
                Ok(Some(Credential::ApiKey(ApiKeyCredential {
                    r#type: "api_key".into(),
                    key: Some("secret".into()),
                    env: None,
                })))
            })
        });
        credentials
            .modify_fn("api-provider", api_modify, AuthOperationOptions::default())
            .await
            .unwrap();

        // oauth-provider: oauth credential
        let expires = chrono::Utc::now().timestamp_millis() + 60_000;
        let oauth_modify: ModifyFn = Box::new(move |_| {
            Box::pin(async move {
                Ok(Some(Credential::OAuth(OAuthCredential {
                    r#type: "oauth".into(),
                    inner: OAuthCredentials {
                        refresh: "refresh".into(),
                        access: "access".into(),
                        expires,
                        extra: Default::default(),
                    },
                })))
            })
        });
        credentials
            .modify_fn("oauth-provider", oauth_modify, AuthOperationOptions::default())
            .await
            .unwrap();

        let list = credentials
            .list(AuthOperationOptions::default())
            .await
            .unwrap();
        // list() returns metadata without exposing secrets
        let mut entries: Vec<(String, String)> = list
            .into_iter()
            .map(|c| (c.provider_id, c.r#type))
            .collect();
        entries.sort();
        assert_eq!(
            entries,
            vec![
                ("api-provider".to_string(), "api_key".to_string()),
                ("oauth-provider".to_string(), "oauth".to_string()),
            ]
        );
    }

    #[tokio::test]
    async fn swallows_provider_source_failures_for_both_all_provider_and_single_provider_listing() {
        let models = create_models();

        // "broken" provider: registered but never successfully refreshed,
        // so its model cache stays empty — the Rust-semantic equivalent of
        // the TS test where getModels() throws and Models.getModels()
        // swallows the error, leaving only healthy providers' models.
        let broken = register_faux_provider(Some(RegisterFauxProviderOptions {
            provider: Some("broken".into()),
            ..Default::default()
        }));
        models.set_provider_arc(broken.provider.clone());
        let ok = register_faux_provider(Some(RegisterFauxProviderOptions {
            provider: Some("ok".into()),
            models: vec![FauxModelDefinition {
                id: "m1".into(),
                ..Default::default()
            }],
            ..Default::default()
        }));
        models.set_provider_arc(ok.provider.clone());
        // Only refresh the healthy provider; "broken" is left with an empty cache.
        models.refresh_provider_models("ok", false, true).await.unwrap();

        // all-provider listing only includes the "ok" provider's model
        let all_ids: Vec<String> =
            models.list_models(None).into_iter().map(|m| m.id).collect();
        assert_eq!(all_ids, vec!["m1".to_string()]);

        // single-provider listing for the broken provider returns empty
        assert!(models.list_models(Some("broken")).is_empty());
    }

    #[tokio::test]
    #[ignore = "requires dynamic provider refresh with RefreshModelsContext publish()"]
    async fn refresh_updates_every_configured_dynamic_provider_and_reports_failures() {}

    #[tokio::test]
    #[ignore = "requires refresh() with provider selection filter"]
    async fn restricts_refresh_work_to_selected_providers() {}

    #[tokio::test]
    #[ignore = "requires cached model restoration before network auth"]
    async fn restores_cached_models_before_waiting_for_network_auth() {}

    #[tokio::test]
    #[ignore = "requires atomic persistent deletion and ephemeral publication"]
    async fn lets_providers_choose_persistent_deletion_and_ephemeral_publication_atomically() {}

    #[tokio::test]
    #[ignore = "requires dynamic catalog persistence and offline restoration"]
    async fn persists_dynamic_catalogs_and_restores_them_without_network_access() {}

    #[tokio::test]
    #[ignore = "requires effective API-key credential passing and unconfigured provider skipping"]
    async fn passes_effective_api_key_credentials_and_refresh_options_while_skipping_unconfigured_providers() {}

    #[tokio::test]
    #[ignore = "requires expired OAuth refresh before model refresh"]
    async fn refreshes_expired_oauth_before_refreshing_models() {}

    #[tokio::test]
    #[ignore = "requires concrete signal always passed to providers"]
    async fn always_gives_providers_a_concrete_signal() {}

    #[tokio::test]
    #[ignore = "requires model-store waits bound to provider refresh signal"]
    async fn binds_model_store_waits_to_the_provider_refresh_signal() {}

    #[tokio::test]
    #[ignore = "requires aborted state without provider cancellation error"]
    async fn returns_aborted_state_without_reporting_cancellation_as_a_provider_error() {}

    #[tokio::test]
    #[ignore = "requires stopping on abort when provider ignores signal"]
    async fn stops_waiting_on_abort_when_a_provider_ignores_its_signal() {}

    #[tokio::test]
    #[ignore = "requires rejecting late publication from superseded provider"]
    async fn rejects_late_publication_from_a_superseded_non_cooperative_provider() {}

    #[tokio::test]
    #[ignore = "requires caller signals passed to provider auth callbacks"]
    async fn passes_caller_signals_to_provider_auth_callbacks() {}

    #[tokio::test]
    #[ignore = "requires stopping for non-cooperative auth callbacks"]
    async fn stops_waiting_for_non_cooperative_auth_callbacks() {}

    #[tokio::test]
    #[ignore = "requires cancelling queued credential mutations"]
    async fn cancels_queued_credential_mutations_without_running_them_later() {}

    #[tokio::test]
    #[ignore = "requires cancellation passed to OAuth refresh with credential preservation"]
    async fn passes_cancellation_to_oauth_refresh_and_preserves_the_previous_credential() {}

    #[tokio::test]
    #[ignore = "requires auth resolution: stored credential vs ambient"]
    async fn resolves_auth_stored_credential_owns_the_provider_ambient_only_when_nothing_stored() {}

    #[tokio::test]
    #[ignore = "requires checkAuth without OAuth refresh and available model filtering"]
    async fn checks_provider_auth_without_refreshing_oauth_and_filters_available_models() {}

    #[tokio::test]
    #[ignore = "requires provider login/logout through credential store"]
    async fn runs_provider_login_and_logout_through_the_credential_store() {}

    #[tokio::test]
    #[ignore = "requires stored credential without matching handler blocking ambient fallback"]
    async fn a_stored_credential_without_a_matching_handler_blocks_ambient_fallback() {}

    #[tokio::test]
    #[ignore = "requires expired OAuth refresh and rotated credential persistence"]
    async fn refreshes_expired_oauth_credentials_and_persists_the_rotated_credential() {}

    #[tokio::test]
    #[ignore = "requires OAuth refresh with < 5 min remaining"]
    async fn refreshes_oauth_credentials_with_less_than_five_minutes_remaining() {}

    #[tokio::test]
    #[ignore = "requires honoring caller's longer OAuth minimum validity"]
    async fn honors_a_callers_longer_oauth_minimum_validity() {}

    #[tokio::test]
    #[ignore = "requires OAuth refresh failure with code oauth and credential preservation"]
    async fn rejects_with_code_oauth_when_refresh_fails_preserving_the_stored_credential() {}

    #[tokio::test]
    #[ignore = "requires serialized concurrent OAuth refreshes (no double refresh)"]
    async fn serializes_concurrent_oauth_refreshes_through_store_modify_no_double_refresh() {}

    #[tokio::test]
    #[ignore = "requires valid OAuth tokens resolving without touching modify"]
    async fn valid_oauth_tokens_resolve_without_touching_modify() {}

    #[tokio::test]
    #[ignore = "requires wrapping credential store failures in ModelsError"]
    async fn wraps_credential_store_failures_in_models_error() {}

    #[tokio::test]
    #[ignore = "requires keeping underlying reason in wrapped OAuth refresh errors"]
    async fn keeps_the_underlying_reason_in_wrapped_oauth_refresh_errors() {}

    #[tokio::test]
    #[ignore = "requires wrapping api-key auth failures in ModelsError"]
    async fn wraps_api_key_auth_failures_in_models_error() {}

    #[tokio::test]
    #[ignore = "requires explicit request api key and env during auth resolution"]
    async fn uses_explicit_request_api_key_and_env_during_provider_auth_resolution() {}

    #[tokio::test]
    #[ignore = "requires merging resolved auth into stream options with per-field override"]
    async fn merges_resolved_auth_into_stream_options_explicit_options_win_per_field() {}

    #[tokio::test]
    #[ignore = "requires model headers for model auth and single header transform"]
    async fn adds_model_headers_only_for_model_auth_and_transforms_assembled_headers_once() {}
}

// ===========================================================================
// abort.test.ts — 41 tests (all #[ignore], require real API keys)
// ===========================================================================

mod abort {
    macro_rules! ignored_abort_test {
        ($name:ident, $desc:literal) => {
            #[tokio::test]
            #[ignore = "requires real API credentials and network access"]
            async fn $name() {
                // Mirrors TS abort test: $desc
            }
        };
    }

    // Google Provider
    ignored_abort_test!(google_should_abort_mid_stream, "Google Provider should abort mid-stream");
    ignored_abort_test!(google_should_handle_immediate_abort, "Google Provider should handle immediate abort");

    // OpenAI Completions Provider
    ignored_abort_test!(openai_completions_should_abort_mid_stream, "OpenAI Completions Provider should abort mid-stream");
    ignored_abort_test!(openai_completions_should_handle_immediate_abort, "OpenAI Completions Provider should handle immediate abort");

    // OpenAI Responses Provider
    ignored_abort_test!(openai_responses_should_abort_mid_stream, "OpenAI Responses Provider should abort mid-stream");
    ignored_abort_test!(openai_responses_should_handle_immediate_abort, "OpenAI Responses Provider should handle immediate abort");

    // Azure OpenAI Responses Provider
    ignored_abort_test!(azure_openai_responses_should_abort_mid_stream, "Azure OpenAI Responses Provider should abort mid-stream");
    ignored_abort_test!(azure_openai_responses_should_handle_immediate_abort, "Azure OpenAI Responses Provider should handle immediate abort");

    // Anthropic Provider
    ignored_abort_test!(anthropic_should_abort_mid_stream, "Anthropic Provider should abort mid-stream");
    ignored_abort_test!(anthropic_should_handle_immediate_abort, "Anthropic Provider should handle immediate abort");

    // Mistral Provider
    ignored_abort_test!(mistral_should_abort_mid_stream, "Mistral Provider should abort mid-stream");
    ignored_abort_test!(mistral_should_handle_immediate_abort, "Mistral Provider should handle immediate abort");

    // Together AI Provider
    ignored_abort_test!(together_ai_should_abort_mid_stream, "Together AI Provider should abort mid-stream");
    ignored_abort_test!(together_ai_should_handle_immediate_abort, "Together AI Provider should handle immediate abort");

    // Baseten Provider
    ignored_abort_test!(baseten_should_abort_mid_stream, "Baseten Provider should abort mid-stream");
    ignored_abort_test!(baseten_should_handle_immediate_abort, "Baseten Provider should handle immediate abort");

    // MiniMax Provider
    ignored_abort_test!(minimax_should_abort_mid_stream, "MiniMax Provider should abort mid-stream");
    ignored_abort_test!(minimax_should_handle_immediate_abort, "MiniMax Provider should handle immediate abort");

    // Xiaomi MiMo (API billing)
    ignored_abort_test!(xiaomi_mimo_api_billing_should_abort_mid_stream, "Xiaomi MiMo (API billing) should abort mid-stream");
    ignored_abort_test!(xiaomi_mimo_api_billing_should_handle_immediate_abort, "Xiaomi MiMo (API billing) should handle immediate abort");

    // Xiaomi MiMo Token Plan (CN)
    ignored_abort_test!(xiaomi_mimo_token_plan_cn_should_abort_mid_stream, "Xiaomi MiMo Token Plan (CN) should abort mid-stream");
    ignored_abort_test!(xiaomi_mimo_token_plan_cn_should_handle_immediate_abort, "Xiaomi MiMo Token Plan (CN) should handle immediate abort");

    // Xiaomi MiMo Token Plan (AMS)
    ignored_abort_test!(xiaomi_mimo_token_plan_ams_should_abort_mid_stream, "Xiaomi MiMo Token Plan (AMS) should abort mid-stream");
    ignored_abort_test!(xiaomi_mimo_token_plan_ams_should_handle_immediate_abort, "Xiaomi MiMo Token Plan (AMS) should handle immediate abort");

    // Xiaomi MiMo Token Plan (SGP)
    ignored_abort_test!(xiaomi_mimo_token_plan_sgp_should_abort_mid_stream, "Xiaomi MiMo Token Plan (SGP) should abort mid-stream");
    ignored_abort_test!(xiaomi_mimo_token_plan_sgp_should_handle_immediate_abort, "Xiaomi MiMo Token Plan (SGP) should handle immediate abort");

    // Qwen Token Plan
    ignored_abort_test!(qwen_token_plan_should_abort_mid_stream, "Qwen Token Plan should abort mid-stream");
    ignored_abort_test!(qwen_token_plan_should_handle_immediate_abort, "Qwen Token Plan should handle immediate abort");

    // Qwen Token Plan Individual
    ignored_abort_test!(qwen_token_plan_individual_should_abort_mid_stream, "Qwen Token Plan Individual should abort mid-stream");
    ignored_abort_test!(qwen_token_plan_individual_should_handle_immediate_abort, "Qwen Token Plan Individual should handle immediate abort");

    // Qwen Token Plan (CN)
    ignored_abort_test!(qwen_token_plan_cn_should_abort_mid_stream, "Qwen Token Plan (CN) should abort mid-stream");
    ignored_abort_test!(qwen_token_plan_cn_should_handle_immediate_abort, "Qwen Token Plan (CN) should handle immediate abort");

    // Kimi For Coding
    ignored_abort_test!(kimi_for_coding_should_abort_mid_stream, "Kimi For Coding should abort mid-stream");
    ignored_abort_test!(kimi_for_coding_should_handle_immediate_abort, "Kimi For Coding should handle immediate abort");

    // Vercel AI Gateway
    ignored_abort_test!(vercel_ai_gateway_should_abort_mid_stream, "Vercel AI Gateway should abort mid-stream");
    ignored_abort_test!(vercel_ai_gateway_should_handle_immediate_abort, "Vercel AI Gateway should handle immediate abort");

    // OpenAI Codex
    ignored_abort_test!(openai_codex_should_abort_mid_stream, "OpenAI Codex should abort mid-stream");
    ignored_abort_test!(openai_codex_should_handle_immediate_abort, "OpenAI Codex should handle immediate abort");

    // Amazon Bedrock (3 tests)
    ignored_abort_test!(amazon_bedrock_should_abort_mid_stream, "Amazon Bedrock should abort mid-stream");
    ignored_abort_test!(amazon_bedrock_should_handle_immediate_abort, "Amazon Bedrock should handle immediate abort");
    ignored_abort_test!(amazon_bedrock_should_handle_abort_then_new_message, "Amazon Bedrock should handle abort then new message");
}
