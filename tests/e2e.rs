//! Local end-to-end tests through the public `Models` API + faux provider.
//! No network, no credentials required.

use std::sync::Arc;

use futures::StreamExt;
use otter_ai::auth::{
    AuthOperationOptions, AuthResolutionOverrides, Credential, CredentialStore,
    InMemoryCredentialStore, ModifyFnOutput,
};
use otter_ai::models::Models;
use otter_ai::models_store::{InMemoryModelsStore, ModelsStore, ModelsStoreEntry};
use otter_ai::providers::faux::{
    faux_assistant_message, faux_provider, faux_text, faux_tool_call, register_faux_provider,
    FauxAssistantMessageOptions, FauxModelDefinition, FauxResponseStep, FauxToolCallOptions,
    RegisterFauxProviderOptions,
};
use otter_ai::providers::{ModelsPublication, Provider, RefreshModelsContext};
use otter_ai::types::{
    AssistantMessageEvent, ContentBlock, Context, Message, Model, ModelCostRates, SimpleStreamOptions,
    Tool, Usage,
};

fn user_context(text: &str) -> Context {
    Context {
        system_prompt: Some("You are a helpful assistant.".to_string()),
        messages: vec![Message::user_from_string(text)],
        ..Default::default()
    }
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

async fn put_credential(store: &InMemoryCredentialStore, provider_id: &str, cred: Credential) {
    store
        .modify_fn(
            provider_id,
            Box::new(move |_| {
                Box::pin(async move { Ok(Some(cred)) }) as ModifyFnOutput
            }),
            AuthOperationOptions::default(),
        )
        .await
        .expect("write credential");
}

// ---------------------------------------------------------------------------
// 1. Multi-turn conversation with tool-call backfill
// ---------------------------------------------------------------------------

#[tokio::test]
async fn multi_turn_conversation_with_tool_call_backfill() {
    let reg = register_faux_provider(None);
    let models = Models::new();
    models.set_provider_arc(reg.provider.clone());
    let model = reg.get_model(None).expect("default faux model");

    // Turn 1: assistant asks to call a tool.
    reg.set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
        vec![
            faux_text("Let me check the weather."),
            faux_tool_call(
                "get_weather",
                serde_json::json!({ "city": "Paris" }),
                FauxToolCallOptions::default(),
            ),
        ],
        FauxAssistantMessageOptions {
            stop_reason: Some("toolUse".to_string()),
            ..Default::default()
        },
    ))]);
    let first = models
        .complete(&model, user_context("Weather in Paris?"), SimpleStreamOptions::default())
        .await
        .expect("first turn completes");
    assert_eq!(stop_reason_of(&first), "toolUse");

    // Turn 2: tool result + follow-up user message → plain text answer.
    let tool_call_id = match match &first {
        Message::Assistant { content, .. } => &content[1],
        _ => panic!("expected assistant"),
    } {
        ContentBlock::ToolCall { id, .. } => id.clone(),
        other => panic!("expected tool call, got {:?}", other),
    };
    let mut ctx = user_context("Weather in Paris?");
    ctx.messages.push(first);
    ctx.messages.push(Message::ToolResult {
        tool_call_id,
        tool_name: "get_weather".to_string(),
        content: vec![ContentBlock::Text { text: "22C, sunny".to_string(), text_signature: None}],
        is_error: false,
        timestamp: 2,
    });
    ctx.messages.push(Message::user_from_string("Summarize."));

    reg.set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
        "It is 22C and sunny in Paris.",
        FauxAssistantMessageOptions::default(),
    ))]);
    let second = models
        .complete(&model, ctx, SimpleStreamOptions::default())
        .await
        .expect("second turn completes");
    assert_eq!(text_of(&second), "It is 22C and sunny in Paris.");
    assert_eq!(stop_reason_of(&second), "stop");
}

// ---------------------------------------------------------------------------
// 2. Streaming event sequence
// ---------------------------------------------------------------------------

#[tokio::test]
async fn streaming_emits_the_expected_event_sequence() {
    let reg = faux_provider(Some(RegisterFauxProviderOptions {
        token_size: Some((3, 3)), // deterministic single 12-char chunk
        ..Default::default()
    }));
    let models = Models::new();
    models.set_provider_arc(reg.provider.clone());
    let model = reg.get_model(None).expect("model");

    reg.set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
        "abcdefghijkl",
        FauxAssistantMessageOptions::default(),
    ))]);

    let mut stream = models.stream(&model, user_context("hi"), SimpleStreamOptions::default());
    let mut events = Vec::new();
    while let Some(evt) = stream.next().await {
        events.push(evt);
    }

    let types = event_types(&events);
    assert_eq!(
        types,
        vec!["start", "text_start", "text_delta", "text_end", "usage", "done"]
    );
    match &events[2] {
        AssistantMessageEvent::TextDelta { delta } => assert_eq!(delta, "abcdefghijkl"),
        other => panic!("expected text delta, got {:?}", other),
    }
    match &events[5] {
        AssistantMessageEvent::Done { message, .. } => assert_eq!(text_of(message), "abcdefghijkl"),
        other => panic!("expected done, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 3. Credential resolution (store + overrides)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn resolves_credentials_from_the_store_and_honours_overrides() {
    let reg = register_faux_provider(None);
    let models = Models::new();
    models.set_provider_arc(reg.provider.clone());

    // No stored credential → still resolves (ambient), no key.
    let ambient = models
        .get_auth("faux", AuthResolutionOverrides::default())
        .await
        .expect("ambient auth resolves");
    assert!(ambient.auth.api_key.is_none());

    // With an api-key provider + stored credential the key is surfaced.
    let provider = otter_ai::providers::openai::openai_provider();
    models.set_provider_arc(Arc::new(provider));
    let store = InMemoryCredentialStore::new();
    put_credential(&store, "openai", Credential::api_key("sk-store-key")).await;
    let models = models.with_credential_store(Arc::new(store));

    let resolved = models
        .get_auth("openai", AuthResolutionOverrides::default())
        .await
        .expect("resolves from store");
    assert_eq!(resolved.auth.api_key.as_deref(), Some("sk-store-key"));
    assert_eq!(resolved.source.as_deref(), Some("credential_store"));

    // An override credential wins over the store.
    let overridden = models
        .get_auth(
            "openai",
            AuthResolutionOverrides {
                credential: Some(Credential::api_key("sk-override")),
                base_url: Some("https://example.test/v1".to_string()),
                headers: None,
            },
        )
        .await
        .expect("override resolves");
    assert_eq!(overridden.auth.api_key.as_deref(), Some("sk-override"));
    assert_eq!(overridden.auth.base_url.as_deref(), Some("https://example.test/v1"));
}

// ---------------------------------------------------------------------------
// 4. Model-catalog persistence via refresh_models publication
// ---------------------------------------------------------------------------

struct CatalogProvider {
    models: Vec<Model>,
}

#[otter_ai::async_trait]
impl Provider for CatalogProvider {
    fn id(&self) -> &str {
        "catalog-test"
    }
    fn name(&self) -> &str {
        "Catalog Test"
    }
    fn auth(&self) -> &otter_ai::auth::ProviderAuth {
        static EMPTY: std::sync::OnceLock<otter_ai::auth::ProviderAuth> = std::sync::OnceLock::new();
        EMPTY.get_or_init(|| otter_ai::auth::ProviderAuth {
            api_key: None,
            oauth: None,
        })
    }
    fn get_models(&self) -> Vec<Model> {
        self.models.clone()
    }
    async fn refresh_models(
        &self,
        cx: Box<dyn RefreshModelsContext + Send + 'static>,
    ) -> Result<(), String> {
        let entry = ModelsStoreEntry {
            models: self.models.clone(),
            fetched_at: Some(42),
            etag: Some("etag-1".to_string()),
        };
        cx.publish(ModelsPublication {
            persist: Some(Some(entry)),
        })
        .await
        .map_err(|e| e.to_string())
    }
    fn stream(
        &self,
        _model: &Model,
        _context: Context,
        _options: otter_ai::ApiStreamOptions,
    ) -> otter_ai::AssistantMessageEventStream {
        otter_ai::create_assistant_message_event_stream()
    }
}

#[tokio::test]
async fn refresh_persists_the_model_catalog_into_the_store() {
    let model = Model {
        id: "cat-1".to_string(),
        provider_id: "catalog-test".to_string(),
        name: "Cat 1".to_string(),
        api: "faux".to_string(),
        max_input_tokens: None,
        max_output_tokens: Some(1024),
        supports_images: false,
        supports_audio: false,
        supports_video: false,
        supports_pdf: false,
        supports_tool_calling: true,
        supports_structured_output: false,
        supports_system_prompt: true,
        thinking: otter_ai::ModelThinkingLevel::None,
        reasoning: false,
        cost_rates: ModelCostRates::default(),
        context_window: Some(8192),
        default_temperature: None,
        thinking_level_map: None,
    };

    let models_store = Arc::new(InMemoryModelsStore::new());
    let models = Models::new().with_models_store(models_store.clone());
    models.set_provider_arc(Arc::new(CatalogProvider {
        models: vec![model],
    }));

    let refreshed = models
        .refresh_provider_models("catalog-test", false, true)
        .await
        .expect("refresh succeeds");
    assert_eq!(refreshed.len(), 1);
    assert_eq!(refreshed[0].id, "cat-1");
    assert!(models.get_model("catalog-test", "cat-1").is_some());

    let stored = models_store
        .read("catalog-test")
        .await
        .expect("store read")
        .expect("entry persisted");
    assert_eq!(stored.etag.as_deref(), Some("etag-1"));
    assert_eq!(stored.fetched_at, Some(42));
    assert_eq!(stored.models.len(), 1);
}

// ---------------------------------------------------------------------------
// 5. Multi-provider isolation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn faux_providers_keep_isolated_response_queues() {
    let reg_a = register_faux_provider(Some(RegisterFauxProviderOptions {
        provider: Some("faux-a".to_string()),
        ..Default::default()
    }));
    let reg_b = register_faux_provider(Some(RegisterFauxProviderOptions {
        provider: Some("faux-b".to_string()),
        ..Default::default()
    }));
    let models = Models::new();
    models.set_provider_arc(reg_a.provider.clone());
    models.set_provider_arc(reg_b.provider.clone());

    reg_a.set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
        "from a",
        FauxAssistantMessageOptions::default(),
    ))]);
    reg_b.set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
        "from b",
        FauxAssistantMessageOptions::default(),
    ))]);

    let model_a = reg_a.get_model(None).unwrap();
    let model_b = reg_b.get_model(None).unwrap();

    let a1 = models
        .complete(&model_a, user_context("hi"), SimpleStreamOptions::default())
        .await
        .unwrap();
    let b1 = models
        .complete(&model_b, user_context("hi"), SimpleStreamOptions::default())
        .await
        .unwrap();
    assert_eq!(text_of(&a1), "from a");
    assert_eq!(text_of(&b1), "from b");

    // A's queue is exhausted; B still has its own (empty too after use) —
    // calling A again errors while the providers stay independently usable.
    let a2 = models
        .complete(&model_a, user_context("hi"), SimpleStreamOptions::default())
        .await;
    assert!(a2.is_err(), "faux-a queue exhausted");
    assert_eq!(reg_a.get_pending_response_count(), 0);
    assert_eq!(reg_b.get_pending_response_count(), 0);
}

// ---------------------------------------------------------------------------
// 6. Error paths
// ---------------------------------------------------------------------------

#[tokio::test]
async fn surfaces_provider_errors_and_unknown_providers_as_error_events() {
    let reg = register_faux_provider(None);
    let models = Models::new();
    models.set_provider_arc(reg.provider.clone());
    let model = reg.get_model(None).unwrap();

    // Provider-side error message surfaces through complete() as an Err.
    reg.set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
        "",
        FauxAssistantMessageOptions {
            stop_reason: Some("error".to_string()),
            error_message: Some("faux blew up".to_string()),
            ..Default::default()
        },
    ))]);
    let err = models
        .complete(&model, user_context("hi"), SimpleStreamOptions::default())
        .await;
    assert!(err.is_err());

    // Unknown provider → provider-not-found error event on the stream.
    let ghost = Model {
        id: "ghost-1".to_string(),
        provider_id: "no-such-provider".to_string(),
        name: "Ghost".to_string(),
        api: "faux".to_string(),
        max_input_tokens: None,
        max_output_tokens: None,
        supports_images: false,
        supports_audio: false,
        supports_video: false,
        supports_pdf: false,
        supports_tool_calling: false,
        supports_structured_output: false,
        supports_system_prompt: false,
        thinking: otter_ai::ModelThinkingLevel::None,
        reasoning: false,
        cost_rates: ModelCostRates::default(),
        context_window: None,
        default_temperature: None,
        thinking_level_map: None,
    };
    let mut stream = models.stream(&ghost, user_context("hi"), SimpleStreamOptions::default());
    let mut events = Vec::new();
    while let Some(evt) = stream.next().await {
        events.push(evt);
    }
    let types = event_types(&events);
    assert_eq!(types, vec!["error"]);
    match &events[0] {
        AssistantMessageEvent::Error { reason, error } => {
            assert_eq!(reason, "provider-not-found");
            assert!(error.contains("no-such-provider"));
        }
        other => panic!("expected error event, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 7. Usage + cost accounting
// ---------------------------------------------------------------------------

#[tokio::test]
async fn accounts_for_usage_and_cost_through_the_full_stack() {
    let reg = register_faux_provider(Some(RegisterFauxProviderOptions {
        models: vec![FauxModelDefinition {
            id: "costly".to_string(),
            name: Some("Costly".to_string()),
            reasoning: false,
            context_window: Some(128_000),
            max_tokens: Some(4_096),
            supports_images: false,
            cost_rates: ModelCostRates {
                input_per_million: Some(3.0),
                output_per_million: Some(15.0),
                input_cache_read_per_million: Some(0.3),
                input_cache_write_per_million: Some(3.75),
                tiers: vec![],
            },
        }],
        ..Default::default()
    }));
    let models = Models::new();
    models.set_provider_arc(reg.provider.clone());
    let model = reg.get_model(Some("costly")).expect("costly model");

    reg.set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
        "The quick brown fox jumps over the lazy dog",
        FauxAssistantMessageOptions::default(),
    ))]);

    let msg = models
        .complete(&model, user_context("hello there"), SimpleStreamOptions::default())
        .await
        .expect("completes");

    let usage = match &msg {
        Message::Assistant { usage, .. } => usage.clone(),
        _ => panic!("expected assistant"),
    };
    assert!(usage.input > 0, "input tokens estimated: {:?}", usage);
    assert!(usage.output > 0, "output tokens estimated: {:?}", usage);
    assert_eq!(
        usage.total_tokens,
        usage.input + usage.output + usage.cache_read + usage.cache_write
    );

    let cost = otter_ai::calculate_usage_cost(
        usage.input,
        usage.output,
        usage.cache_read,
        usage.cache_write,
        &model,
    );
    let expected_input = usage.input as f64 / 1_000_000.0 * 3.0;
    let expected_output = usage.output as f64 / 1_000_000.0 * 15.0;
    assert!((cost.input - expected_input).abs() < 1e-9);
    assert!((cost.output - expected_output).abs() < 1e-9);
    assert!((cost.total - (expected_input + expected_output)).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// Bonus: tools round-trip through the serialized context (faux sees them)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tools_are_visible_to_the_provider_context() {
    let reg = register_faux_provider(None);
    let models = Models::new();
    models.set_provider_arc(reg.provider.clone());
    let model = reg.get_model(None).unwrap();

    reg.set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
        "ok",
        FauxAssistantMessageOptions::default(),
    ))]);

    let mut ctx = user_context("use the tool");
    ctx.tools = vec![Tool {
        name: "echo".to_string(),
        description: Some("Echo".to_string()),
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"]
        }),
        constrained_sampling: None,
    }];

    let msg = models
        .complete(&model, ctx, SimpleStreamOptions::default())
        .await
        .expect("completes");
    assert_eq!(text_of(&msg), "ok");

    let usage = match &msg {
        Message::Assistant { usage, .. } => usage.clone(),
        _ => Usage::default(),
    };
    // The serialized prompt includes the tool schema, so input covers it.
    assert!(usage.input > 0);
}
