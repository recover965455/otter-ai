//! Integration tests for pi-ai-rs — uses the Faux provider so no network calls needed.

use futures::StreamExt;
use pi_ai_rs::auth::types::Credential;
use pi_ai_rs::models::create_models;
use pi_ai_rs::providers::faux::{faux_provider, FauxResponseConfig};
use pi_ai_rs::utils::validation::calculate_usage_cost;
use pi_ai_rs::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[tokio::test]
async fn test_faux_provider_refresh_and_lookup() {
    let models = create_models();
    let faux = faux_provider();
    models.set_provider(faux);

    // Refresh model catalog
    let count = models.refresh_all(false, true).await.expect("refresh failed");
    assert!(count > 0, "should have refreshed at least 1 model");

    // Look up a model
    let mini = models
        .get_model("faux", "faux-mini")
        .expect("faux-mini should exist");
    assert_eq!(mini.id, "faux-mini");
    assert_eq!(mini.provider_id, "faux");
    assert!(mini.supports_tool_calling);

    // List providers and models
    let providers = models.list_providers();
    assert_eq!(providers.len(), 1);
    assert_eq!(providers[0].0, "faux");

    let all_models = models.list_models(None);
    assert!(all_models.len() >= 2);

    let faux_only = models.list_models(Some("faux"));
    assert_eq!(faux_only.len(), all_models.len());
}

#[tokio::test]
async fn test_faux_complete_text_response() {
    let models = create_models();
    let faux = faux_provider();
    faux.enqueue_response(FauxResponseConfig {
        text: "Hello, test!".into(),
        input_tokens: 10,
        output_tokens: 3,
        ..Default::default()
    });
    models.set_provider(faux);
    models.refresh_all(false, true).await.unwrap();
    let model = models.get_model("faux", "faux-mini").unwrap();

    let ctx = Context {
        system_prompt: Some("Be helpful.".into()),
        messages: vec![Message::User {
            content: vec![ContentBlock::Text {
                text: "Hi!".into(),
            }],
            timestamp: 0,
        }],
        ..Default::default()
    };

    let response = models
        .complete(&model, ctx, SimpleStreamOptions::default())
        .await
        .expect("complete should succeed");

    match &response {
        Message::Assistant {
            content,
            usage,
            model: mid,
            stop_reason,
            ..
        } => {
            assert_eq!(mid.as_deref(), Some("faux-mini"));
            assert_eq!(stop_reason.as_deref(), Some("end_turn"));
            assert_eq!(usage.input, 10);
            assert_eq!(usage.output, 3);
            assert!(usage.cost.total > 0.0);
            let text = content_text(content);
            assert_eq!(text, "Hello, test!");
        }
        _ => panic!("expected assistant message"),
    }
}

#[tokio::test]
async fn test_faux_stream_collects_events() {
    let models = create_models();
    let faux = faux_provider();
    faux.enqueue_response(FauxResponseConfig {
        text: "Hi!".into(),
        thinking: Some("let me think...".into()),
        tool_calls: vec![(
            "get_weather".into(),
            serde_json::json!({ "city": "SF" }),
        )],
        input_tokens: 5,
        output_tokens: 5,
        ..Default::default()
    });
    models.set_provider(faux);
    models.refresh_all(false, true).await.unwrap();
    let model = models.get_model("faux", "faux-pro").unwrap();

    let ctx = Context {
        messages: vec![Message::User {
            content: vec![ContentBlock::Text {
                text: "weather?".into(),
            }],
            timestamp: 0,
        }],
        ..Default::default()
    };

    let mut stream = models.stream(&model, ctx, SimpleStreamOptions::default());
    let mut events: Vec<AssistantMessageEvent> = vec![];
    while let Some(evt) = stream.next().await {
        events.push(evt);
    }

    assert!(!events.is_empty(), "should have produced events");

    // Check event sequence
    let types: Vec<&str> = events
        .iter()
        .map(|e| match e {
            AssistantMessageEvent::Start { .. } => "start",
            AssistantMessageEvent::ThinkingStart => "thinking_start",
            AssistantMessageEvent::ThinkingDelta { .. } => "thinking_delta",
            AssistantMessageEvent::ThinkingEnd { .. } => "thinking_end",
            AssistantMessageEvent::TextStart => "text_start",
            AssistantMessageEvent::TextDelta { .. } => "text_delta",
            AssistantMessageEvent::TextEnd => "text_end",
            AssistantMessageEvent::ToolcallStart { .. } => "toolcall_start",
            AssistantMessageEvent::ToolcallDelta { .. } => "toolcall_delta",
            AssistantMessageEvent::ToolcallEnd { .. } => "toolcall_end",
            AssistantMessageEvent::Usage { .. } => "usage",
            AssistantMessageEvent::Done { .. } => "done",
            AssistantMessageEvent::Error { .. } => "error",
        })
        .collect();

    assert!(types.contains(&"start"));
    assert!(types.contains(&"thinking_start"));
    assert!(types.contains(&"thinking_end"));
    assert!(types.contains(&"text_start"));
    assert!(types.contains(&"text_end"));
    assert!(types.contains(&"toolcall_start"));
    assert!(types.contains(&"toolcall_end"));
    assert!(types.contains(&"usage"));
    assert!(matches!(
        events.last().unwrap(),
        AssistantMessageEvent::Done { .. }
    ));
}

#[tokio::test]
async fn test_faux_error_response() {
    let models = create_models();
    let faux = faux_provider();
    faux.enqueue_response(FauxResponseConfig {
        error: Some("intentional failure".into()),
        ..Default::default()
    });
    models.set_provider(faux);
    models.refresh_all(false, true).await.unwrap();
    let model = models.get_model("faux", "faux-mini").unwrap();

    let ctx = Context::default();
    let result = models
        .complete(&model, ctx, SimpleStreamOptions::default())
        .await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("intentional failure"));
}

#[test]
fn test_calculate_usage_cost_applies_rates() {
    let model = Model {
        id: "test".into(),
        provider_id: "test".into(),
        name: "Test".into(),
        api: "test".into(),
        max_input_tokens: None,
        max_output_tokens: None,
        supports_images: false,
        supports_audio: false,
        supports_video: false,
        supports_pdf: false,
        supports_tool_calling: false,
        supports_structured_output: false,
        supports_system_prompt: false,
        thinking: ModelThinkingLevel::None,
        cost_rates: ModelCostRates {
            input_per_million: Some(1.0),
            output_per_million: Some(2.0),
            input_cache_read_per_million: Some(0.5),
            input_cache_write_per_million: Some(0.25),
        },
        context_window: None,
        default_temperature: None,
    };

    // 1M input, 1M output, 1M cache_read, 1M cache_write → 1.0 + 2.0 + 0.5 + 0.25 = 3.75
    let cost = calculate_usage_cost(1_000_000, 1_000_000, 1_000_000, 1_000_000, &model);
    assert!((cost.input - 1.0).abs() < 1e-9);
    assert!((cost.output - 2.0).abs() < 1e-9);
    assert!((cost.cache_read - 0.5).abs() < 1e-9);
    assert!((cost.cache_write - 0.25).abs() < 1e-9);
    assert!((cost.total - 3.75).abs() < 1e-9);
}

#[test]
fn test_calculate_usage_cost_missing_rates_are_zero() {
    let model = Model {
        id: "test".into(),
        provider_id: "test".into(),
        name: "Test".into(),
        api: "test".into(),
        max_input_tokens: None,
        max_output_tokens: None,
        supports_images: false,
        supports_audio: false,
        supports_video: false,
        supports_pdf: false,
        supports_tool_calling: false,
        supports_structured_output: false,
        supports_system_prompt: false,
        thinking: ModelThinkingLevel::None,
        cost_rates: ModelCostRates::default(),
        context_window: None,
        default_temperature: None,
    };

    let cost = calculate_usage_cost(1_000_000, 1_000_000, 1_000_000, 1_000_000, &model);
    assert_eq!(cost.total, 0.0);
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct GetWeatherArgs {
    /// City name
    city: String,
    /// Temperature unit
    unit: Option<String>,
}

#[test]
fn test_tool_from_schema_generates_valid_json_schema() {
    let tool = tool_from_schema::<GetWeatherArgs>("get_weather", Some("Fetch the current weather".to_string()));
    assert_eq!(tool.name, "get_weather");
    assert_eq!(tool.description.as_deref(), Some("Fetch the current weather"));

    // Should be an object schema with required/optional properties
    let params = &tool.parameters;
    assert_eq!(params.get("type").and_then(|v| v.as_str()), Some("object"));
    let props = params
        .get("properties")
        .and_then(|p| p.as_object())
        .expect("should have properties");
    assert!(props.contains_key("city"));
    assert!(props.contains_key("unit"));
    // "unit" should not be required because it's Option<String>
    let required = params
        .get("required")
        .and_then(|r| r.as_array())
        .cloned()
        .unwrap_or_default();
    assert!(required.iter().any(|s| s == "city"));
    assert!(!required.iter().any(|s| s == "unit"));
}

#[test]
fn test_string_enum_schema() {
    let schema = utils::string_enum_schema(
        &["low", "medium", "high"],
        Some("Difficulty level"),
        Some("medium"),
    );
    assert_eq!(schema.get("type").and_then(|v| v.as_str()), Some("string"));
    let enum_vals = schema
        .get("enum")
        .and_then(|e| e.as_array())
        .expect("enum should be array");
    assert_eq!(enum_vals.len(), 3);
    assert_eq!(
        schema.get("description").and_then(|v| v.as_str()),
        Some("Difficulty level")
    );
    assert_eq!(
        schema.get("default").and_then(|v| v.as_str()),
        Some("medium")
    );
}

#[test]
fn test_content_text_extracts_only_text_blocks() {
    let blocks = vec![
        ContentBlock::Text {
            text: "hello".into(),
        },
        ContentBlock::ToolCall {
            id: "tc1".into(),
            name: "f".into(),
            arguments: serde_json::json!({}),
        },
        ContentBlock::Text {
            text: "world".into(),
        },
    ];
    assert_eq!(content_text(&blocks), "hello\nworld");
}

#[test]
fn test_cancellation_token_basic() {
    let token = CancellationToken::new();
    assert!(!token.is_cancelled());
    token.cancel();
    assert!(token.is_cancelled());

    let child = CancellationToken::new();
    let copy = child.child_token();
    child.cancel();
    assert!(copy.is_cancelled());
}

#[test]
fn test_uuidv7_is_valid_uuid() {
    let id = uuidv7();
    // UUID v7 → 13th hex char (after stripping hyphens) should be 7
    let stripped: String = id.chars().filter(|c| *c != '-').collect();
    assert_eq!(stripped.len(), 32);
    assert_eq!(stripped.chars().nth(12), Some('7'));
}

#[tokio::test]
async fn test_in_memory_credential_store_crud() {
    use pi_ai_rs::auth::context::InMemoryCredentialStore;
    use pi_ai_rs::auth::types::{CredentialStore, AuthOperationOptions};

    let store = InMemoryCredentialStore::new();
    let opts = AuthOperationOptions::default();

    // Initially empty
    let cred = store.read("openai", opts.clone()).await.unwrap();
    assert!(cred.is_none());

    // Create via modify_fn
    let result = store
        .modify_fn(
            "openai",
            Box::new(|_prev| Box::pin(async { Ok(Some(Credential::api_key("sk-test"))) })),
            opts.clone(),
        )
        .await
        .unwrap();
    assert!(result.is_some());

    // Read it back
    let read_back = store.read("openai", opts.clone()).await.unwrap();
    assert!(read_back.is_some());

    // List includes it
    let list = store.list(opts.clone()).await.unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].provider_id, "openai");
    assert_eq!(list[0].r#type, "api_key");

    // Delete
    store.delete("openai", opts.clone()).await.unwrap();
    let gone = store.read("openai", opts).await.unwrap();
    assert!(gone.is_none());
}
