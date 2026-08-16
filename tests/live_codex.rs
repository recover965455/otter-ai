//! Live tests against the real ChatGPT Codex backend.
//!
//! Credentials are read from `~/.otter/credentials.json` (the `openai`
//! OAuth entry). These tests consume real subscription quota — run them
//! explicitly:
//!
//! ```sh
//! cargo test --test live_codex -- --ignored --nocapture
//! ```

use std::sync::Arc;

use futures::StreamExt;
use otter_ai::auth::{AuthOperationOptions, Credential, CredentialStore, FileCredentialStore};
use otter_ai::models::Models;
use otter_ai::providers::chatgpt_plus::chatgpt_plus_provider;
use otter_ai::providers::openai_responses::{stream_codex, CodexStreamOptions, CodexTransport};
use otter_ai::providers::Provider;
use otter_ai::types::{
    AssistantMessageEvent, Context, Message, ModelThinkingLevel, SimpleStreamOptions,
};

async fn load_openai_credential() -> Credential {
    let store = FileCredentialStore::open().expect("open ~/.otter/credentials.json");
    store
        .read("openai", AuthOperationOptions::default())
        .await
        .expect("read credentials.json")
        .expect("credentials.json has an `openai` entry")
}

fn access_token(cred: &Credential) -> String {
    match cred {
        Credential::OAuth(oc) => oc.inner.access.clone(),
        Credential::ApiKey(k) => k.key.clone().expect("api key"),
    }
}

fn pick_model(models: &[otter_ai::Model]) -> otter_ai::Model {
    models
        .iter()
        .find(|m| m.id == "gpt-5.4-mini")
        .or_else(|| models.first())
        .expect("chatgpt-plus model catalog is non-empty")
        .clone()
}

fn live_model() -> otter_ai::Model {
    pick_model(&chatgpt_plus_provider().get_models())
}

fn prompt(text: &str) -> Context {
    Context {
        system_prompt: None,
        messages: vec![Message::user_from_string(text)],
        ..Default::default()
    }
}

#[tokio::test]
#[ignore = "hits the real ChatGPT Codex backend and consumes subscription quota"]
async fn live_codex_stream_returns_text_and_usage() {
    let cred = load_openai_credential().await;
    let token = access_token(&cred);

    let opts = CodexStreamOptions {
        api_key: token,
        transport: CodexTransport::Sse,
        ..Default::default()
    };
    let model = live_model();
    let stream = stream_codex(&model, prompt("Reply with exactly: codex-e2e-ok"), opts);
    let result = stream.result_future();
    let mut events: Vec<AssistantMessageEvent> = Vec::new();
    let mut s = stream;
    while let Some(evt) = s.next().await {
        events.push(evt);
    }
    let msg = result.await;

    let types: Vec<&str> = events
        .iter()
        .map(|e| match e {
            AssistantMessageEvent::Start { .. } => "start",
            AssistantMessageEvent::TextStart => "text_start",
            AssistantMessageEvent::TextDelta { .. } => "text_delta",
            AssistantMessageEvent::TextEnd => "text_end",
            AssistantMessageEvent::Usage { .. } => "usage",
            AssistantMessageEvent::Done { .. } => "done",
            AssistantMessageEvent::Error { .. } => "error",
            _ => "other",
        })
        .collect();
    println!("live event sequence: {:?}", types);
    println!(
        "live model {} -> {:?}",
        model.id,
        otter_ai::content_text(match &msg {
            Message::Assistant { content, .. } => content,
            _ => &[],
        })
    );

    assert_eq!(
        msg.stop_reason().unwrap_or_default(),
        "stop",
        "events: {:?}",
        types
    );
    let text = otter_ai::content_text(match &msg {
        Message::Assistant { content, .. } => content,
        _ => &[],
    });
    assert!(!text.is_empty(), "expected non-empty text output");
    if let Message::Assistant { usage, .. } = &msg {
        assert!(usage.total_tokens > 0, "usage reported: {:?}", usage);
        println!("live usage: {:?}", usage);
    }
}

#[tokio::test]
#[ignore = "hits the real ChatGPT Codex backend and consumes subscription quota"]
async fn live_codex_websocket_transport_returns_text_and_usage() {
    let cred = load_openai_credential().await;
    let token = access_token(&cred);

    let opts = CodexStreamOptions {
        api_key: token,
        transport: CodexTransport::Auto,
        ..Default::default()
    };
    let model = live_model();
    let stream = stream_codex(&model, prompt("Reply with exactly: ws-live-ok"), opts);
    let result = stream.result_future();
    let mut events: Vec<AssistantMessageEvent> = Vec::new();
    let mut s = stream;
    while let Some(evt) = s.next().await {
        events.push(evt);
    }
    let msg = result.await;

    let types: Vec<&str> = events
        .iter()
        .map(|e| match e {
            AssistantMessageEvent::Start { .. } => "start",
            AssistantMessageEvent::TextStart => "text_start",
            AssistantMessageEvent::TextDelta { .. } => "text_delta",
            AssistantMessageEvent::TextEnd => "text_end",
            AssistantMessageEvent::Usage { .. } => "usage",
            AssistantMessageEvent::Done { .. } => "done",
            AssistantMessageEvent::Error { .. } => "error",
            _ => "other",
        })
        .collect();
    println!("live ws event sequence: {:?}", types);
    println!(
        "live ws model {} -> {:?}",
        model.id,
        otter_ai::content_text(match &msg {
            Message::Assistant { content, .. } => content,
            _ => &[],
        })
    );

    assert_eq!(
        msg.stop_reason().unwrap_or_default(),
        "stop",
        "events: {:?}",
        types
    );
    let text = otter_ai::content_text(match &msg {
        Message::Assistant { content, .. } => content,
        _ => &[],
    });
    assert!(!text.is_empty(), "expected non-empty text output");
    if let Message::Assistant { usage, .. } = &msg {
        assert!(usage.total_tokens > 0, "usage reported: {:?}", usage);
        println!("live ws usage: {:?}", usage);
    }
}

#[tokio::test]
#[ignore = "hits the real ChatGPT Codex backend and consumes subscription quota"]
async fn live_models_full_stack_completes_from_file_credentials() {
    let store = Arc::new(FileCredentialStore::open().expect("open ~/.otter/credentials.json"));

    let models = Models::new().with_credential_store(store);
    models.set_provider_arc(Arc::new(chatgpt_plus_provider()));

    let model = live_model();
    let msg = models
        .complete(
            &model,
            prompt("Reply with exactly: codex-e2e-ok"),
            SimpleStreamOptions {
                thinking: Some(ModelThinkingLevel::None),
                ..Default::default()
            },
        )
        .await
        .expect("live completion through the full stack");

    assert_eq!(msg.stop_reason().unwrap_or_default(), "stop");
    let text = otter_ai::content_text(match &msg {
        Message::Assistant { content, .. } => content,
        _ => &[],
    });
    assert!(!text.is_empty());
    if let Message::Assistant {
        usage, model: m, ..
    } = &msg
    {
        assert_eq!(m.as_deref(), Some(model.id.as_str()));
        println!("live full-stack usage: {:?}", usage);
    }
}
