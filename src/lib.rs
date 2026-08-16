//! # otter-ai
//!
//! Unified LLM API with provider collections, automatic auth resolution,
//! token and cost tracking, and simple context persistence and hand-off
//! to other models mid-session.
//!
//! This is a Rust re-implementation of the `@earendil-works/pi-ai` TypeScript package.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use otter_ai::*;
//! use otter_ai::types::*;
//! use otter_ai::models::create_models;
//! use otter_ai::providers::faux::register_faux_provider;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // 1. Create the models registry
//! let models = create_models();
//!
//! // 2. Register a provider
//! let reg = register_faux_provider(None);
//! models.set_provider_arc(reg.provider.clone());
//!
//! // 3. Refresh the model catalog
//! models.refresh_all(false, true).await?;
//!
//! // 4. Look up a model
//! let model = models.get_model("faux", "faux-mini")
//!     .expect("model not found");
//!
//! // 5. Build a conversation context
//! let context = Context {
//!     system_prompt: Some("You are a helpful assistant.".into()),
//!     messages: vec![Message::User {
//!         content: vec![ContentBlock::Text { text: "Hello!".into() }],
//!         timestamp: chrono::Utc::now().timestamp_millis(),
//!     }],
//!     ..Default::default()
//! };
//!
//! // 6. Get a complete response
//! let options = SimpleStreamOptions::default();
//! let response = models.complete(&model, context, options).await?;
//! println!("{}", content_text(&match &response {
//!     Message::Assistant { content, .. } => content,
//!     _ => panic!("unexpected message type"),
//! }));
//! # Ok(())
//! # }
//! ```

pub mod auth;
pub mod models;
pub mod models_store;
pub mod providers;
pub mod types;
pub mod utils;

// Re-exports from types
pub use types::{
    content_text, uuidv7, Api, ApiStreamOptions, AssistantMessage, AssistantMessageEvent,
    CancellationToken, ContentBlock, Context, ImageContent, JsonSchemaFormat, KnownApi, Message,
    Model, ModelCostRates, ModelThinkingLevel, ProviderEnv, ProviderHeaders, ProviderId,
    ProviderRequestOptions, ResponseFormat, SimpleStreamOptions, Tool, ToolChoice, Usage,
    UsageCost,
};

// Re-exports from models
pub use models::{create_models, Models};

// Re-exports from models_store
pub use models_store::{InMemoryModelsStore, ModelsStore, ModelsStoreEntry};

// Re-exports from auth
pub use auth::{
    default_provider_auth_context, AuthContext, Credential, CredentialStore,
    InMemoryCredentialStore, ModelAuth,
};

// Re-exports from utils
pub use utils::{
    calculate_usage_cost, create_assistant_message_event_stream, parse_partial_json,
    string_enum_schema, tool_from_schema, validate_tool_arguments, with_retry,
    AssistantMessageEventStream, EventStream, RetryConfig,
};

// Re-export schemars for schema generation (replaces TypeBox from TS)
pub use schemars;
pub use schemars::schema_for;
pub use schemars::JsonSchema;

// Re-export serde_json for Value
pub use serde_json;
