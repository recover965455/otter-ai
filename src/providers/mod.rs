use async_trait::async_trait;

use crate::auth::types::{Credential, ProviderAuth};
use crate::models_store::{ModelsStore, ModelsStoreEntry};
use crate::types::{ApiStreamOptions, Context, Model, SimpleStreamOptions};
use crate::utils::event_stream::AssistantMessageEventStream;

// Shared primitives used by most providers below.
pub mod oauth_compat;
pub mod openai_compat;
pub mod openai_responses;

// Built-in providers (opt-in at compile time via feature flags; "faux" is
// always available because it's the test mock everyone builds on top of).
#[cfg(feature = "providers-ant-ling")]
pub mod ant_ling;
#[cfg(feature = "providers-anthropic")]
pub mod anthropic;
#[cfg(feature = "providers-azure")]
pub mod azure;
#[cfg(feature = "providers-baseten")]
pub mod baseten;
#[cfg(feature = "providers-bedrock")]
pub mod bedrock;
#[cfg(feature = "providers-cerebras")]
pub mod cerebras;
#[cfg(feature = "providers-chatgpt-plus")]
pub mod chatgpt_plus;
#[cfg(feature = "providers-claude-pro-max")]
pub mod claude_pro_max;
#[cfg(feature = "providers-cloudflare-ai-gateway")]
pub mod cloudflare_ai_gateway;
#[cfg(feature = "providers-cloudflare-workers-ai")]
pub mod cloudflare_workers_ai;
#[cfg(feature = "providers-deepseek")]
pub mod deepseek;
pub mod faux;
#[cfg(feature = "providers-fireworks")]
pub mod fireworks;
#[cfg(feature = "providers-github-copilot")]
pub mod github_copilot;
#[cfg(feature = "providers-google")]
pub mod google;
#[cfg(feature = "providers-google-vertex")]
pub mod google_vertex;
#[cfg(feature = "providers-groq")]
pub mod groq;
#[cfg(feature = "providers-mistral")]
pub mod mistral;
#[cfg(feature = "providers-moonshot")]
pub mod moonshot;
#[cfg(feature = "providers-nvidia")]
pub mod nvidia;
#[cfg(feature = "providers-openai")]
pub mod openai;
#[cfg(feature = "providers-openrouter")]
pub mod openrouter;
#[cfg(feature = "providers-openrouter-oauth")]
pub mod openrouter_oauth;
#[cfg(feature = "providers-qwen-token-plan")]
pub mod qwen_token_plan;
#[cfg(feature = "providers-radius")]
pub mod radius;
#[cfg(feature = "providers-vercel-ai-gateway")]
pub mod vercel_ai_gateway;
#[cfg(feature = "providers-xai")]
pub mod xai;
#[cfg(feature = "providers-xai-subscription")]
pub mod xai_subscription;
#[cfg(feature = "providers-zai")]
pub mod zai;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsPublication {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persist: Option<Option<ModelsStoreEntry>>,
}

pub trait RefreshModelsContext {
    fn credential(&self) -> Option<&Credential>;
    fn stored(&self) -> Option<&ModelsStoreEntry>;
    fn allow_network(&self) -> bool;
    fn force(&self) -> bool;
    fn provider_id(&self) -> &str;
    fn publish(
        &self,
        publication: ModelsPublication,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send + '_>>;
}

#[async_trait]
pub trait Provider: Send + Sync {
    fn id(&self) -> &str;
    fn name(&self) -> &str;

    fn auth(&self) -> &ProviderAuth;

    fn get_models(&self) -> Vec<Model>;

    async fn refresh_models(
        &self,
        cx: Box<dyn RefreshModelsContext + Send + 'static>,
    ) -> Result<(), String> {
        let _ = cx;
        Ok(())
    }

    fn stream(
        &self,
        model: &Model,
        context: Context,
        options: ApiStreamOptions,
    ) -> AssistantMessageEventStream;

    fn stream_simple(
        &self,
        model: &Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let signal = options.signal.clone();
        let mut api_opts = ApiStreamOptions {
            signal,
            ..Default::default()
        };
        self.apply_simple_options(&context, &options, &mut api_opts);
        self.stream(model, context, api_opts)
    }

    /// Optional: override to merge provider-specific options (thinking levels, etc.)
    fn apply_simple_options(
        &self,
        _ctx: &Context,
        _options: &SimpleStreamOptions,
        _api_opts: &mut ApiStreamOptions,
    ) {
    }
}

#[derive(Clone)]
pub struct RefreshCtxState {
    pub credential: Option<Credential>,
    pub stored: Option<ModelsStoreEntry>,
    pub allow_network: bool,
    pub force: bool,
    pub provider_id: String,
    pub models_store: std::sync::Arc<dyn ModelsStore + Send + Sync>,
}

pub struct RefreshContext<'a> {
    pub state: std::borrow::Cow<'a, RefreshCtxState>,
}

impl<'a> RefreshModelsContext for RefreshContext<'a> {
    fn credential(&self) -> Option<&Credential> {
        self.state.credential.as_ref()
    }

    fn stored(&self) -> Option<&ModelsStoreEntry> {
        self.state.stored.as_ref()
    }

    fn allow_network(&self) -> bool {
        self.state.allow_network
    }

    fn force(&self) -> bool {
        self.state.force
    }

    fn provider_id(&self) -> &str {
        &self.state.provider_id
    }

    fn publish(
        &self,
        publication: ModelsPublication,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send + '_>> {
        Box::pin(async move {
            let provider_id = self.state.provider_id.clone();
            let store = self.state.models_store.clone();
            if let Some(entry_opt) = publication.persist {
                match entry_opt {
                    Some(entry) => store.write(&provider_id, &entry).await?,
                    None => store.delete(&provider_id).await?,
                }
            }
            Ok(())
        })
    }
}
