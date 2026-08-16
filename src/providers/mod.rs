use async_trait::async_trait;

use crate::auth::types::{Credential, ProviderAuth};
use crate::models_store::{ModelsStore, ModelsStoreEntry};
use crate::types::{ApiStreamOptions, Context, Model, SimpleStreamOptions};
use crate::utils::event_stream::AssistantMessageEventStream;

#[cfg(feature = "providers-anthropic")]
pub mod anthropic;
pub mod faux;
#[cfg(feature = "providers-openai")]
pub mod openai;

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
