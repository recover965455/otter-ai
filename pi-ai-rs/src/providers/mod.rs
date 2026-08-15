use async_trait::async_trait;

use crate::auth::types::{Credential, ModelAuth, ProviderAuth};
use crate::models_store::{ModelsStore, ModelsStoreEntry};
use crate::types::{
    ApiStreamOptions, AssistantMessage, AssistantMessageEvent, Context, Model,
    SimpleStreamOptions,
};

pub mod faux;
#[cfg(feature = "providers-openai")]
pub mod openai;
#[cfg(feature = "providers-anthropic")]
pub mod anthropic;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsPublication {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persist: Option<Option<ModelsStoreEntry>>,
}

pub struct RefreshModelsContext<'a> {
    pub credential: Option<Credential>,
    pub stored: Option<ModelsStoreEntry>,
    pub allow_network: bool,
    pub force: bool,
    pub models_store: &'a (dyn ModelsStore + Send + Sync),
    pub provider_id: &'a str,
}

impl<'a> RefreshModelsContext<'a> {
    pub async fn publish(&self, publication: ModelsPublication) -> anyhow::Result<bool> {
        if let Some(entry_opt) = publication.persist {
            match entry_opt {
                Some(entry) => self.models_store.write(self.provider_id, &entry).await?,
                None => self.models_store.delete(self.provider_id).await?,
            }
        }
        Ok(true)
    }
}

#[async_trait]
pub trait Provider: Send + Sync {
    fn id(&self) -> &str;
    fn name(&self) -> &str;

    fn auth(&self) -> &ProviderAuth;

    async fn refresh_models(&self, ctx: RefreshModelsContext<'_>) -> anyhow::Result<Vec<Model>>;

    fn stream(
        &self,
        model: &Model,
        auth: ModelAuth,
        context: Context,
        options: ApiStreamOptions,
    ) -> std::pin::Pin<
        Box<dyn futures::Stream<Item = AssistantMessageEvent> + Send + 'static>,
    >;

    fn complete(
        &self,
        model: &Model,
        auth: ModelAuth,
        context: Context,
        options: ApiStreamOptions,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<AssistantMessage>> + Send>>;

    /// Optional: override to merge provider-specific options (thinking levels, etc.)
    fn apply_simple_options(
        &self,
        _ctx: &Context,
        options: &SimpleStreamOptions,
        api_opts: &mut ApiStreamOptions,
    ) {
        let _ = options;
        let _ = api_opts;
    }
}

use serde::{Deserialize, Serialize};
