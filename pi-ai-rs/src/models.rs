use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;
use parking_lot::RwLock;

use crate::auth::context::DefaultAuthContext;
use crate::auth::context::InMemoryCredentialStore;
use crate::auth::{resolve_provider_auth, AuthResolutionOverrides, ModelsError};
use crate::auth::types::{AuthContext, CredentialStore};
use crate::models_store::{InMemoryModelsStore, ModelsStore};
use crate::providers::{Provider, RefreshContext, RefreshCtxState};
use crate::types::{
    ApiStreamOptions, AssistantMessage, AssistantMessageEvent, Context, CancellationToken, Model,
    SimpleStreamOptions,
};

pub struct Models {
    providers: RwLock<HashMap<String, Arc<dyn Provider>>>,
    model_index: RwLock<HashMap<(String, String), Model>>,
    provider_models: RwLock<HashMap<String, Vec<Model>>>,
    credential_store: Arc<dyn CredentialStore + Send + Sync>,
    auth_context: Arc<dyn AuthContext + Send + Sync>,
    models_store: Arc<dyn ModelsStore + Send + Sync>,
}

impl Default for Models {
    fn default() -> Self {
        Self::new()
    }
}

impl Models {
    pub fn new() -> Self {
        Self {
            providers: RwLock::new(HashMap::new()),
            model_index: RwLock::new(HashMap::new()),
            provider_models: RwLock::new(HashMap::new()),
            credential_store: Arc::new(InMemoryCredentialStore::new()),
            auth_context: Arc::new(DefaultAuthContext),
            models_store: Arc::new(InMemoryModelsStore::new()),
        }
    }

    pub fn with_credential_store(
        mut self,
        store: Arc<dyn CredentialStore + Send + Sync>,
    ) -> Self {
        self.credential_store = store;
        self
    }

    pub fn with_auth_context(
        mut self,
        ctx: Arc<dyn AuthContext + Send + Sync>,
    ) -> Self {
        self.auth_context = ctx;
        self
    }

    pub fn with_models_store(
        mut self,
        store: Arc<dyn ModelsStore + Send + Sync>,
    ) -> Self {
        self.models_store = store;
        self
    }

    pub fn set_provider<P: Provider + 'static>(&self, provider: P) {
        self.set_provider_arc(Arc::new(provider));
    }

    pub fn set_provider_arc(&self, provider: Arc<dyn Provider>) {
        let id = provider.id().to_string();
        self.providers.write().insert(id, provider);
    }

    pub fn get_provider(&self, provider_id: &str) -> Option<Arc<dyn Provider>> {
        self.providers.read().get(provider_id).cloned()
    }

    pub async fn refresh_provider_models(
        &self,
        provider_id: &str,
        force: bool,
        allow_network: bool,
    ) -> anyhow::Result<Vec<Model>> {
        let provider = self
            .get_provider(provider_id)
            .ok_or_else(|| anyhow::anyhow!("Provider {} not found", provider_id))?;

        let stored = self.models_store.read(provider_id).await.unwrap_or(None);
        let credential = self
            .credential_store
            .read(provider_id, crate::auth::types::AuthOperationOptions::default())
            .await
            .unwrap_or(None);

        let refresh_ctx_state = RefreshCtxState {
            credential,
            stored,
            allow_network,
            force,
            provider_id: provider_id.to_string(),
            models_store: self.models_store.clone(),
        };
        let refresh_ctx = RefreshContext {
            state: std::borrow::Cow::Owned(refresh_ctx_state),
        };

        provider.refresh_models(Box::new(refresh_ctx)).await
            .map_err(|e| anyhow::anyhow!(e))?;

        let models = provider.get_models();

        // Update indexes
        let mut provider_models_lock = self.provider_models.write();
        provider_models_lock.insert(provider_id.to_string(), models.clone());

        let mut index_lock = self.model_index.write();
        for m in &models {
            index_lock.insert((provider_id.to_string(), m.id.clone()), m.clone());
        }

        Ok(models)
    }

    pub async fn refresh_all(&self, force: bool, allow_network: bool) -> anyhow::Result<usize> {
        let ids: Vec<String> = self.providers.read().keys().cloned().collect();
        let mut total = 0;
        for id in ids {
            let models = self
                .refresh_provider_models(&id, force, allow_network)
                .await?;
            total += models.len();
        }
        Ok(total)
    }

    pub fn get_model(&self, provider_id: &str, model_id: &str) -> Option<Model> {
        self.model_index
            .read()
            .get(&(provider_id.to_string(), model_id.to_string()))
            .cloned()
            .or_else(|| {
                // Fallback: try to find in default provider models without refresh
                self.provider_models
                    .read()
                    .get(provider_id)
                    .and_then(|v| v.iter().find(|m| m.id == model_id).cloned())
            })
    }

    pub fn list_models(&self, provider_id: Option<&str>) -> Vec<Model> {
        match provider_id {
            Some(pid) => self
                .provider_models
                .read()
                .get(pid)
                .cloned()
                .unwrap_or_default(),
            None => self.provider_models.read().values().flatten().cloned().collect(),
        }
    }

    pub fn list_providers(&self) -> Vec<(String, String)> {
        self.providers
            .read()
            .iter()
            .map(|(id, p)| (id.clone(), p.name().to_string()))
            .collect()
    }

    pub async fn get_auth(
        &self,
        provider_id: &str,
        overrides: AuthResolutionOverrides,
    ) -> Result<crate::auth::types::AuthResult, ModelsError> {
        let provider = self
            .get_provider(provider_id)
            .ok_or_else(|| ModelsError {
                code: crate::auth::ModelsErrorCode::NotFound(format!(
                    "Provider {} not found",
                    provider_id
                )),
                source: None,
            })?;
        let signal = CancellationToken::new();
        resolve_provider_auth(
            provider.as_ref(),
            self.auth_context.as_ref(),
            self.credential_store.as_ref(),
            overrides,
            &signal,
        )
        .await
    }

    pub fn stream(
        &self,
        model: &Model,
        mut context: Context,
        options: SimpleStreamOptions,
    ) -> std::pin::Pin<
        Box<dyn futures::Stream<Item = AssistantMessageEvent> + Send + 'static>,
    > {
        let provider_id = model.provider_id.clone();
        let model = model.clone();
        let me = self.clone_ref();

        Box::pin(async_stream::stream! {
            let provider = match me.get_provider(&provider_id) {
                Some(p) => p,
                None => {
                    yield AssistantMessageEvent::Error {
                        reason: "provider-not-found".into(),
                        error: format!("Provider {} not found", provider_id),
                    };
                    return;
                }
            };

            // Merge temperature and max_tokens into context, if provided
            if options.temperature.is_some() {
                context.temperature = options.temperature;
            }
            if options.max_tokens.is_some() {
                context.max_tokens = options.max_tokens;
            }
            if options.response_format.is_some() {
                context.response_format = options.response_format.clone();
            }
            if options.tool_choice.is_some() {
                context.tool_choice = options.tool_choice.clone();
            }

            let auth_result = match me
                .get_auth(&provider_id, AuthResolutionOverrides::default())
                .await
            {
                Ok(a) => a,
                Err(e) => {
                    yield AssistantMessageEvent::Error { reason: "auth-error".into(), error: e.to_string() };
                    return;
                }
            };
            let _ = auth_result;

            let mut api_opts = ApiStreamOptions::default();
            api_opts.signal = options.signal.clone();
            provider.apply_simple_options(&context, &options, &mut api_opts);

            let provider_stream = provider.stream_simple(&model, context, options);
            futures::pin_mut!(provider_stream);
            while let Some(evt) = provider_stream.next().await {
                yield evt;
            }
        })
    }

    pub fn complete(
        &self,
        model: &Model,
        mut context: Context,
        options: SimpleStreamOptions,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<AssistantMessage>> + Send>>
    {
        let provider_id = model.provider_id.clone();
        let model = model.clone();
        let me = self.clone_ref();

        Box::pin(async move {
            let provider = me
                .get_provider(&provider_id)
                .ok_or_else(|| anyhow::anyhow!("Provider {} not found", provider_id))?;

            if options.temperature.is_some() {
                context.temperature = options.temperature;
            }
            if options.max_tokens.is_some() {
                context.max_tokens = options.max_tokens;
            }
            if options.response_format.is_some() {
                context.response_format = options.response_format.clone();
            }
            if options.tool_choice.is_some() {
                context.tool_choice = options.tool_choice.clone();
            }

            let auth_result = me
                .get_auth(&provider_id, AuthResolutionOverrides::default())
                .await
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            let _ = auth_result;

            let mut api_opts = ApiStreamOptions::default();
            api_opts.signal = options.signal.clone();
            provider.apply_simple_options(&context, &options, &mut api_opts);

            // complete via stream: consume events and return the Done message,
            // matching the behavior faux tests expect.
            use futures::StreamExt;
            let mut stream = provider.stream_simple(&model, context, options);
            let mut last_done: Option<AssistantMessage> = None;
            let mut last_error: Option<String> = None;
            while let Some(evt) = stream.next().await {
                match evt {
                    AssistantMessageEvent::Done { message, .. } => {
                        last_done = Some(message);
                    }
                    AssistantMessageEvent::Error { error, .. } => {
                        last_error = Some(error);
                    }
                    _ => {}
                }
            }
            if let Some(msg) = last_done {
                Ok(msg)
            } else if let Some(err) = last_error {
                Err(anyhow::anyhow!("{}", err))
            } else {
                Err(anyhow::anyhow!("Stream completed without Done or Error event"))
            }
        })
    }

    fn clone_ref(&self) -> ModelsRef {
        ModelsRef {
            providers: self.clone_providers_map(),
            model_index: self.clone_model_index(),
            provider_models: self.clone_provider_models(),
            credential_store: self.credential_store.clone(),
            auth_context: self.auth_context.clone(),
            models_store: self.models_store.clone(),
        }
    }

    fn clone_providers_map(&self) -> Arc<RwLock<HashMap<String, Arc<dyn Provider>>>> {
        // Workaround: we need to clone the inner struct fields; for a real impl
        // put all of `Models` behind an Arc. We'll do that via ModelsRef below.
        Arc::new(RwLock::new(self.providers.read().clone()))
    }
    fn clone_model_index(&self) -> Arc<RwLock<HashMap<(String, String), Model>>> {
        Arc::new(RwLock::new(self.model_index.read().clone()))
    }
    fn clone_provider_models(&self) -> Arc<RwLock<HashMap<String, Vec<Model>>>> {
        Arc::new(RwLock::new(self.provider_models.read().clone()))
    }
}

// A cheap cloneable reference to Models. Used internally by async stream/complete fns.
struct ModelsRef {
    providers: Arc<RwLock<HashMap<String, Arc<dyn Provider>>>>,
    model_index: Arc<RwLock<HashMap<(String, String), Model>>>,
    provider_models: Arc<RwLock<HashMap<String, Vec<Model>>>>,
    credential_store: Arc<dyn CredentialStore + Send + Sync>,
    auth_context: Arc<dyn AuthContext + Send + Sync>,
    models_store: Arc<dyn ModelsStore + Send + Sync>,
}

impl ModelsRef {
    fn get_provider(&self, provider_id: &str) -> Option<Arc<dyn Provider>> {
        self.providers.read().get(provider_id).cloned()
    }

    async fn get_auth(
        &self,
        provider_id: &str,
        overrides: AuthResolutionOverrides,
    ) -> Result<crate::auth::types::AuthResult, ModelsError> {
        let provider = self.get_provider(provider_id).ok_or_else(|| ModelsError {
            code: crate::auth::ModelsErrorCode::NotFound(format!(
                "Provider {} not found",
                provider_id
            )),
            source: None,
        })?;
        let signal = CancellationToken::new();
        resolve_provider_auth(
            provider.as_ref(),
            self.auth_context.as_ref(),
            self.credential_store.as_ref(),
            overrides,
            &signal,
        )
        .await
    }
}

pub fn create_models() -> Models {
    Models::new()
}
