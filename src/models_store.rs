use crate::types::Model;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelsStoreEntry {
    #[serde(default)]
    pub models: Vec<Model>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fetched_at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub etag: Option<String>,
}

#[async_trait::async_trait]
pub trait ModelsStore: Send + Sync {
    async fn read(&self, provider_id: &str) -> anyhow::Result<Option<ModelsStoreEntry>>;
    async fn write(&self, provider_id: &str, entry: &ModelsStoreEntry) -> anyhow::Result<()>;
    async fn delete(&self, provider_id: &str) -> anyhow::Result<()>;
}

#[derive(Clone, Default)]
pub struct InMemoryModelsStore {
    inner: Arc<Mutex<HashMap<String, ModelsStoreEntry>>>,
}

impl InMemoryModelsStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait::async_trait]
impl ModelsStore for InMemoryModelsStore {
    async fn read(&self, provider_id: &str) -> anyhow::Result<Option<ModelsStoreEntry>> {
        Ok(self.inner.lock().get(provider_id).cloned())
    }

    async fn write(&self, provider_id: &str, entry: &ModelsStoreEntry) -> anyhow::Result<()> {
        self.inner
            .lock()
            .insert(provider_id.to_string(), entry.clone());
        Ok(())
    }

    async fn delete(&self, provider_id: &str) -> anyhow::Result<()> {
        self.inner.lock().remove(provider_id);
        Ok(())
    }
}
