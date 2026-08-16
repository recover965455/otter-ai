//! FileCredentialStore tests — loose-format parsing, alias lookup and
//! read/modify/write round-trips against a temp credentials file.

use otter_ai::auth::{
    parse_loose_credential, AuthOperationOptions, Credential, CredentialStore, FileCredentialStore,
    ModifyFnOutput,
};
use serde_json::json;

fn temp_store(tag: &str) -> FileCredentialStore {
    let dir = std::env::temp_dir().join(format!("otter-cred-test-{}-{}", tag, std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    FileCredentialStore::with_path(dir.join("credentials.json"))
}

#[test]
fn parses_loose_api_and_oauth_entries() {
    let api = parse_loose_credential(&json!({
        "type": "api",
        "key": "sk-123",
    }))
    .expect("api entry parses");
    match api {
        Credential::ApiKey(k) => assert_eq!(k.key.as_deref(), Some("sk-123")),
        other => panic!("expected api key, got {:?}", other),
    }

    let oauth = parse_loose_credential(&json!({
        "type": "oauth",
        "access_token": "at",
        "refresh_token": "rt",
        "expires_at": "2026-08-21T06:48:47Z",
        "chatgpt_account_id": "acc-1",
        "scope": "openid",
    }))
    .expect("oauth entry parses");
    match oauth {
        Credential::OAuth(o) => {
            assert_eq!(o.inner.access, "at");
            assert_eq!(o.inner.refresh, "rt");
            assert_eq!(o.inner.expires, 1_787_294_927_000);
            assert_eq!(
                o.inner.extra.get("account_id").and_then(|v| v.as_str()),
                Some("acc-1")
            );
        }
        other => panic!("expected oauth, got {:?}", other),
    }
}

#[tokio::test]
async fn round_trips_credentials_through_the_file() {
    let store = temp_store("roundtrip");
    assert!(store
        .read("openai", AuthOperationOptions::default())
        .await
        .unwrap()
        .is_none());

    // Write through modify_fn with a loose (hand-written) file on disk first.
    std::fs::write(
        store.path(),
        serde_json::to_string_pretty(&json!({
            "openai": {
                "type": "oauth",
                "access_token": "at-1",
                "refresh_token": "rt-1",
                "expires_at": "2030-01-01T00:00:00Z",
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let read = store
        .read("openai", AuthOperationOptions::default())
        .await
        .unwrap()
        .expect("entry exists");
    match read {
        Credential::OAuth(o) => {
            assert_eq!(o.inner.access, "at-1");
            assert!(o.inner.expires > 1_893_000_000_000);
        }
        other => panic!("expected oauth, got {:?}", other),
    }

    // Replace via modify_fn → persisted back in the native format.
    let new_cred = Credential::api_key("sk-new");
    let new_cred_clone = new_cred.clone();
    store
        .modify_fn(
            "deepseek",
            Box::new(move |_| Box::pin(async move { Ok(Some(new_cred_clone)) }) as ModifyFnOutput),
            AuthOperationOptions::default(),
        )
        .await
        .unwrap();

    let after = store
        .read("deepseek", AuthOperationOptions::default())
        .await
        .unwrap()
        .expect("replaced entry exists");
    assert_eq!(
        match after {
            Credential::ApiKey(k) => k.key,
            other => panic!("expected api key, got {:?}", other),
        }
        .as_deref(),
        Some("sk-new")
    );

    let listed = store.list(AuthOperationOptions::default()).await.unwrap();
    assert!(listed.iter().any(|i| i.provider_id == "openai"));
    assert!(listed.iter().any(|i| i.provider_id == "deepseek"));
}

#[tokio::test]
async fn chatgpt_plus_lookups_fall_back_to_the_openai_key() {
    let store = temp_store("alias");
    std::fs::write(
        store.path(),
        serde_json::to_string(&json!({
            "openai": {
                "type": "api",
                "key": "sk-alias",
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let direct = store
        .read("chatgpt-plus", AuthOperationOptions::default())
        .await
        .unwrap()
        .expect("alias lookup finds the openai entry");
    assert!(matches!(direct, Credential::ApiKey(_)));

    // Unknown providers still resolve to None.
    assert!(store
        .read("nope", AuthOperationOptions::default())
        .await
        .unwrap()
        .is_none());

    // Deleting the aliased provider must not clobber the `openai` entry.
    store
        .delete("chatgpt-plus", AuthOperationOptions::default())
        .await
        .unwrap();
    assert!(store
        .read("openai", AuthOperationOptions::default())
        .await
        .unwrap()
        .is_some());
}

#[cfg(unix)]
#[test]
fn credentials_file_gets_owner_only_permissions() {
    let store = temp_store("perms");
    std::fs::write(store.path(), "{}").unwrap();
    store_write_again(&store);
    use std::os::unix::fs::PermissionsExt;
    let mode = std::fs::metadata(store.path())
        .unwrap()
        .permissions()
        .mode();
    assert_eq!(mode & 0o777, 0o600);
}

fn store_write_again(store: &FileCredentialStore) {
    // Trigger a write through the public API so permissions get applied.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let cred = Credential::api_key("sk-perm");
        store
            .modify_fn(
                "x",
                Box::new(move |_| Box::pin(async move { Ok(Some(cred)) }) as ModifyFnOutput),
                AuthOperationOptions::default(),
            )
            .await
            .unwrap();
    });
}
