//! Radius — a dynamic `pi-messages` gateway.  `/login radius` stores OAuth
//! tokens in `auth.json`; the gateway catalog is refreshed independently and
//! cached in `models-store.json`.  Custom Radius gateways can be declared in
//! `models.json` with `"oauth": "radius"` and a gateway `baseUrl`.

use super::oauth_compat::{build_oauth_provider, GenericOAuthProvider, OAuthProviderSpec};

pub type RadiusProvider = GenericOAuthProvider;

pub fn radius_provider() -> RadiusProvider {
    build_oauth_provider(OAuthProviderSpec {
        id: "radius",
        display_name: "Radius",
        base_url: "https://api.radius.ai/v1",
        // TODO: verify client_id / endpoints from pi-ai source.
        client_id: "radius-cli",
        scopes: &["openid", "profile", "offline_access"],
        auth_url: Some("https://auth.radius.ai/authorize"),
        token_url: Some("https://auth.radius.ai/oauth/token"),
        device_auth_url: None,
        redirect_uri: Some("http://localhost:1455/callback"),
        is_subscription: false,
        login_label: Some("Radius"),
        api_label: "pi-messages",
        default_models_fn: default_models,
        extra_headers: None,
    })
}

fn default_models() -> Vec<crate::types::Model> {
    use crate::types::{Model, ModelCostRates, ModelThinkingLevel};
    // Radius is a gateway — the actual model catalogue is fetched at
    // runtime via refresh_models().  Ship a minimal placeholder so the
    // provider is always usable even before the first refresh.
    vec![Model {
        id: "radius-auto".into(),
        provider_id: "radius".into(),
        name: "Radius Auto (gateway-selected)".into(),
        api: "pi-messages".into(),
        max_input_tokens: Some(128_000),
        max_output_tokens: Some(32_768),
        supports_images: true,
        supports_audio: false,
        supports_video: false,
        supports_pdf: true,
        supports_tool_calling: true,
        supports_structured_output: true,
        supports_system_prompt: true,
        thinking: ModelThinkingLevel::None,
        reasoning: false,
        cost_rates: ModelCostRates::default(),
        context_window: Some(128_000),
        default_temperature: Some(1.0),
        thinking_level_map: None,
    }]
}
