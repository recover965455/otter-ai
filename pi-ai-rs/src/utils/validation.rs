use schemars::JsonSchema;
use serde::Serialize;

use crate::types::{CostTier, Model, Usage, UsageCost};

// --- Cost calculation (TS: models.ts calculateCost) with tier support ---
/// Compute the cost for a Usage based on a model's cost rates, applying
/// price tiers when `cacheWrite + input + cacheRead` exceeds the
/// tier's `input_tokens_above` threshold. This matches the TS behavior:
/// tier is selected based on whether total tokens minus output exceed the
/// configured threshold; each rate is a "per million tokens" price, so we
/// divide the raw token counts by 1_000_000.
pub fn calculate_usage_cost(
    input_tokens: u64,
    output_tokens: u64,
    cache_read_tokens: u64,
    cache_write_tokens: u64,
    model: &Model,
) -> UsageCost {
    let usage = Usage {
        input: input_tokens,
        output: output_tokens,
        cache_read: cache_read_tokens,
        cache_write: cache_write_tokens,
        total_tokens: input_tokens + output_tokens + cache_read_tokens + cache_write_tokens,
        cost: UsageCost::default(),
    };
    calculate_cost(model, &usage)
}

pub fn calculate_cost(model: &Model, usage: &Usage) -> UsageCost {
    let non_output_total = usage.input + usage.cache_read + usage.cache_write;

    // Anchor fallback using base rates; then tiers override based on the
    // `inputTokensAbove` threshold, with higher thresholds winning on tie.
    // This matches the TS `faux-provider.test.ts` tier-selection semantics.
    let fallback = CostTier {
        input_tokens_above: 0,
        input_per_million: model.cost_rates.input_per_million.unwrap_or(0.0),
        output_per_million: model.cost_rates.output_per_million.unwrap_or(0.0),
        cache_read_per_million: model.cost_rates.input_cache_read_per_million.unwrap_or(0.0),
        cache_write_per_million: model.cost_rates.input_cache_write_per_million.unwrap_or(0.0),
    };
    let mut effective: &CostTier = &fallback;
    let mut sorted_tiers: Vec<&CostTier> = model.cost_rates.tiers.iter().collect();
    sorted_tiers.sort_by_key(|t| t.input_tokens_above);
    for tier in sorted_tiers {
        if non_output_total > tier.input_tokens_above {
            effective = tier;
        }
    }

    let per_million = |tokens: u64, rate: f64| (tokens as f64) * rate / 1_000_000.0;
    let input_cost = per_million(usage.input, effective.input_per_million);
    let output_cost = per_million(usage.output, effective.output_per_million);
    let cache_read_cost = per_million(usage.cache_read, effective.cache_read_per_million);
    let cache_write_cost = per_million(usage.cache_write, effective.cache_write_per_million);
    UsageCost {
        input: input_cost,
        output: output_cost,
        cache_read: cache_read_cost,
        cache_write: cache_write_cost,
        total: input_cost + output_cost + cache_read_cost + cache_write_cost,
    }
}

/// Create a `Tool` definition from a JSON-Schema-generic Rust struct using
/// `schemars`. Mirrors the TS `Type.Object(...)` / `Type.String()` pattern
/// in the faux-provider test suite.
pub fn tool_from_schema<T: JsonSchema + Serialize>(
    name: &str,
    description: Option<String>,
) -> crate::types::Tool {
    let schema = schemars::schema_for!(T);
    let value = serde_json::to_value(&schema.schema).unwrap_or(serde_json::Value::Object(
        serde_json::Map::new(),
    ));
    crate::types::Tool {
        name: name.to_string(),
        description,
        parameters: value,
    }
}

/// Build a JSON `{ type: "string", enum: [...], description?, default? }`
/// schema, matching the TS usage for string enum parameters.
pub fn string_enum_schema(
    values: &[&str],
    description: Option<&str>,
    default: Option<&str>,
) -> serde_json::Value {
    use serde_json::json;
    let mut map = serde_json::Map::new();
    map.insert("type".into(), json!("string"));
    map.insert(
        "enum".into(),
        json!(values.iter().cloned().collect::<Vec<_>>()),
    );
    if let Some(desc) = description {
        map.insert("description".into(), json!(desc));
    }
    if let Some(def) = default {
        map.insert("default".into(), json!(def));
    }
    serde_json::Value::Object(map)
}

/// Very light validation: a tool arguments payload must be a JSON object.
pub fn validate_tool_arguments(
    _tool: &crate::types::Tool,
    arguments: &serde_json::Value,
) -> Result<(), String> {
    if !arguments.is_object() {
        return Err("tool arguments must be a JSON object".into());
    }
    Ok(())
}
