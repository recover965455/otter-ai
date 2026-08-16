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
        cache_write_per_million: model
            .cost_rates
            .input_cache_write_per_million
            .unwrap_or(0.0),
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
    let value = serde_json::to_value(&schema.schema)
        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
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

/// Validate and coerce tool arguments against the tool's JSON-Schema parameters.
///
/// Mirrors the AJV-compatible coercion rules from the TS `validateToolArguments`:
/// - `number`/`integer`: "42"→42, true→1, null→0, "42.1"→42.1 (integer rejects)
/// - `boolean`: "true"→true, "false"→false, 1→true, 0→false, "1"/"0"→reject
/// - `string`: null→"", true→"true"
/// - `null`: ""→null, 0→null, false→null, "null"→reject
/// - type arrays: try each type in order, first match wins
/// - `anyOf`/`oneOf`: try each arm; if value already matches, preserve; else coerce
/// - optional non-nullable properties with null value → omit (strip)
/// - nullable properties (anyOf/oneOf with null arm) → preserve null
pub fn validate_tool_arguments(
    tool: &crate::types::Tool,
    arguments: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let schema = &tool.parameters;
    let mut args = arguments.clone();

    // Top-level must be an object
    if !args.is_object() {
        return Err("Validation failed: arguments must be an object".into());
    }

    coerce_object(schema, &mut args)?;
    Ok(args)
}

fn coerce_object(schema: &serde_json::Value, value: &mut serde_json::Value) -> Result<(), String> {
    if !value.is_object() {
        return Ok(());
    }

    let obj = value.as_object_mut().unwrap();
    let props = schema.get("properties").and_then(|p| p.as_object());
    let required: Vec<String> = schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let defs = schema
        .get("$defs")
        .or_else(|| schema.get("definitions"))
        .and_then(|d| d.as_object());

    if let Some(props) = props {
        let keys: Vec<String> = obj.keys().cloned().collect();
        for key in keys {
            let prop_schema = props.get(&key);
            if let Some(ps) = prop_schema {
                let val = obj.get_mut(&key).unwrap();
                let is_required = required.contains(&key);

                // Resolve $ref
                let resolved = resolve_ref(ps, defs);

                // Check if schema is nullable (has null in anyOf/oneOf or type array)
                let nullable = is_nullable(resolved);

                // If value is null and property is optional and not nullable → omit
                if val.is_null() && !is_required && !nullable {
                    obj.remove(&key);
                    continue;
                }

                // Coerce the value
                match coerce_value(resolved, val) {
                    Ok(()) => {}
                    Err(e) => return Err(format!("Validation failed: {}: {}", key, e)),
                }
            }
        }
    }

    Ok(())
}

fn resolve_ref<'a>(
    schema: &'a serde_json::Value,
    defs: Option<&'a serde_json::Map<String, serde_json::Value>>,
) -> &'a serde_json::Value {
    if let Some(ref_str) = schema.get("$ref").and_then(|r| r.as_str()) {
        // #/$defs/name or #/definitions/name
        let name = ref_str.rsplit('/').next().unwrap_or("");
        if let Some(defs_map) = defs {
            if let Some(resolved) = defs_map.get(name) {
                return resolve_ref(resolved, defs);
            }
        }
    }
    schema
}

fn is_nullable(schema: &serde_json::Value) -> bool {
    // type: "null" or type: ["...", "null"]
    if let Some(t) = schema.get("type") {
        match t {
            serde_json::Value::String(s) if s == "null" => return true,
            serde_json::Value::Array(arr) => {
                if arr.iter().any(|v| v.as_str() == Some("null")) {
                    return true;
                }
            }
            _ => {}
        }
    }
    // anyOf / oneOf with null arm
    for kw in &["anyOf", "oneOf"] {
        if let Some(arr) = schema.get(kw).and_then(|v| v.as_array()) {
            for arm in arr {
                if is_nullable(arm) {
                    return true;
                }
            }
        }
    }
    false
}

fn coerce_value(schema: &serde_json::Value, value: &mut serde_json::Value) -> Result<(), String> {
    // Handle anyOf
    if let Some(arr) = schema.get("anyOf").and_then(|v| v.as_array()) {
        return coerce_union(arr, value, false);
    }
    // Handle oneOf
    if let Some(arr) = schema.get("oneOf").and_then(|v| v.as_array()) {
        return coerce_union(arr, value, true);
    }

    let types = get_types(schema);

    // If no type specified, accept as-is (but still recurse into objects
    // that declare their own `properties`, matching TS behavior).
    if types.is_empty() {
        if value.is_object() && schema.get("properties").is_some() {
            return coerce_object(schema, value);
        }
        return Ok(());
    }

    // First, if the value already matches a type as-is, preserve it.
    // This mirrors AJV: a value matching any declared type is kept unchanged
    // (e.g. `type: ["number","string"]` with `"1"` stays `"1"`).
    // Only when no type matches do we attempt coercion.
    for t in &types {
        if matches_type(t, value) {
            if t == "object" && value.is_object() && schema.get("properties").is_some() {
                return coerce_object(schema, value);
            }
            return Ok(());
        }
    }

    // No type matches as-is; try coercion in order, first that succeeds wins
    for t in &types {
        if try_coerce_single(t, value)? {
            // After a successful object match, recurse into nested properties
            // so optional non-nullable nulls are stripped recursively.
            if t == "object" && value.is_object() && schema.get("properties").is_some() {
                return coerce_object(schema, value);
            }
            return Ok(());
        }
    }

    Err(format!(
        "value {:?} does not match any of {:?}",
        value, types
    ))
}

fn get_types(schema: &serde_json::Value) -> Vec<String> {
    match schema.get("type") {
        Some(serde_json::Value::String(s)) => vec![s.clone()],
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => vec![],
    }
}

fn coerce_union(
    arms: &[serde_json::Value],
    value: &mut serde_json::Value,
    _is_one_of: bool,
) -> Result<(), String> {
    // If value already matches an arm, preserve it
    for arm in arms {
        let types = get_types(arm);
        if types.iter().any(|t| matches_type(t, value)) {
            return Ok(());
        }
    }

    // Try to coerce into each arm
    for arm in arms {
        let types = get_types(arm);
        for t in &types {
            if t == "null" {
                continue; // don't coerce into null, only preserve
            }
            let mut tmp = value.clone();
            if try_coerce_single(t, &mut tmp)? {
                *value = tmp;
                return Ok(());
            }
        }
    }

    Err(format!("value {:?} does not match any union arm", value))
}

fn matches_type(type_str: &str, value: &serde_json::Value) -> bool {
    match type_str {
        "number" | "integer" => value.is_number(),
        "boolean" => value.is_boolean(),
        "string" => value.is_string(),
        "null" => value.is_null(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => false,
    }
}

/// Build a JSON number from an f64, preferring an integer representation
/// when the value is a finite whole number. This matches the TS behavior
/// where `42` and `42.0` are indistinguishable, so the Rust coercion yields
/// the integer form expected by tests like `json!(42)`.
fn json_number_from_f64(n: f64) -> serde_json::Value {
    if n.is_finite() && n.fract() == 0.0 && n >= i64::MIN as f64 && n <= i64::MAX as f64 {
        serde_json::Value::from(n as i64)
    } else {
        serde_json::Value::from(n)
    }
}

fn try_coerce_single(type_str: &str, value: &mut serde_json::Value) -> Result<bool, String> {
    match type_str {
        "number" => {
            *value = match value {
                serde_json::Value::String(s) => match s.parse::<f64>() {
                    Ok(n) => json_number_from_f64(n),
                    Err(_) => return Ok(false),
                },
                serde_json::Value::Bool(true) => serde_json::json!(1),
                serde_json::Value::Bool(false) => serde_json::json!(0),
                serde_json::Value::Null => serde_json::json!(0),
                serde_json::Value::Number(_) => return Ok(true),
                _ => return Ok(false),
            };
            Ok(true)
        }
        "integer" => {
            *value = match value {
                serde_json::Value::String(s) => match s.parse::<i64>() {
                    Ok(n) => serde_json::json!(n),
                    Err(_) => return Ok(false),
                },
                serde_json::Value::Bool(true) => serde_json::json!(1),
                serde_json::Value::Bool(false) => serde_json::json!(0),
                serde_json::Value::Null => serde_json::json!(0),
                serde_json::Value::Number(n) => {
                    if n.is_i64() || n.is_u64() {
                        return Ok(true);
                    }
                    return Ok(false);
                }
                _ => return Ok(false),
            };
            Ok(true)
        }
        "boolean" => {
            *value = match value {
                serde_json::Value::String(s) => match s.as_str() {
                    "true" => serde_json::json!(true),
                    "false" => serde_json::json!(false),
                    _ => return Ok(false),
                },
                serde_json::Value::Number(n) => {
                    if n.as_f64() == Some(1.0) {
                        serde_json::json!(true)
                    } else if n.as_f64() == Some(0.0) {
                        serde_json::json!(false)
                    } else {
                        return Ok(false);
                    }
                }
                serde_json::Value::Bool(_) => return Ok(true),
                _ => return Ok(false),
            };
            Ok(true)
        }
        "string" => {
            *value = match value {
                serde_json::Value::Null => serde_json::json!(""),
                serde_json::Value::Bool(b) => serde_json::json!(b.to_string()),
                serde_json::Value::String(_) => return Ok(true),
                _ => return Ok(false),
            };
            Ok(true)
        }
        "null" => {
            *value = match value {
                serde_json::Value::String(s) if s.is_empty() => serde_json::json!(null),
                serde_json::Value::Number(n) if n.as_f64() == Some(0.0) => serde_json::json!(null),
                serde_json::Value::Bool(false) => serde_json::json!(null),
                serde_json::Value::Null => return Ok(true),
                _ => return Ok(false),
            };
            Ok(true)
        }
        "array" => {
            // Don't coerce, just accept if already array
            Ok(value.is_array())
        }
        "object" => Ok(value.is_object()),
        _ => Ok(true), // unknown type, accept
    }
}
