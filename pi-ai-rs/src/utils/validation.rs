use schemars::{schema_for, JsonSchema};
use serde_json::Value;

use crate::types::{Model, Tool, UsageCost};

pub fn tool_from_schema<T: JsonSchema>(
    name: impl Into<String>,
    description: impl Into<Option<String>>,
) -> Tool {
    let schema_value = serde_json::to_value(&schema_for!(T).schema)
        .unwrap_or_else(|_| Value::Object(Default::default()));
    Tool {
        name: name.into(),
        description: description.into(),
        parameters: schema_value,
    }
}

pub fn string_enum_schema(
    values: &[&str],
    description: Option<&str>,
    default: Option<&str>,
) -> serde_json::Value {
    let enum_vals: Vec<Value> = values
        .iter()
        .map(|s| Value::String(s.to_string()))
        .collect();
    let mut obj = serde_json::Map::new();
    obj.insert("type".to_string(), Value::String("string".to_string()));
    obj.insert("enum".to_string(), Value::Array(enum_vals));
    if let Some(desc) = description {
        obj.insert("description".to_string(), Value::String(desc.to_string()));
    }
    if let Some(def) = default {
        obj.insert("default".to_string(), Value::String(def.to_string()));
    }
    Value::Object(obj)
}

pub fn calculate_usage_cost(
    input: u64,
    output: u64,
    cache_read: u64,
    cache_write: u64,
    model: &Model,
) -> UsageCost {
    let rates = &model.cost_rates;
    let input_cost = rates
        .input_per_million
        .map(|r| (input as f64 / 1_000_000.0) * r)
        .unwrap_or(0.0);
    let output_cost = rates
        .output_per_million
        .map(|r| (output as f64 / 1_000_000.0) * r)
        .unwrap_or(0.0);
    let cache_read_cost = rates
        .input_cache_read_per_million
        .map(|r| (cache_read as f64 / 1_000_000.0) * r)
        .unwrap_or(0.0);
    let cache_write_cost = rates
        .input_cache_write_per_million
        .map(|r| (cache_write as f64 / 1_000_000.0) * r)
        .unwrap_or(0.0);
    UsageCost {
        input: input_cost,
        output: output_cost,
        cache_read: cache_read_cost,
        cache_write: cache_write_cost,
        total: input_cost + output_cost + cache_read_cost + cache_write_cost,
    }
}

pub fn validate_tool_arguments(
    _tool: &Tool,
    _arguments: &serde_json::Value,
) -> Result<(), Vec<String>> {
    // Basic placeholder: skip rigorous JSON Schema validation for now.
    // A real implementation could use a crate like `jsonschema`.
    Ok(())
}
