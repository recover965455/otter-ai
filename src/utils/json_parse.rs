//! Partial JSON parsing utilities for streaming tool call arguments.

use serde_json::Value;

/// Parses a partial JSON string, filling missing parts with defaults.
/// Returns Ok(Some(value)) when the string parses fully or partially.
pub fn parse_partial_json(input: &str) -> Result<Option<Value>, serde_json::Error> {
    match serde_json::from_str::<Value>(input) {
        Ok(v) => Ok(Some(v)),
        Err(_) => {
            for strategy in [complete_object, complete_array, add_quoted_string_tail] {
                let completed = strategy(input);
                if let Ok(v) = serde_json::from_str::<Value>(&completed) {
                    return Ok(Some(v));
                }
            }
            Ok(None)
        }
    }
}

fn complete_object(s: &str) -> String {
    let mut result = s.to_string();
    let mut brace_count = 0i32;
    let mut bracket_count = 0i32;
    let mut in_string = false;
    let mut escape = false;

    for c in s.chars() {
        match (escape, in_string, c) {
            (true, _, _) => escape = false,
            (false, _, '\\') => escape = true,
            (false, true, '"') => in_string = false,
            (false, false, '"') => in_string = true,
            (false, false, '{') => brace_count += 1,
            (false, false, '}') => brace_count -= 1,
            (false, false, '[') => bracket_count += 1,
            (false, false, ']') => bracket_count -= 1,
            _ => {}
        }
    }

    while result.ends_with(',') || result.ends_with(':') || result.ends_with(' ') {
        result.pop();
    }

    if in_string {
        result.push('"');
    }

    while bracket_count > 0 {
        result.push(']');
        bracket_count -= 1;
    }
    while brace_count > 0 {
        result.push('}');
        brace_count -= 1;
    }

    result
}

fn complete_array(s: &str) -> String {
    complete_object(s)
}

fn add_quoted_string_tail(s: &str) -> String {
    let mut r = s.to_string();
    if !r.ends_with('"') && !r.ends_with('}') && !r.ends_with(']') {
        r.push('"');
    }
    complete_object(&r)
}
