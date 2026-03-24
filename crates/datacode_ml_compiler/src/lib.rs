//! Parameter name lists for ML-related native/method calls used by the `data-code` compiler
//! (`resolve_function_args`). Source of truth: `../../compiler/ml_native_named_args.json`.

use serde::Deserialize;
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Deserialize)]
struct MlNativeNamedArgsFile {
    param_lists: HashMap<String, Vec<String>>,
    aliases: HashMap<String, String>,
}

/// Returns ordered parameter names for ML compiler metadata (train, layer freeze, etc.), if defined.
pub fn native_named_arg_params(function_name: &str) -> Option<Vec<String>> {
    static CACHE: OnceLock<MlNativeNamedArgsFile> = OnceLock::new();
    let file = CACHE.get_or_init(|| {
        const JSON_STR: &str = include_str!("../../../compiler/ml_native_named_args.json");
        serde_json::from_str(JSON_STR).expect("ml_native_named_args.json must be valid JSON")
    });
    let key = file
        .aliases
        .get(function_name)
        .map(String::as_str)
        .unwrap_or(function_name);
    file.param_lists.get(key).cloned()
}
