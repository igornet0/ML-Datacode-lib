//! Named-argument lists for **compile-time** resolution in the host `data-code` compiler (`resolve_function_args`).
//! Bundles `ml_native_named_args.json` in this crate. Native runtime (`ml` dylib) links only `datacode_abi` /
//! `datacode_sdk`, not this crate.

use serde::Deserialize;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Key in `ml_native_named_args.json` for `dataset.split` / `native_dataset_split` kwargs (compile-time).
pub const NATIVE_DATASET_SPLIT_PARAM_LIST_KEY: &str = "native_dataset_split";
/// Key for `dataset.push_data` / `native_dataset_push_data` kwargs.
pub const NATIVE_DATASET_PUSH_DATA_PARAM_LIST_KEY: &str = "native_dataset_push_data";
/// Key for `dataset.concat` / `native_dataset_concat` kwargs.
pub const NATIVE_DATASET_CONCAT_PARAM_LIST_KEY: &str = "native_dataset_concat";

#[derive(Deserialize)]
struct MlNativeNamedArgsFile {
    param_lists: HashMap<String, Vec<String>>,
    aliases: HashMap<String, String>,
}

/// Returns ordered parameter names for ML-related compiler metadata (`train`, `freeze`, …), if defined.
pub fn native_named_arg_params(function_name: &str) -> Option<Vec<String>> {
    static CACHE: OnceLock<MlNativeNamedArgsFile> = OnceLock::new();
    let file = CACHE.get_or_init(|| {
        const JSON_STR: &str = include_str!("../ml_native_named_args.json");
        serde_json::from_str(JSON_STR).expect("ml_native_named_args.json must be valid JSON")
    });
    let key = file
        .aliases
        .get(function_name)
        .map(String::as_str)
        .unwrap_or(function_name);
    file.param_lists.get(key).cloned()
}

/// Ordered kwarg names for `dataset.split` / `native_dataset_split` (receiver + these six in order).
/// Matches the plugin’s `native_dataset_split` `args[1..=6]` and the VM ABI bridge.
pub fn native_dataset_split_kwarg_param_names() -> &'static [&'static str] {
    static NAMES: OnceLock<&'static [&'static str]> = OnceLock::new();
    *NAMES.get_or_init(|| {
        let v: Vec<String> = native_named_arg_params(NATIVE_DATASET_SPLIT_PARAM_LIST_KEY)
            .expect("ml_native_named_args.json must define native_dataset_split");
        let arr: Vec<&'static str> = v
            .into_iter()
            .map(|s| Box::leak(s.into_boxed_str()) as &'static str)
            .collect();
        let slice: &'static mut [&'static str] = Box::leak(arr.into_boxed_slice());
        &*slice
    })
}

/// Ordered kwarg names for `dataset.push_data` / `native_dataset_push_data`.
pub fn native_dataset_push_data_kwarg_param_names() -> &'static [&'static str] {
    static NAMES: OnceLock<&'static [&'static str]> = OnceLock::new();
    *NAMES.get_or_init(|| {
        let v: Vec<String> = native_named_arg_params(NATIVE_DATASET_PUSH_DATA_PARAM_LIST_KEY)
            .expect("ml_native_named_args.json must define native_dataset_push_data");
        let arr: Vec<&'static str> = v
            .into_iter()
            .map(|s| Box::leak(s.into_boxed_str()) as &'static str)
            .collect();
        let slice: &'static mut [&'static str] = Box::leak(arr.into_boxed_slice());
        &*slice
    })
}
