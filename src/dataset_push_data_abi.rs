//! Named arguments for [`crate::vm::natives::native_dataset_push_data`] / `dataset.push_data(...)`.
//!
//! Keep in sync with `crates/datacode_ml_compiler/ml_native_named_args.json` key `native_dataset_push_data`
//! and `native_dataset_push_data` `args[1..=2]` in the plugin.

/// Ordered names for kwargs after the dataset receiver.
pub const DATASET_PUSH_DATA_NAMED_ARG_NAMES: &[&str] = &["features", "targets"];

#[cfg(test)]
mod tests {
    use super::DATASET_PUSH_DATA_NAMED_ARG_NAMES;
    use datacode_ml_compiler::native_dataset_push_data_kwarg_param_names;

    #[test]
    fn named_args_match_datacode_ml_compiler_json() {
        assert_eq!(
            DATASET_PUSH_DATA_NAMED_ARG_NAMES,
            native_dataset_push_data_kwarg_param_names(),
            "keep in sync with crates/datacode_ml_compiler/ml_native_named_args.json",
        );
    }
}
