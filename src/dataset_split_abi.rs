//! Named arguments for [`crate::vm::natives::native_dataset_split`] / `dataset.split(...)` on the VM bridge.
//!
//! The host compiler resolves kwargs using the `datacode_ml_compiler` crate (this repo:
//! `crates/datacode_ml_compiler`, JSON key `native_dataset_split`). This slice must stay aligned
//! with that file and with `args[1..=6]` in `native_dataset_split`.
//!
//! ## Omitted kwargs and `Null`
//!
//! Positional slots for named parameters are filled in order; any argument not supplied in source
//! is passed as [`crate::vm_value::Value::Null`] (see the host’s `resolve_function_args`). For
//! `test_size`, `train_size`, and `random_state`, `Null` means “unspecified” and is handled like
//! today. For the three boolean options below, `Null` means “use the default” (sklearn-like):
//! `shuffle` → true, `stratify` → false, `return_indices` → false.
//!
//! ## `test_size` / `train_size` as numbers
//!
//! Non-null values are passed as `Value::Number`. Interpretation matches
//! [`crate::dataset::Dataset::split`]: strictly between 0 and 1 means a **fraction**; a whole
//! number ≥ 1 means an **absolute sample count** (and must be less than `n`).

/// Default for `shuffle` when the slot is `Null` (omitted kwarg).
pub const DEFAULT_SPLIT_SHUFFLE: bool = true;
/// Default for `stratify` when the slot is `Null` (omitted kwarg).
pub const DEFAULT_SPLIT_STRATIFY: bool = false;
/// Default for `return_indices` when the slot is `Null` (omitted kwarg).
pub const DEFAULT_SPLIT_RETURN_INDICES: bool = false;

/// Ordered names for kwargs after the dataset receiver (matches VM `native_dataset_split` and compiler JSON).
pub const DATASET_SPLIT_NAMED_ARG_NAMES: &[&str] = &[
    "test_size",
    "train_size",
    "shuffle",
    "random_state",
    "stratify",
    "return_indices",
];

#[cfg(test)]
mod tests {
    use super::DATASET_SPLIT_NAMED_ARG_NAMES;
    use datacode_ml_compiler::native_dataset_split_kwarg_param_names;

    #[test]
    fn named_args_match_datacode_ml_compiler_json() {
        assert_eq!(
            DATASET_SPLIT_NAMED_ARG_NAMES,
            native_dataset_split_kwarg_param_names(),
            "keep in sync with crates/datacode_ml_compiler/ml_native_named_args.json",
        );
    }
}
