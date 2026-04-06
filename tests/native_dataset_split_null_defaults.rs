//! `native_dataset_split`: omitted bool kwargs arrive as `Value::Null` from the host; must not error.

use ml::dataset::Dataset;
use ml::ml_types::MlValueKind;
use ml::natives::native_dataset_split;
use ml::runtime::dataset_to_value;
use ml::tensor::Tensor;
use ml::PluginValue as Value;

fn small_dataset() -> Dataset {
    let f = Tensor::new((0..18).map(|i| i as f32).collect(), vec![9, 2]).unwrap();
    let t = Tensor::new(vec![0.0; 9], vec![9, 1]).unwrap();
    Dataset::from_tensors(f, t).unwrap()
}

#[test]
fn split_all_optional_bools_null_succeeds() {
    let ds = small_dataset();
    let dv: Value = dataset_to_value(ds).into();
    let (tag, id) = match dv {
        Value::PluginOpaque { tag, id } => (tag, id),
        _ => panic!("expected PluginOpaque dataset"),
    };
    assert_eq!(tag, MlValueKind::Dataset as u8);

    // Simulates host filling missing kwargs with Null (train_size, random_state, stratify omitted).
    let args = vec![
        Value::PluginOpaque { tag, id },
        Value::Number(0.5), // test_size
        Value::Null,        // train_size
        Value::Null,        // shuffle → default true
        Value::Null,        // random_state
        Value::Null,        // stratify → default false
        Value::Bool(false), // return_indices explicit
    ];
    let out = native_dataset_split(&args);
    assert!(
        !matches!(out, Value::Null),
        "split with Null bool slots should return a value, not Null"
    );
}

#[test]
fn split_stratify_null_uses_default_false() {
    let ds = small_dataset();
    let dv: Value = dataset_to_value(ds).into();
    let (tag, id) = match dv {
        Value::PluginOpaque { tag, id } => (tag, id),
        _ => panic!("expected PluginOpaque dataset"),
    };

    let args = vec![
        Value::PluginOpaque { tag, id },
        Value::Number(0.33),
        Value::Null,
        Value::Bool(false),
        Value::Null,
        Value::Null, // stratify omitted — previously errored with "stratify must be bool"
        Value::Bool(false),
    ];
    let out = native_dataset_split(&args);
    assert!(!matches!(out, Value::Null));
}
