//! `native_dataset_concat` / `native_dataset_push_data` and `Dataset` helpers.

use ml::dataset::Dataset;
use ml::ml_types::MlValueKind;
use ml::natives::{native_dataset_concat, native_dataset_push_data};
use ml::runtime::dataset_to_value;
use ml::tensor::Tensor;
use ml::PluginValue as Value;

fn ds_2x2() -> Dataset {
    let f = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let t = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
    Dataset::from_tensors(f, t).unwrap()
}

#[test]
fn concat_two_datasets_doubles_batch() {
    let a = ds_2x2();
    let b = ds_2x2();
    let va: Value = dataset_to_value(a).into();
    let vb: Value = dataset_to_value(b).into();
    let (tag_a, id_a) = match va {
        Value::PluginOpaque { tag, id } => (tag, id),
        _ => panic!("expected dataset opaque"),
    };
    let (tag_b, id_b) = match vb {
        Value::PluginOpaque { tag, id } => (tag, id),
        _ => panic!("expected dataset opaque"),
    };
    assert_eq!(tag_a, MlValueKind::Dataset as u8);
    assert_eq!(tag_b, MlValueKind::Dataset as u8);

    let args = vec![
        Value::PluginOpaque { tag: tag_a, id: id_a },
        Value::PluginOpaque { tag: tag_b, id: id_b },
    ];
    let out = native_dataset_concat(&args);
    assert!(matches!(out, Value::Null));
    let ds = ml::runtime::get_dataset(id_a).expect("left dataset");
    assert_eq!(ds.borrow().batch_size(), 4);
}

#[test]
fn push_data_appends_rows() {
    let base = ds_2x2();
    let va: Value = dataset_to_value(base).into();
    let (tag, id) = match va {
        Value::PluginOpaque { tag, id } => (tag, id),
        _ => panic!("expected dataset opaque"),
    };
    let f = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    let t = Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap();
    let args = vec![
        Value::PluginOpaque { tag, id },
        ml::runtime::tensor_to_value(f),
        ml::runtime::tensor_to_value(t),
    ];
    let out = native_dataset_push_data(&args);
    assert!(matches!(out, Value::Null));
    let ds = ml::runtime::get_dataset(id).expect("dataset");
    assert_eq!(ds.borrow().batch_size(), 4);
}

#[test]
fn concat_mismatched_names_fails() {
    let a = ds_2x2();
    // Mismatch: 3 feature columns vs 2 (names x0,x1,x2 vs x0,x1).
    let f3 = Tensor::new(vec![1.0; 6], vec![2, 3]).unwrap();
    let t3 = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
    let c = Dataset::from_tensors(f3, t3).unwrap();
    let va: Value = dataset_to_value(a).into();
    let vc: Value = dataset_to_value(c).into();
    let (tag_a, id_a) = match va {
        Value::PluginOpaque { tag, id } => (tag, id),
        _ => panic!("expected dataset opaque"),
    };
    let (tag_c, id_c) = match vc {
        Value::PluginOpaque { tag, id } => (tag, id),
        _ => panic!("expected dataset opaque"),
    };
    let args = vec![
        Value::PluginOpaque { tag: tag_a, id: id_a },
        Value::PluginOpaque { tag: tag_c, id: id_c },
    ];
    let out = native_dataset_concat(&args);
    assert!(matches!(out, Value::Null));
    // Error path: batch unchanged.
    let ds = ml::runtime::get_dataset(id_a).expect("left");
    assert_eq!(ds.borrow().batch_size(), 2);
}
