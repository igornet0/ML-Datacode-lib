//! Тесты `ml.load_dataset`, `DatasetCatalog` и перегрузки `.split("train"|"test")` / sklearn.

use ml::datasets::DatasetType;
use ml::ml_types::MlValueKind;
use ml::natives::{native_load_dataset, native_plugin_call};
use ml::runtime::get_dataset;
use ml::PluginValue as Value;

const ENV_NATIVE_ML_ROOT: &str = "DATACODE_NATIVE_MODULE_ML_ROOT";

#[test]
fn parse_builtin_dataset_name_accepts_aliases() {
    use ml::parse_builtin_dataset_name;

    assert_eq!(
        parse_builtin_dataset_name("mnist"),
        Some(DatasetType::Mnist)
    );
    assert_eq!(
        parse_builtin_dataset_name("MNIST"),
        Some(DatasetType::Mnist)
    );
    assert_eq!(
        parse_builtin_dataset_name("cifar-10"),
        Some(DatasetType::Cifar10)
    );
    assert_eq!(
        parse_builtin_dataset_name("cifar10"),
        Some(DatasetType::Cifar10)
    );
    assert_eq!(
        parse_builtin_dataset_name("cifar-100"),
        Some(DatasetType::Cifar100)
    );
    assert_eq!(
        parse_builtin_dataset_name("cifar100"),
        Some(DatasetType::Cifar100)
    );
    assert_eq!(parse_builtin_dataset_name("imagenet"), None);
}

#[test]
fn ml_value_kind_dataset_catalog_discriminant() {
    assert_eq!(MlValueKind::DatasetCatalog as u8, 15);
    assert_eq!(
        MlValueKind::try_from(15),
        Ok(MlValueKind::DatasetCatalog)
    );
}

#[test]
fn load_dataset_wrong_arity_returns_null() {
    let out = native_load_dataset(&[]);
    assert!(matches!(out, Value::Null));

    let out = native_load_dataset(&[
        Value::String("mnist".to_string()),
        Value::String("extra".to_string()),
    ]);
    assert!(matches!(out, Value::Null));
}

#[test]
fn load_dataset_non_string_returns_null() {
    let out = native_load_dataset(&[Value::Number(1.0)]);
    assert!(matches!(out, Value::Null));
}

#[test]
fn load_dataset_unknown_name_returns_null() {
    let out = native_load_dataset(&[Value::String("unknown-dataset".to_string())]);
    assert!(matches!(out, Value::Null));
}

#[test]
fn catalog_len_before_split_returns_null() {
    let catalog = native_load_dataset(&[Value::String("mnist".to_string())]);
    let Value::PluginOpaque { tag, .. } = catalog else {
        // без данных в dist загрузка может вернуть Null — пропускаем ветку
        return;
    };
    if tag != MlValueKind::DatasetCatalog as u8 {
        return;
    }
    let out = native_plugin_call(&[
        catalog,
        Value::String("len".to_string()),
    ]);
    assert!(matches!(out, Value::Null));
}

/// Сценарий как в `datasets_manager::tests::resolve_with_native_module_ml_root_env`: `dist/ml-metal` и MNIST.
#[test]
fn load_dataset_split_train_materializes_mnist() {
    let ml_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("dist/ml-metal");
    std::env::set_var(
        ENV_NATIVE_ML_ROOT,
        ml_root.to_string_lossy().as_ref(),
    );

    let catalog = native_load_dataset(&[Value::String("mnist".to_string())]);
    let Value::PluginOpaque {
        tag: cat_tag,
        id: _cat_id,
    } = catalog
    else {
        panic!("load_dataset(mnist) expected catalog or Null, got {:?}", catalog);
    };
    assert_eq!(
        cat_tag,
        MlValueKind::DatasetCatalog as u8,
        "expected DatasetCatalog tag"
    );

    let bound = native_plugin_call(&[
        catalog.clone(),
        Value::String("split".to_string()),
    ]);
    let Value::PluginOpaque {
        tag: b_tag,
        ..
    } = bound
    else {
        panic!("expected bound method for split, got {:?}", bound);
    };
    assert_eq!(b_tag, MlValueKind::BoundMethod as u8);

    let materialized = native_plugin_call(&[
        bound,
        catalog,
        Value::String("train".to_string()),
    ]);

    let Value::PluginOpaque {
        tag: ds_tag,
        id: ds_id,
    } = materialized
    else {
        panic!(
            "split(\"train\") expected materialized dataset, got {:?}",
            materialized
        );
    };
    assert_eq!(ds_tag, MlValueKind::Dataset as u8);

    let ds = get_dataset(ds_id).expect("dataset handle");
    let n = ds.borrow().batch_size();
    assert_eq!(
        n, 60_000,
        "MNIST train split should have 60000 samples (got {})",
        n
    );
}

#[test]
fn catalog_sklearn_split_returns_pair_of_datasets() {
    let ml_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("dist/ml-metal");
    std::env::set_var(
        ENV_NATIVE_ML_ROOT,
        ml_root.to_string_lossy().as_ref(),
    );

    let catalog = native_load_dataset(&[Value::String("mnist".to_string())]);
    let Value::PluginOpaque {
        tag: cat_tag,
        ..
    } = catalog
    else {
        return;
    };
    if cat_tag != MlValueKind::DatasetCatalog as u8 {
        return;
    }

    let bound = native_plugin_call(&[
        catalog.clone(),
        Value::String("split".to_string()),
    ]);
    let Value::PluginOpaque { tag: b_tag, .. } = bound else {
        return;
    };
    if b_tag != MlValueKind::BoundMethod as u8 {
        return;
    }

    // bound + receiver + 6 опций sklearn (как `native_dataset_split`)
    let out = native_plugin_call(&[
        bound,
        catalog,
        Value::Number(0.1), // test_size
        Value::Null,        // train_size
        Value::Bool(false), // shuffle
        Value::Null,        // random_state
        Value::Bool(false), // stratify
        Value::Bool(false), // return_indices
    ]);

    let Value::Array(arr) = out else {
        panic!("expected [train_ds, test_ds] array, got {:?}", out);
    };
    let rows = arr.borrow();
    assert_eq!(rows.len(), 2, "expected train and test datasets only");
    let Value::PluginOpaque { tag: t0, .. } = &rows[0] else {
        panic!("train slot should be Dataset");
    };
    let Value::PluginOpaque { tag: t1, .. } = &rows[1] else {
        panic!("test slot should be Dataset");
    };
    assert_eq!(*t0, MlValueKind::Dataset as u8);
    assert_eq!(*t1, MlValueKind::Dataset as u8);
}
