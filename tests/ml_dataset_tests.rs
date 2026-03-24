// Tests for ML Dataset (import ml — native dylib; Table values cannot cross the ABI).

mod test_support;

use data_code::Value;
use test_support::run_ml;

#[test]
fn test_dataset_from_table() {
    // Tensor path mirrors table layout: 3 rows, 2 features + 1 target
    let code = r#"
        import ml
        let features = ml.tensor([1.0, 2.0, 4.0, 5.0, 7.0, 8.0], [3, 2])
        let targets = ml.tensor([3.0, 6.0, 9.0], [3, 1])
        let ds = ml.dataset_from_tensors(features, targets)
        typeof(ds)
    "#;
    let result = run_ml(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::String(s) => {
            assert_eq!(s, "plugin_opaque", "Dataset handle should be plugin_opaque");
        }
        _ => panic!("Expected String for type"),
    }
}

#[test]
fn test_dataset_features() {
    let code = r#"
        import ml
        let features = ml.tensor([1.0, 2.0, 4.0, 5.0], [2, 2])
        let targets = ml.tensor([3.0, 6.0], [2, 1])
        let ds = ml.dataset_from_tensors(features, targets)
        let f = ml.dataset_features(ds)
        ml.shape(f)
    "#;
    let result = run_ml(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            assert_eq!(arr_ref.len(), 2, "Features shape should have 2 dimensions");
            match (&arr_ref[0], &arr_ref[1]) {
                (Value::Number(n1), Value::Number(n2)) => {
                    assert_eq!(*n1 as usize, 2, "Batch size should be 2");
                    assert_eq!(*n2 as usize, 2, "Feature count should be 2");
                }
                _ => panic!("Expected Number values in shape"),
            }
        }
        _ => panic!("Expected Array for shape"),
    }
}

#[test]
fn test_dataset_targets() {
    let code = r#"
        import ml
        let features = ml.tensor([1.0, 2.0, 4.0, 5.0], [2, 2])
        let targets = ml.tensor([3.0, 6.0], [2, 1])
        let ds = ml.dataset_from_tensors(features, targets)
        let t = ml.dataset_targets(ds)
        ml.shape(t)
    "#;
    let result = run_ml(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            assert_eq!(arr_ref.len(), 2, "Targets shape should have 2 dimensions");
            match (&arr_ref[0], &arr_ref[1]) {
                (Value::Number(n1), Value::Number(n2)) => {
                    assert_eq!(*n1 as usize, 2, "Batch size should be 2");
                    assert_eq!(*n2 as usize, 1, "Target count should be 1");
                }
                _ => panic!("Expected Number values in shape"),
            }
        }
        _ => panic!("Expected Array for shape"),
    }
}

#[test]
fn test_dataset_with_linear_regression() {
    let code = r#"
        import ml
        let features = ml.tensor([1.0, 2.0, 2.0, 3.0, 3.0, 4.0], [3, 2])
        let targets = ml.tensor([3.0, 5.0, 7.0], [3, 1])
        let ds = ml.dataset_from_tensors(features, targets)
        let xf = ml.dataset_features(ds)
        let yt = ml.dataset_targets(ds)
        let model = ml.linear_regression(2)
        let loss_history = ml.lr_train(model, xf, yt, 10, 0.01)
        len(loss_history)
    "#;
    let result = run_ml(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert_eq!(n as usize, 10, "Loss history should have 10 entries");
        }
        _ => panic!("Expected Number for loss history length"),
    }
}

#[test]
fn test_dataset_multiple_targets() {
    let code = r#"
        import ml
        let features = ml.tensor([1.0, 2.0, 5.0, 6.0], [2, 2])
        let targets = ml.tensor([3.0, 4.0, 7.0, 8.0], [2, 2])
        let ds = ml.dataset_from_tensors(features, targets)
        let t = ml.dataset_targets(ds)
        let shape = ml.shape(t)
        shape[1]
    "#;
    let result = run_ml(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert_eq!(n as usize, 2, "Should have 2 target columns");
        }
        _ => panic!("Expected Number for target count"),
    }
}

#[test]
fn test_load_mnist_train() {
    let code = r#"
        import ml
        let dataset_train = ml.load_mnist("train")
        typeof(dataset_train)
    "#;
    let result = run_ml(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::String(s) => {
            assert_eq!(s, "plugin_opaque", "Expected plugin_opaque dataset handle");
        }
        _ => panic!("Expected String for type"),
    }
}

#[test]
fn test_load_mnist_test() {
    let code = r#"
        import ml
        let dataset_test = ml.load_mnist("test")
        typeof(dataset_test)
    "#;
    let result = run_ml(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::String(s) => {
            assert_eq!(s, "plugin_opaque", "Expected plugin_opaque dataset handle");
        }
        _ => panic!("Expected String for type"),
    }
}

#[test]
fn test_mnist_dataset_size() {
    let code = r#"
        import ml
        let dataset_train = ml.load_mnist("train")
        ml.dataset_len(dataset_train)
    "#;
    let result = run_ml(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert_eq!(n as usize, 60000, "MNIST train should have 60000 samples");
        }
        _ => panic!("Expected Number for dataset size"),
    }
}

#[test]
fn test_mnist_dataset_features_shape() {
    let code = r#"
        import ml
        let dataset_train = ml.load_mnist("train")
        let features = ml.dataset_features(dataset_train)
        ml.shape(features)
    "#;
    let result = run_ml(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            assert_eq!(arr_ref.len(), 2, "Features shape should have 2 dimensions");
            match (&arr_ref[0], &arr_ref[1]) {
                (Value::Number(n1), Value::Number(n2)) => {
                    assert_eq!(*n1 as usize, 60000, "Batch size should be 60000");
                    assert_eq!(*n2 as usize, 784, "Feature count should be 784 (28x28)");
                }
                _ => panic!("Expected Number values in shape"),
            }
        }
        _ => panic!("Expected Array for shape"),
    }
}

#[test]
fn test_mnist_dataset_targets_shape() {
    let code = r#"
        import ml
        let dataset_train = ml.load_mnist("train")
        let targets = ml.dataset_targets(dataset_train)
        ml.shape(targets)
    "#;
    let result = run_ml(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            assert_eq!(arr_ref.len(), 2, "Targets shape should have 2 dimensions");
            match (&arr_ref[0], &arr_ref[1]) {
                (Value::Number(n1), Value::Number(n2)) => {
                    assert_eq!(*n1 as usize, 60000, "Batch size should be 60000");
                    assert_eq!(*n2 as usize, 1, "Target count should be 1");
                }
                _ => panic!("Expected Number values in shape"),
            }
        }
        _ => panic!("Expected Array for shape"),
    }
}

#[test]
fn test_mnist_dataset_element_access() {
    let code = r#"
        import ml
        let dataset_train = ml.load_mnist("train")
        let sample = dataset_train[0]
        len(sample)
    "#;
    let result = run_ml(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert_eq!(n as usize, 2, "Sample should be [features, target] tensors");
        }
        _ => panic!("Expected Number for sample length"),
    }
}

#[test]
fn test_mnist_dataset_iteration() {
    let code = r#"
        import ml
        let dataset_train = ml.load_mnist("train")
        let count = 0.0
        let i = 0.0
        while i < 10.0 {
            let _ = ml.dataset_get(dataset_train, i)
            count = count + 1.0
            i = i + 1.0
        }
        count
    "#;
    let result = run_ml(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert_eq!(n as usize, 10, "Should iterate 10 times");
        }
        _ => panic!("Expected Number for count"),
    }
}

#[test]
fn test_mnist_dataset_label_values() {
    let code = r#"
        import ml
        let dataset_train = ml.load_mnist("train")
        let labels = []
        let count = 0.0
        let i = 0.0
        while count < 100.0 {
            let sample = ml.dataset_get(dataset_train, i)
            let y = sample[1]
            let label = ml.data(y)[0]
            labels = labels + [label]
            count = count + 1.0
            i = i + 1.0
        }
        let min_label = 10.0
        let max_label = -1.0
        for label in labels {
            if label < min_label {
                min_label = label
            }
            if label > max_label {
                max_label = label
            }
        }
        [min_label, max_label]
    "#;
    let result = run_ml(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            assert_eq!(arr_ref.len(), 2, "Should return [min, max]");
            match (&arr_ref[0], &arr_ref[1]) {
                (Value::Number(min), Value::Number(max)) => {
                    assert!(*min >= 0.0 && *min <= 9.0, "Min label should be 0-9");
                    assert!(*max >= 0.0 && *max <= 9.0, "Max label should be 0-9");
                }
                _ => panic!("Expected Number values for min/max"),
            }
        }
        _ => panic!("Expected Array for min/max labels"),
    }
}
