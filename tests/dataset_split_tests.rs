//! Unit tests for `Dataset::split`.

use ml::dataset::Dataset;
use ml::tensor::Tensor;
use std::collections::HashSet;

fn make_dataset(n: usize, n_features: usize, labels: &[f32]) -> Dataset {
    assert_eq!(labels.len(), n);
    let fdata: Vec<f32> = (0..n * n_features).map(|i| i as f32).collect();
    let tdata: Vec<f32> = labels.to_vec();
    let f = Tensor::new(fdata, vec![n, n_features]).unwrap();
    let t = Tensor::new(tdata, vec![n, 1]).unwrap();
    Dataset::from_tensors(f, t).unwrap()
}

#[test]
fn split_basic_sizes() {
    let ds = make_dataset(10, 2, &[0.0; 10]);
    let r = ds
        .split(Some(0.2), None, false, None, false, false)
        .unwrap();
    assert_eq!(r.test.batch_size(), 2);
    assert_eq!(r.train.batch_size(), 8);
}

#[test]
fn split_shuffle_preserves_partition() {
    let ds = make_dataset(20, 1, &[0.0; 20]);
    let r = ds
        .split(Some(0.25), None, true, None, false, true)
        .unwrap();
    let t = r.test_indices.as_ref().unwrap();
    let tr = r.train_indices.as_ref().unwrap();
    assert_eq!(t.len() + tr.len(), 20);
    let mut set: HashSet<usize> = t.iter().copied().collect();
    for &i in tr {
        assert!(set.insert(i));
    }
    assert_eq!(set.len(), 20);
}

#[test]
fn split_random_state_deterministic() {
    let ds = make_dataset(30, 2, &[0.0; 30]);
    let a = ds
        .split(Some(0.2), None, true, Some(42), false, true)
        .unwrap();
    let b = ds
        .split(Some(0.2), None, true, Some(42), false, true)
        .unwrap();
    assert_eq!(
        a.test_indices.as_ref().unwrap(),
        b.test_indices.as_ref().unwrap()
    );
    assert_eq!(
        a.train_indices.as_ref().unwrap(),
        b.train_indices.as_ref().unwrap()
    );
}

#[test]
fn split_no_shuffle_order() {
    let ds = make_dataset(10, 1, &[0.0; 10]);
    let r = ds
        .split(Some(0.2), None, false, None, false, true)
        .unwrap();
    // sklearn-style: train rows first, test rows last when shuffle is false.
    assert_eq!(r.train_indices.as_ref().unwrap(), &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(r.test_indices.as_ref().unwrap(), &[8, 9]);
}

#[test]
fn split_stratify_balanced() {
    let mut labels = Vec::with_capacity(100);
    for c in 0..4 {
        for _ in 0..25 {
            labels.push(c as f32);
        }
    }
    let ds = make_dataset(100, 1, &labels);
    let r = ds
        .split(Some(0.2), None, true, Some(7), true, true)
        .unwrap();
    let tidx = r.test_indices.as_ref().unwrap();
    assert_eq!(tidx.len(), 20);
    let mut per_class = [0usize; 4];
    for &i in tidx {
        let c = labels[i] as usize;
        per_class[c] += 1;
    }
    for c in per_class {
        assert_eq!(c, 5);
    }
}

#[test]
fn split_return_indices() {
    let ds = make_dataset(10, 1, &[0.0; 10]);
    let r = ds
        .split(Some(0.2), None, false, None, false, true)
        .unwrap();
    assert_eq!(r.train_indices.as_ref().unwrap().len(), 8);
    assert_eq!(r.test_indices.as_ref().unwrap().len(), 2);
    let r2 = ds
        .split(Some(0.2), None, false, None, false, false)
        .unwrap();
    assert!(r2.train_indices.is_none());
    assert!(r2.test_indices.is_none());
}

#[test]
fn split_errors_both_sizes() {
    let ds = make_dataset(5, 1, &[0.0; 5]);
    assert!(ds
        .split(Some(0.2), Some(0.8), false, None, false, false)
        .is_err());
}

#[test]
fn split_error_invalid_test_count() {
    let ds = make_dataset(1, 1, &[0.0]);
    assert!(ds
        .split(Some(0.5), None, false, None, false, false)
        .is_err());
}

#[test]
fn split_error_test_size_out_of_range() {
    let ds = make_dataset(10, 1, &[0.0; 10]);
    assert!(ds.split(Some(0.0), None, false, None, false, false).is_err());
    assert!(ds.split(Some(10.0), None, false, None, false, false).is_err());
}

#[test]
fn split_integer_test_size_absolute_count() {
    let ds = make_dataset(100, 1, &[0.0; 100]);
    let r = ds
        .split(Some(25.0), None, false, None, false, true)
        .unwrap();
    assert_eq!(r.test.batch_size(), 25);
    assert_eq!(r.train.batch_size(), 75);
}

#[test]
fn split_integer_train_size_absolute_count() {
    let ds = make_dataset(100, 1, &[0.0; 100]);
    let r = ds
        .split(None, Some(80.0), false, None, false, true)
        .unwrap();
    assert_eq!(r.train.batch_size(), 80);
    assert_eq!(r.test.batch_size(), 20);
}
