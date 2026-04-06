//! Материализация встроенных датасетов (MNIST, CIFAR-10/100) из файлов в `dist`.

use crate::dataset::Dataset;
use crate::datasets::DatasetType;
use crate::datasets_manager::{resolve_dataset_paths, ResolvedDatasetPaths};

/// Загрузить один канонический сплит (`"train"` / `"test"`).
pub fn materialize_catalog_split(kind: DatasetType, split: &str) -> Result<Dataset, String> {
    let resolved = resolve_dataset_paths(kind, split)?;
    match (kind, resolved) {
        (DatasetType::Mnist, ResolvedDatasetPaths::Mnist { images, labels }) => {
            Dataset::from_mnist(&images, &labels)
        }
        (DatasetType::Cifar10, ResolvedDatasetPaths::Cifar10 { paths }) => {
            Dataset::from_cifar10_bin_paths(&paths)
        }
        (DatasetType::Cifar100, ResolvedDatasetPaths::Cifar100 { path }) => {
            Dataset::from_cifar_bin_file(&path)
        }
        _ => Err("builtin dataset: internal path mismatch".to_string()),
    }
}

/// Полный набор для `dataset.split(test_size=...)` (train+test объединены).
pub fn materialize_catalog_full(kind: DatasetType) -> Result<Dataset, String> {
    match kind {
        DatasetType::Mnist => {
            let mut train = materialize_catalog_split(DatasetType::Mnist, "train")?;
            let test = materialize_catalog_split(DatasetType::Mnist, "test")?;
            train.concat_in_place(&test)?;
            Ok(train)
        }
        DatasetType::Cifar10 => {
            let mut train = materialize_catalog_split(DatasetType::Cifar10, "train")?;
            let test = materialize_catalog_split(DatasetType::Cifar10, "test")?;
            train.concat_in_place(&test)?;
            Ok(train)
        }
        DatasetType::Cifar100 => {
            let mut train = materialize_catalog_split(DatasetType::Cifar100, "train")?;
            let test = materialize_catalog_split(DatasetType::Cifar100, "test")?;
            train.concat_in_place(&test)?;
            Ok(train)
        }
    }
}

/// Убедиться, что файлы датасета есть (скачивание через `resolve_dataset_paths`).
pub fn ensure_builtin_dataset_ready(kind: DatasetType) -> Result<(), String> {
    resolve_dataset_paths(kind, "train").map(|_| ())
}

#[cfg(test)]
mod tests {
    use crate::dataset::Dataset;

    #[test]
    fn cifar_record_roundtrip() {
        const R: usize = 3073;
        let mut buf = vec![0u8; R * 2];
        buf[0] = 3;
        buf[1] = 255;
        buf[R + 0] = 7;
        buf[R + 100] = 128;
        let path = std::env::temp_dir().join("cifar_test.bin");
        std::fs::write(&path, &buf).unwrap();
        let ds = Dataset::from_cifar_bin_file(path.to_str().unwrap()).unwrap();
        assert_eq!(ds.batch_size(), 2);
        assert_eq!(ds.num_features(), 3072);
        let _ = std::fs::remove_file(&path);
    }
}
