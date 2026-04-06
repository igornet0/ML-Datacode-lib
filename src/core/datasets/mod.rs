use crate::cifar_path::{cifar100_relative_for_split, cifar10_relative_for_split};
use crate::mnist_path::mnist_relative_for_split;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DatasetType {
    Mnist,
    Cifar10,
    Cifar100,
}

/// Относительные пути от корня пакета / `DATACODE_DIST` для проверки наличия файлов.
pub enum DatasetRelLayout<'a> {
    Mnist {
        images: &'static str,
        labels: &'static str,
    },
    Cifar10 {
        paths: &'a [&'static str],
    },
    Cifar100 {
        path: &'static str,
    },
}

/// Имена для `ml.load_dataset("...")`: `"mnist"`, `"cifar-10"`, `"cifar-100"`.
pub fn parse_builtin_dataset_name(name: &str) -> Option<DatasetType> {
    let s = name.trim();
    if s.eq_ignore_ascii_case("mnist") {
        Some(DatasetType::Mnist)
    } else if s.eq_ignore_ascii_case("cifar-10") || s.eq_ignore_ascii_case("cifar10") {
        Some(DatasetType::Cifar10)
    } else if s.eq_ignore_ascii_case("cifar-100") || s.eq_ignore_ascii_case("cifar100") {
        Some(DatasetType::Cifar100)
    } else {
        None
    }
}

impl DatasetType {
    pub fn name(&self) -> &'static str {
        match self {
            DatasetType::Mnist => "mnist",
            DatasetType::Cifar10 => "cifar10",
            DatasetType::Cifar100 => "cifar100",
        }
    }

    /// Имя цели Makefile: `download-datasets-<target>`.
    pub fn make_download_target(&self) -> &'static str {
        match self {
            DatasetType::Mnist => "mnist",
            DatasetType::Cifar10 => "cifar-10",
            DatasetType::Cifar100 => "cifar-100",
        }
    }

    pub fn relative_layout_for_split(&self, split: &str) -> Option<DatasetRelLayout<'_>> {
        match self {
            DatasetType::Mnist => mnist_relative_for_split(split).map(|(images, labels)| {
                DatasetRelLayout::Mnist { images, labels }
            }),
            DatasetType::Cifar10 => {
                cifar10_relative_for_split(split).map(|paths| DatasetRelLayout::Cifar10 { paths })
            }
            DatasetType::Cifar100 => cifar100_relative_for_split(split).map(|path| {
                DatasetRelLayout::Cifar100 { path }
            }),
        }
    }
}
