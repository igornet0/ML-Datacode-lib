//! Управление наборами данных: файлы в `{DATACODE_DIST}/datasets/<name>`, при отсутствии — `make download-datasets-<target>`.

use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::datasets::{DatasetRelLayout, DatasetType};

/// Совпадает с именем env, которое выставляет DataCode VM при загрузке `libml` (`DATACODE_NATIVE_MODULE_<NAME>_ROOT`).
const ENV_PACKAGE_ROOT: &str = "DATACODE_NATIVE_MODULE_ML_ROOT";

/// Абсолютные пути к файлам датасета после успешного разрешения.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ResolvedDatasetPaths {
    Mnist { images: String, labels: String },
    Cifar10 { paths: Vec<String> },
    Cifar100 { path: String },
}

fn try_pair(root: &Path, rel_img: &str, rel_lbl: &str) -> Option<(String, String)> {
    let pi = root.join(rel_img);
    let pl = root.join(rel_lbl);
    if pi.is_file() && pl.is_file() {
        Some((
            pi.to_string_lossy().into_owned(),
            pl.to_string_lossy().into_owned(),
        ))
    } else {
        None
    }
}

fn all_files_exist(root: &Path, rel_paths: &[&str]) -> Option<Vec<String>> {
    let mut out = Vec::with_capacity(rel_paths.len());
    for rel in rel_paths {
        let p = root.join(rel);
        if !p.is_file() {
            return None;
        }
        out.push(p.to_string_lossy().into_owned());
    }
    Some(out)
}

fn resolve_under_root(root: &Path, layout: &DatasetRelLayout<'_>) -> Option<ResolvedDatasetPaths> {
    match layout {
        DatasetRelLayout::Mnist { images, labels } => try_pair(root, images, labels).map(|(a, b)| {
            ResolvedDatasetPaths::Mnist {
                images: a,
                labels: b,
            }
        }),
        DatasetRelLayout::Cifar10 { paths } => all_files_exist(root, paths)
            .map(|paths| ResolvedDatasetPaths::Cifar10 { paths }),
        DatasetRelLayout::Cifar100 { path } => {
            let p = root.join(path);
            if p.is_file() {
                Some(ResolvedDatasetPaths::Cifar100 {
                    path: p.to_string_lossy().into_owned(),
                })
            } else {
                None
            }
        }
    }
}

/// Пробует пути под корнем пакета, поднимаясь от `start` к предкам.
fn try_resolve_walk_up(start: &Path, layout: &DatasetRelLayout<'_>) -> Option<ResolvedDatasetPaths> {
    let mut p = start.to_path_buf();
    for _ in 0..12 {
        if let Some(res) = resolve_under_root(&p, layout) {
            return Some(res);
        }
        if !p.pop() {
            break;
        }
    }
    None
}

fn try_all_anchors(dataset: DatasetType, split: &str) -> Option<ResolvedDatasetPaths> {
    let layout = dataset.relative_layout_for_split(split)?;

    if let Ok(root) = env::var(ENV_PACKAGE_ROOT) {
        if let Some(p) = try_resolve_walk_up(Path::new(&root), &layout) {
            return Some(p);
        }
    }

    if let Some(parent) = dylib_parent_dir() {
        if let Some(p) = try_resolve_walk_up(&parent, &layout) {
            return Some(p);
        }
    }

    let dev_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(p) = try_resolve_walk_up(&dev_root, &layout) {
        return Some(p);
    }

    if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        if let Some(p) = try_resolve_walk_up(Path::new(&manifest_dir), &layout) {
            return Some(p);
        }
    }

    if let Ok(cwd) = env::current_dir() {
        if let Some(p) = try_resolve_walk_up(&cwd, &layout) {
            return Some(p);
        }
        for up in 1..=4 {
            let mut p = cwd.clone();
            for _ in 0..up {
                p.pop();
            }
            if let Some(pair) = try_resolve_walk_up(&p, &layout) {
                return Some(pair);
            }
        }
    }

    None
}

fn find_makefile_root(start: &Path) -> Option<PathBuf> {
    let mut p = start.to_path_buf();
    for _ in 0..16 {
        if p.join("Makefile").is_file() {
            return Some(p);
        }
        if !p.pop() {
            break;
        }
    }
    None
}

fn first_make_root() -> Option<PathBuf> {
    if let Ok(root) = env::var(ENV_PACKAGE_ROOT) {
        if let Some(r) = find_makefile_root(&PathBuf::from(root)) {
            return Some(r);
        }
    }
    if let Some(parent) = dylib_parent_dir() {
        if let Some(r) = find_makefile_root(&parent) {
            return Some(r);
        }
    }
    if let Some(r) = find_makefile_root(&PathBuf::from(env!("CARGO_MANIFEST_DIR"))) {
        return Some(r);
    }
    if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        if let Some(r) = find_makefile_root(&PathBuf::from(manifest_dir)) {
            return Some(r);
        }
    }
    if let Ok(cwd) = env::current_dir() {
        if let Some(r) = find_makefile_root(&cwd) {
            return Some(r);
        }
    }
    None
}

fn run_download_make(make_root: &Path, dataset: DatasetType) -> Result<(), String> {
    let target = dataset.make_download_target();
    let status = Command::new("make")
        .current_dir(make_root)
        .arg(format!("download-datasets-{target}"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| {
            format!(
                "load_{}: could not run `make download-datasets-{target}` in {}: {}",
                dataset.name(),
                make_root.display(),
                e
            )
        })?;
    if !status.success() {
        return Err(format!(
            "load_{}: `make download-datasets-{target}` failed in {} (status {})",
            dataset.name(),
            make_root.display(),
            status
        ));
    }
    Ok(())
}

fn expected_dir_hint(dataset: DatasetType) -> &'static str {
    match dataset {
        DatasetType::Mnist => concat!(env!("DATACODE_DIST"), "/datasets/mnist/"),
        DatasetType::Cifar10 => concat!(env!("DATACODE_DIST"), "/datasets/cifar-10/"),
        DatasetType::Cifar100 => concat!(env!("DATACODE_DIST"), "/datasets/cifar-100/"),
    }
}

#[cfg(unix)]
#[inline(never)]
fn dladdr_anchor() {}

#[cfg(unix)]
fn dylib_parent_dir() -> Option<PathBuf> {
    use libc::{c_void, dladdr, Dl_info};
    use std::ffi::CStr;

    let mut info = std::mem::MaybeUninit::<Dl_info>::uninit();
    let addr = dladdr_anchor as *const c_void;
    unsafe {
        if dladdr(addr, info.as_mut_ptr()) == 0 {
            return None;
        }
        let info = info.assume_init();
        if info.dli_fname.is_null() {
            return None;
        }
        let path = CStr::from_ptr(info.dli_fname).to_string_lossy().into_owned();
        PathBuf::from(path).parent().map(|p| p.to_path_buf())
    }
}

#[cfg(not(unix))]
fn dylib_parent_dir() -> Option<PathBuf> {
    None
}

/// Разрешает абсолютные пути к файлам датасета или возвращает сообщение об ошибке.
pub fn resolve_dataset_paths(dataset: DatasetType, split: &str) -> Result<ResolvedDatasetPaths, String> {
    if dataset.relative_layout_for_split(split).is_none() {
        return Err(format!(
            "load_{}: unknown split {:?} (use \"train\" or \"test\")",
            dataset.name(),
            split
        ));
    }

    if let Some(paths) = try_all_anchors(dataset, split) {
        return Ok(paths);
    }

    let Some(make_root) = first_make_root() else {
        let expected = expected_dir_hint(dataset);
        return Err(format!(
            "load_{}: dataset files not found under `{expected}` relative to the ml package root, \
             and could not locate Makefile to run download. See datacode_sdk::module_dist.",
            dataset.name()
        ));
    };

    run_download_make(&make_root, dataset)?;

    try_all_anchors(dataset, split).ok_or_else(|| {
        format!(
            "load_{}: dataset still missing after `make download-datasets-{}` in {}",
            dataset.name(),
            dataset.make_download_target(),
            make_root.display()
        )
    })
}

/// Возвращает пути к файлам изображений и меток MNIST или сообщение об ошибке.
pub fn resolve_mnist_paths(split: &str) -> Result<(String, String), String> {
    match resolve_dataset_paths(DatasetType::Mnist, split)? {
        ResolvedDatasetPaths::Mnist { images, labels } => Ok((images, labels)),
        _ => Err("load_mnist: internal error (unexpected dataset layout)".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn resolve_with_native_module_ml_root_env() {
        let ml_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("dist/ml-metal");
        std::env::set_var(
            ENV_PACKAGE_ROOT,
            ml_root.to_string_lossy().as_ref(),
        );
        let r = resolve_mnist_paths("train");
        assert!(r.is_ok(), "{:?}", r);
    }

    #[test]
    fn resolve_dataset_paths_mnist_matches_resolve_mnist_paths() {
        let ml_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("dist/ml-metal");
        std::env::set_var(
            ENV_PACKAGE_ROOT,
            ml_root.to_string_lossy().as_ref(),
        );
        let a = resolve_mnist_paths("train").expect("mnist");
        let b = resolve_dataset_paths(DatasetType::Mnist, "train").expect("dataset");
        match b {
            ResolvedDatasetPaths::Mnist { images, labels } => {
                assert_eq!(a, (images, labels));
            }
            _ => panic!("expected Mnist variant"),
        }
    }

    #[test]
    fn unknown_split_errors() {
        let e = resolve_dataset_paths(DatasetType::Mnist, "val").unwrap_err();
        assert!(e.contains("unknown split"));
    }
}
