//! Управление наборами данных: файлы в `{DATACODE_DIST}/datasets/<name>` (или `datasets_rep/datasets/<name>`),
//! при отсутствии — загрузка из репозитория датасетов в пользовательский кэш.

use std::env;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::datasets::{DatasetRelLayout, DatasetType};

/// Совпадает с именем env, которое выставляет DataCode VM при загрузке `libml` (`DATACODE_NATIVE_MODULE_<NAME>_ROOT`).
const ENV_PACKAGE_ROOT: &str = "DATACODE_NATIVE_MODULE_ML_ROOT";

/// Переопределяет каталог кэша датасетов (внутри него — `datasets/<name>/...`).
const ENV_DATASETS_DIR: &str = "DATACODE_ML_DATASETS_DIR";

/// Репозиторий датасетов (тот же, что сабмодуль `datasets_rep`). `*.bin` хранятся в Git LFS.
const DATASETS_RAW_BASE: &str = "https://raw.githubusercontent.com/igornet0/ML-Datacode-datasets/main/";
const DATASETS_LFS_BASE: &str = "https://media.githubusercontent.com/media/igornet0/ML-Datacode-datasets/main/";

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

/// `dist/datasets/mnist/x` -> `datasets/mnist/x` (раскладка репозитория датасетов и кэша).
fn strip_dist(rel: &'static str) -> &'static str {
    rel.find("datasets/").map_or(rel, |i| &rel[i..])
}

fn stripped_layout(layout: &DatasetRelLayout<'_>) -> (Vec<&'static str>, DatasetKindTag) {
    match layout {
        DatasetRelLayout::Mnist { images, labels } => {
            (vec![strip_dist(images), strip_dist(labels)], DatasetKindTag::Mnist)
        }
        DatasetRelLayout::Cifar10 { paths } => {
            (paths.iter().map(|p| strip_dist(p)).collect(), DatasetKindTag::Cifar10)
        }
        DatasetRelLayout::Cifar100 { path } => (vec![strip_dist(path)], DatasetKindTag::Cifar100),
    }
}

#[derive(Clone, Copy)]
enum DatasetKindTag {
    Mnist,
    Cifar10,
    Cifar100,
}

/// Проверяет `datasets/...` относительно `root` (кэш, `datasets_rep`).
fn resolve_stripped(root: &Path, layout: &DatasetRelLayout<'_>) -> Option<ResolvedDatasetPaths> {
    let (rels, kind) = stripped_layout(layout);
    let mut files = all_files_exist(root, &rels)?;
    Some(match kind {
        DatasetKindTag::Mnist => {
            let labels = files.pop()?;
            let images = files.pop()?;
            ResolvedDatasetPaths::Mnist { images, labels }
        }
        DatasetKindTag::Cifar10 => ResolvedDatasetPaths::Cifar10 { paths: files },
        DatasetKindTag::Cifar100 => ResolvedDatasetPaths::Cifar100 { path: files.pop()? },
    })
}

/// Пробует пути под корнем пакета, поднимаясь от `start` к предкам.
fn try_resolve_walk_up(start: &Path, layout: &DatasetRelLayout<'_>) -> Option<ResolvedDatasetPaths> {
    let mut p = start.to_path_buf();
    for _ in 0..12 {
        if let Some(res) = resolve_under_root(&p, layout) {
            return Some(res);
        }
        if let Some(res) = resolve_stripped(&p.join("datasets_rep"), layout) {
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

    if let Some(cache) = datasets_cache_dir() {
        if let Some(p) = resolve_stripped(&cache, &layout) {
            return Some(p);
        }
    }

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

/// Каталог кэша: `$DATACODE_ML_DATASETS_DIR` или `<user cache>/datacode/ml`.
fn datasets_cache_dir() -> Option<PathBuf> {
    if let Some(d) = env::var_os(ENV_DATASETS_DIR).filter(|d| !d.is_empty()) {
        return Some(PathBuf::from(d));
    }
    let base = if cfg!(target_os = "macos") {
        env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Caches"))
    } else if cfg!(windows) {
        env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else {
        env::var_os("XDG_CACHE_HOME")
            .filter(|d| !d.is_empty())
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
    }?;
    Some(base.join("datacode").join("ml"))
}

fn http_get(url: &str) -> Result<Vec<u8>, String> {
    let resp = ureq::get(url).call().map_err(|e| format!("GET {url}: {e}"))?;
    let mut body = Vec::new();
    resp.into_reader()
        .read_to_end(&mut body)
        .map_err(|e| format!("GET {url}: {e}"))?;
    Ok(body)
}

/// Скачивает один файл `datasets/...`; LFS-указатель с raw заменяется содержимым из media.
fn download_file(rel: &str, dest: &Path) -> Result<(), String> {
    let mut body = http_get(&format!("{DATASETS_RAW_BASE}{rel}"))?;
    if body.starts_with(b"version https://git-lfs") {
        body = http_get(&format!("{DATASETS_LFS_BASE}{rel}"))?;
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let tmp = dest.with_extension("part");
    let mut f = fs::File::create(&tmp).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    f.write_all(&body)
        .and_then(|_| f.sync_all())
        .map_err(|e| format!("write {}: {e}", tmp.display()))?;
    fs::rename(&tmp, dest).map_err(|e| format!("rename {}: {e}", dest.display()))
}

/// Скачивает недостающие файлы сплита в кэш.
fn download_to_cache(dataset: DatasetType, split: &str, cache: &Path) -> Result<(), String> {
    let layout = dataset
        .relative_layout_for_split(split)
        .ok_or_else(|| format!("load_{}: unknown split {split:?}", dataset.name()))?;
    let (rels, _) = stripped_layout(&layout);
    for rel in rels {
        let dest = cache.join(rel);
        if dest.is_file() {
            continue;
        }
        eprintln!("[ml] downloading {rel} -> {}", dest.display());
        download_file(rel, &dest).map_err(|e| format!("load_{}: download failed: {e}", dataset.name()))?;
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

    let Some(cache) = datasets_cache_dir() else {
        let expected = expected_dir_hint(dataset);
        return Err(format!(
            "load_{}: dataset files not found under `{expected}` relative to the ml package root, \
             and no cache dir (set {ENV_DATASETS_DIR})",
            dataset.name()
        ));
    };

    download_to_cache(dataset, split, &cache)?;

    resolve_stripped(&cache, &dataset.relative_layout_for_split(split).expect("split checked")).ok_or_else(|| {
        format!(
            "load_{}: dataset still missing after download into {}",
            dataset.name(),
            cache.display()
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
    #[ignore = "requires MNIST files in dist/ml-metal/datasets_rep (run `make build-metal` + datasets)"]
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
    #[ignore = "requires MNIST files in dist/ml-metal/datasets_rep (run `make build-metal` + datasets)"]
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
    #[ignore = "network: downloads MNIST test split from ML-Datacode-datasets"]
    fn downloads_mnist_test_into_cache() {
        let dir = std::env::temp_dir().join(format!("dc-ml-ds-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        download_to_cache(DatasetType::Mnist, "test", &dir).expect("download");
        let r = resolve_stripped(&dir, &DatasetType::Mnist.relative_layout_for_split("test").unwrap());
        match r {
            Some(ResolvedDatasetPaths::Mnist { images, .. }) => {
                assert_eq!(std::fs::metadata(images).unwrap().len(), 7_840_016);
            }
            _ => panic!("expected Mnist variant"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unknown_split_errors() {
        let e = resolve_dataset_paths(DatasetType::Mnist, "val").unwrap_err();
        assert!(e.contains("unknown split"));
    }
}
