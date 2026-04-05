//! Разрешение путей к IDX MNIST: env от VM, dladdr, dev (CARGO_MANIFEST_DIR), cwd.

use std::env;
use std::path::{Path, PathBuf};

use crate::mnist_paths::relative_for_split;

/// Совпадает с именем env, которое выставляет DataCode VM при загрузке `libml` (`DATACODE_NATIVE_MODULE_<NAME>_ROOT`).
const ENV_PACKAGE_ROOT: &str = "DATACODE_NATIVE_MODULE_ML_ROOT";

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

/// Возвращает пути к файлам изображений и меток или сообщение об ошибке.
pub fn resolve_mnist_paths(split: &str) -> Result<(String, String), String> {
    let Some((rel_img, rel_lbl)) = relative_for_split(split) else {
        return Err(format!("load_mnist: unknown split {:?} (use \"train\" or \"test\")", split));
    };

    if let Ok(root) = env::var(ENV_PACKAGE_ROOT) {
        if let Some(p) = try_pair(Path::new(&root), rel_img, rel_lbl) {
            return Ok(p);
        }
    }

    if let Some(parent) = dylib_parent_dir() {
        if let Some(p) = try_pair(&parent, rel_img, rel_lbl) {
            return Ok(p);
        }
    }

    let dev_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(p) = try_pair(&dev_root, rel_img, rel_lbl) {
        return Ok(p);
    }

    if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        if let Some(p) = try_pair(Path::new(&manifest_dir), rel_img, rel_lbl) {
            return Ok(p);
        }
    }

    if let Ok(cwd) = env::current_dir() {
        if let Some(p) = try_pair(&cwd, rel_img, rel_lbl) {
            return Ok(p);
        }
        for up in 1..=4 {
            let mut p = cwd.clone();
            for _ in 0..up {
                p.pop();
            }
            if let Some(pair) = try_pair(&p, rel_img, rel_lbl) {
                return Ok(pair);
            }
        }
    }

    Err(format!(
        "load_mnist: MNIST files not found (expected {}/{} under package root, {}, or {}). \
         Install the ml package with DPM so datasets/mnist is present, or set {}.",
        rel_img,
        rel_lbl,
        dev_root.display(),
        "CARGO_MANIFEST_DIR",
        ENV_PACKAGE_ROOT
    ))
}
