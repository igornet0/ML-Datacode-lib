//! Shared setup for integration tests that run DataCode VM code with `import ml`.
//! Copies `libml` cdylib and repo-root [`setup.dcmodule`](../../setup.dcmodule) into DPM-style
//! `tests/fixtures/env/packages/ml/`, then sets `file_import::set_dpm_package_paths` before each `run`.
//!
//! Prefer the dylib produced by the **same** `cargo test` invocation (`target/{debug|release}/deps/libml.*`)
//! so the VM can resolve the native package. If `libml` is missing, run `cargo build` (or `cargo test --no-run`)
//! and ensure `crate-type` includes `cdylib` in the root [`Cargo.toml`](../../Cargo.toml).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

use data_code::vm::file_import;
use data_code::{run_with_base_path, LangError, Value};

/// Opaque tag for ML tensors in the VM (`Value::PluginOpaque`), matching `ml::MlValueKind::Tensor`.
#[allow(dead_code)]
pub const ML_PLUGIN_TENSOR_TAG: u8 = 0;

/// True if `v` is an ML tensor handle as returned through the DataCode VM (`PluginOpaque` with tensor tag).
#[allow(dead_code)]
pub fn is_ml_tensor_value(v: &Value) -> bool {
    matches!(
        v,
        Value::PluginOpaque { tag, .. } if *tag == ML_PLUGIN_TENSOR_TAG
    )
}

static SETUP_LOCK: Mutex<()> = Mutex::new(());
static ML_NATIVE_READY: OnceLock<()> = OnceLock::new();

pub fn packages_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("env")
        .join("packages")
}

fn dylib_filename() -> &'static str {
    if cfg!(target_os = "macos") {
        "libml.dylib"
    } else if cfg!(target_os = "windows") {
        "ml.dll"
    } else {
        "libml.so"
    }
}

/// `cargo` places the cdylib next to `rlib` under `target/<profile>/deps/`.
fn target_root() -> PathBuf {
    std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"))
}

fn deps_dylib_path(profile: &str) -> PathBuf {
    target_root()
        .join(profile)
        .join("deps")
        .join(dylib_filename())
}

fn nested_release_dylib() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("ml_cdylib_target")
        .join("release")
        .join(dylib_filename())
}

fn manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml")
}

fn target_dir_for_nested_build() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("ml_cdylib_target")
}

fn build_cdylib_release_nested() -> Result<(), String> {
    let status = Command::new("cargo")
        .arg("build")
        .arg("--release")
        .arg("--manifest-path")
        .arg(manifest_path())
        .arg("--target-dir")
        .arg(target_dir_for_nested_build())
        .status()
        .map_err(|e| format!("cargo: {}", e))?;
    if !status.success() {
        return Err("cargo build --release for ml (libml cdylib) failed".to_string());
    }
    Ok(())
}

fn resolve_dylib_src() -> Result<PathBuf, String> {
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let primary = deps_dylib_path(profile);
    if primary.is_file() {
        return Ok(primary);
    }
    let release_deps = deps_dylib_path("release");
    if release_deps.is_file() {
        return Ok(release_deps);
    }
    build_cdylib_release_nested()?;
    let nested = nested_release_dylib();
    if nested.is_file() {
        return Ok(nested);
    }
    Err(format!(
        "libml not found (tried {}, {}, nested release); build the `ml` crate with cdylib first",
        primary.display(),
        release_deps.display()
    ))
}

fn copy_dylib_to_packages(src: &Path) -> Result<(), String> {
    let dest_dir = packages_root().join("ml");
    fs::create_dir_all(&dest_dir).map_err(|e| e.to_string())?;
    let name = src.file_name().ok_or("dylib has no file name")?;
    let dest = dest_dir.join(name);
    fs::copy(src, &dest).map_err(|e| e.to_string())?;
    Ok(())
}

/// Copies [`setup.dcmodule`](../../setup.dcmodule) next to the dylib so `import ml` resolves like a DPM package.
fn copy_setup_dcmodule_to_packages() -> Result<(), String> {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("setup.dcmodule");
    if !src.is_file() {
        return Err(format!(
            "setup.dcmodule not found at {} (repo root)",
            src.display()
        ));
    }
    let dest_dir = packages_root().join("ml");
    fs::create_dir_all(&dest_dir).map_err(|e| e.to_string())?;
    let dest = dest_dir.join("setup.dcmodule");
    fs::copy(&src, &dest).map_err(|e| e.to_string())?;
    Ok(())
}

pub fn ensure_ml_native() {
    ML_NATIVE_READY.get_or_init(|| {
        let _guard = SETUP_LOCK.lock().expect("setup lock");
        let src = resolve_dylib_src().unwrap_or_else(|e| {
            panic!(
                "resolve libml dylib failed: {}\n\
                 Hint: run `cargo build` from the repo root so `target/{}/deps/{}` exists.",
                e,
                if cfg!(debug_assertions) {
                    "debug"
                } else {
                    "release"
                },
                dylib_filename()
            );
        });
        copy_dylib_to_packages(&src).expect("copy libml to packages/ml");
        copy_setup_dcmodule_to_packages().expect("copy setup.dcmodule to packages/ml");
    });
}

pub fn ml_package_dir() -> PathBuf {
    packages_root().join("ml")
}

/// Runs DataCode source with `import ml` resolving to the copied `libml` cdylib.
/// Native loader looks for `libml.*` in the VM **base path** (see `data-code` `native_loader`),
/// not only under DPM roots — so base path must be the `packages/ml` directory.
pub fn run_ml(source: &str) -> Result<Value, LangError> {
    ensure_ml_native();
    file_import::set_dpm_package_paths(vec![packages_root()]);
    run_with_base_path(source, &ml_package_dir())
}
