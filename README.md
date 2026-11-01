# ML-Datacode-lib

Standalone ML sources for DataCode (mirrors `src/lib/ml` in the main DataCode repo). Published as a **git dependency** for `dpm add ml` via [Datacode-registry-index](https://github.com/igornet0/Datacode-registry-index) (`config.json`).

**Source layout:** see [ARCHITECTURE.md](ARCHITECTURE.md) (`src/vm`, `src/core`, `src/engine`, `src/nn`, `src/gpu`).

## DPM install

```bash
dpm add ml
```

`dpm` downloads a prebuilt `.dcmodule` bundle for your platform from the [GitHub Releases](https://github.com/igornet0/ML-Datacode-lib/releases) (checked against the `sha256` in the registry) and installs it into `<env>/packages/ml/`. Prebuilt targets:

| Platform | Target | Backend |
|---|---|---|
| macOS arm64 | `aarch64-apple-darwin` | CPU + Metal |
| macOS x86_64 | `x86_64-apple-darwin` | CPU + Metal |
| Linux x86_64 | `x86_64-unknown-linux-gnu` | CPU |
| Linux arm64 | `aarch64-unknown-linux-gnu` | CPU |
| Windows x86_64 | `x86_64-pc-windows-msvc` | CPU |

If there is no bundle for your platform, `dpm` falls back to cloning the repo and building it from source (needs a Rust toolchain).

### CUDA

The prebuilt Linux/Windows bundles are CPU-only. If you need CUDA, install from source:

```bash
DPM_FROM_SOURCE=1 dpm add ml
```

This clones the repo and builds it with `cargo` on your machine. `setup.dcmodule` tries the CUDA build first and falls back to CPU if the CUDA toolkit is not found.

### Datasets

`load_mnist` / `load_cifar10` / `load_cifar100` look for files in the package (`dist/datasets/...` or the `datasets_rep` submodule). If they are missing (e.g. a prebuilt install), the files are downloaded on first use from [ML-Datacode-datasets](https://github.com/igornet0/ML-Datacode-datasets) into the user cache:

- macOS: `~/Library/Caches/datacode/ml/datasets/`
- Linux: `$XDG_CACHE_HOME/datacode/ml/datasets/` (or `~/.cache/...`)
- Windows: `%LOCALAPPDATA%\datacode\ml\datasets\`

Override the cache location with `DATACODE_ML_DATASETS_DIR`.

### Releasing

Push a tag `vX.Y.Z` that matches `Cargo.toml`. The `release` workflow builds every target, publishes the bundles and `registry-artifacts.json`, and that JSON goes into the `artifacts` field of the `ml` entry in the registry `config.json`.

## Native module `import ml`

The VM loads `libml.dylib` / `libml.so` / `ml.dll` from:

- the script’s `base_path`, or
- **`<dpm>/packages/ml/libml.dylib`** (VM native loader in the main DataCode repo).

To ship a native module, build a `cdylib` with `datacode_module` (see `datacode_sdk` / `define_module!`). Full parity with in-tree tensor APIs requires ABI support for tensor handles; until then a minimal native can expose only `AbiValue`-compatible APIs.

## In-tree copy

The authoritative copy of this logic for the VM lives under `src/lib/ml/` in the [DataCode](https://github.com/igornet0/DataCode) repository.
