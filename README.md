# ML-Datacode-lib

Standalone ML sources for DataCode (mirrors `src/lib/ml` in the main DataCode repo). Published as a **git dependency** for `dpm add ml` via [Datacode-registry-index](https://github.com/igornet0/Datacode-registry-index) (`config.json`).

**Source layout:** see [ARCHITECTURE.md](ARCHITECTURE.md) (`src/vm`, `src/core`, `src/engine`, `src/nn`, `src/gpu`).

## DPM install

```bash
dpm add ml
```

`dpm` resolves `git+...` from the registry and clones into `<env>/packages/ml/`.

## Native module `import ml`

The VM loads `libml.dylib` / `libml.so` / `ml.dll` from:

- the script’s `base_path`, or
- **`<dpm>/packages/ml/libml.dylib`** (VM native loader in the main DataCode repo).

To ship a native module, build a `cdylib` with `datacode_module` (see `datacode_sdk` / `define_module!`). Full parity with in-tree tensor APIs requires ABI support for tensor handles; until then a minimal native can expose only `AbiValue`-compatible APIs.

## In-tree copy

The authoritative copy of this logic for the VM lives under `src/lib/ml/` in the [DataCode](https://github.com/igornet0/DataCode) repository.
