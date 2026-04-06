# Compiler metadata (DataCode)

Имена параметров для разрешения **именованных** аргументов в компиляторе `data-code` задаются в **этом репозитории**:

- **Источник правды:** `crates/datacode_ml_compiler/ml_native_named_args.json`
- **Crate:** `crates/datacode_ml_compiler/`

Корневой пакет `data-code` в монорепо подключает `datacode_ml_compiler` по path к этому каталогу. Рантайм-плагин `ml` линкуется с VM только через **ABI + `datacode_sdk`**; этот crate нужен только компилятору хоста.

Интеграция в корень DataCode: [integration/README.md](integration/README.md), патч [integration/datacode_ml_named_args.patch](integration/datacode_ml_named_args.patch).

Структура JSON:

- `param_lists` — ключи сигнатур (`train`, `train_sh`, `freeze`, `native_dataset_split`, …).
- `aliases` — отображение имён (`nn_train` → `train` и т.д.).

Ключ `native_dataset_split` — kwargs для `dataset.split(...)` / `native_dataset_split`; порядок совпадает с [`DATASET_SPLIT_NAMED_ARG_NAMES`](../src/dataset_split_abi.rs).
