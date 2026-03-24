# Compiler metadata (DataCode)

`ml_native_named_args.json` — имена параметров для разрешения **именованных и позиционных** аргументов в компиляторе `data-code` (через crate `datacode_ml_compiler`).

Источник правды: этот JSON + crate `crates/datacode_ml_compiler`. В корневом пакете `data-code` (`src/compiler/natives.rs`) вызывается `datacode_ml_compiler::native_named_arg_params`.

Структура:

- `param_lists` — ключи логических сигнатур (`train`, `train_sh`, `freeze`, `unfreeze`, …) и упорядоченные имена параметров. Для `train` / `train_sh` первый параметр — `nn` (объект модели для `model.train(...)`). Пустой массив — только receiver, без явных аргументов (методы слоя `freeze` / `unfreeze`).
- `aliases` — какие имена функций/методов в языке отображаются на какую сигнатуру (`nn_train` → `train` и т.д.).
