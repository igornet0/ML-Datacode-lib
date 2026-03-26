# Интеграция `datacode_ml_compiler` в репозиторий DataCode

Чтобы компилятор разрешал именованные аргументы для `model.train`, `ml.nn_train`, `train_sh`, `freeze` / `unfreeze` и т.д., в корень [DataCode](https://github.com/igornet0/DataCode) нужно внести изменения из патча.

## Требования

- Ветка **ML-Datacode-lib** с workspace и каталогом `crates/datacode_ml_compiler` должна быть в **remote** (например `main`), иначе `git`‑зависимость Cargo не соберётся.
- Локальная разработка без push: см. ниже **path‑зависимость**.

## Применение патча

Из корня клона DataCode:

```bash
git apply /path/to/ML-Datacode-lib/compiler/integration/datacode_ml_named_args.patch
```

или:

```bash
patch -p1 < /path/to/ML-Datacode-lib/compiler/integration/datacode_ml_named_args.patch
```

Затем:

```bash
cargo build --release
```

## Локальная подстановка вместо `git`

В `Cargo.toml` DataCode замените строку `datacode_ml_compiler = { git = ... }` на:

```toml
datacode_ml_compiler = { path = "../ML-Datacode-lib/crates/datacode_ml_compiler" }
```

(путь относительно корня клона DataCode.)

## Проверка

Соберите `libml` и убедитесь, что скрипт с вызовом:

```dc
model.train(x, y, epochs, bs, lr, loss="cross_entropy", optimizer="Adam", x_val=xv, y_val=yv)
```

компилируется без **E0200**.
