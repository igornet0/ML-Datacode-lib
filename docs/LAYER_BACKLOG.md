# Roadmap: `ml.layer` API

Экспортируемые фабрики задаются в [`src/vm/module_entry.rs`](../src/vm/module_entry.rs) (`dc_fn!("layer.<name>", ...)`) и реализуются в [`src/vm/natives.rs`](../src/vm/natives.rs) как `native_*_layer` + типы в [`src/nn/layer.rs`](../src/nn/layer.rs). Операции и градиенты: [`src/engine/ops.rs`](../src/engine/ops.rs), [`src/engine/autograd.rs`](../src/engine/autograd.rs), [`src/engine/conv.rs`](../src/engine/conv.rs). Режим train/eval для dropout: [`src/forward_mode.rs`](../src/forward_mode.rs) + `NeuralNetwork.training` в [`src/nn/model.rs`](../src/nn/model.rs).

## Реализовано сейчас (полный forward + autograd где нужны веса)

| Имя в `ml.layer` | Примечание |
|------------------|------------|
| `linear`, `dense` | `linear_layer(in, out)` или с bias |
| `relu`, `leaky_relu` | LeakyReLU: без аргументов (α=0.01) или `leaky_relu(alpha)` |
| `sigmoid`, `tanh` | Autograd через `GradOp` |
| `softmax`, `log_softmax` | Ось по умолчанию 1; `softmax_layer(axis)` для axis 0/1 (2D) |
| `gelu`, `softplus`, `elu`, `selu` | Элементные активации |
| `prelu` | Скаляр α |
| `flatten` | |
| `dropout`, `dropout2d`, `dropconnect` | |
| `conv2d` | `conv2d_layer(in_c, out_c, kh, kw)` — stride 1, padding 0 |
| `conv1d` | `conv1d_layer(in_c, out_c, k)` — stride 1, padding 0, bias on; вход [N,C,L] |
| `max_pool2d` | `max_pool2d_layer(kh, kw)` — stride = kernel |
| `max_pool1d` | `max_pool1d_layer(k, stride)` — вход [N,C,L] |
| `avg_pool1d` | `avg_pool1d_layer(k, stride)` |
| `avg_pool2d` | `avg_pool2d_layer(kh, kw, sy, sx)` |
| `global_max_pool` | `global_max_pool_layer()` без аргументов; [N,C,H,W] → [N,C,1,1] |
| `global_avg_pool` | `global_avg_pool_layer()` без аргументов |

## Заглушки (identity forward: выход = вход)

Имена ниже зарегистрированы для `from ml.layer import ...` и проверок `typeof`. **Не использовать в реальных сетях** — forward пока тождественный (вход без изменений), до полной реализации.

`conv3d`, `depthwise_conv2d`, `separable_conv2d`, `transposed_conv2d`, `batch_norm1d`, `batch_norm2d`, `layer_norm`, `instance_norm`, `group_norm`, `rnn`, `lstm`, `gru`, `attention`, `self_attention`, `multihead_attention`, `embedding`, `positional_encoding`, `reshape`, `permute`, `concatenate`, `stack`, `add`, `residual`, `skip_connection`, `upsample`, `upsample_nearest`, `upsample_bilinear`, `graph_conv`, `graph_attention`, `transformer_block`, `feed_forward`.

Тип в реестре: `PlaceholderLayer { name }`.

## В бэклоге (ещё нет настоящего слоя + autograd)

Заменить заглушки на полноценные реализации по приоритету:

- **Свёртки:** `conv3d`, depthwise / separable / transposed (на базе групп и 1×1).
- **Нормализация:** batch / layer / instance / group norm (running stats, train/eval).
- **Рекуррентные:** `rnn`, `gru`, `lstm`.
- **Внимание и трансформер:** `attention`, `multihead_attention`, `transformer_block`, `feed_forward`.
- **Embedding и форма:** `embedding`, `positional_encoding`, `reshape`, `permute`, `concatenate`, `stack` как слои с графом.
- **Residual / upsample / graph:** `add`, `residual`, `skip_connection`, `upsample*`, `graph_*`.

### Согласованность с моделью

- Расширить save/load `.nn` и `ml_model_info` для новых типов по мере реализации.
- Candle/Metal: [`src/gpu/candle_integration.rs`](../src/gpu/candle_integration.rs) — новые CPU-слои при необходимости помечать как «только CPU train_sh».

## Критерий готовности по пункту

Тип `Layer` + forward/backward для обучаемых параметров + `dc_fn!` + при необходимости строки в `model_info`/load + отказ от `PlaceholderLayer` для этого имени.
