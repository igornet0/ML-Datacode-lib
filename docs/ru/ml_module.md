# ML модуль DataCode

Модуль `ml` предоставляет полный набор функций для машинного обучения в DataCode, включая работу с тензорами, создание нейронных сетей, обучение моделей и работу с данными.

**📚 Примеры использования:**
- Базовые примеры: [`examples/ru/11-mnist-mlp/`](../../examples/ru/11-mnist-mlp/)
- Процесс обучения: [Схема обучения нейронной сети](./training_flow.md)
- Формат сохранения моделей: [Формат сохранения моделей](./model_save_format.md)

## Содержание

1. [Введение](#введение)
2. [Tensor операции](#tensor-операции)
3. [Graph операции](#graph-операции)
4. [Linear Regression](#linear-regression)
5. [Optimizers](#optimizers)
6. [Loss функции](#loss-функции)
7. [Dataset функции](#dataset-функции)
8. [Layer функции](#layer-функции)
9. [Neural Network функции](#neural-network-функции)
10. [Методы объектов](#методы-объектов)
11. [Примеры использования](#примеры-использования)

---

## Введение

Модуль `ml` импортируется в начале программы:

```datacode
import ml
```

После импорта доступны все функции модуля через префикс `ml.`, например:
- `ml.tensor()` - создание тензора
- `ml.neural_network()` - создание нейронной сети
- `ml.layer.linear()` - создание линейного слоя

Модуль поддерживает работу на CPU и GPU (Metal для macOS, CUDA для Linux/Windows).

---

## Tensor операции

### `ml.tensor(data, shape?)`

Создает тензор из данных. Форма может быть автоматически определена из структуры данных.

**Аргументы:**
- `data` (array | number) - данные для тензора (массив чисел или вложенные массивы)
- `shape` (array, опционально) - явная форма тензора `[dim1, dim2, ...]`

**Возвращает:** `tensor` - объект тензора

**Примеры:**
```datacode
# Автоматическое определение формы
t1 = ml.tensor([1.0, 2.0, 3.0])  # Форма: [3]
t2 = ml.tensor([[1, 2], [3, 4]])  # Форма: [2, 2]

# Явное указание формы
t3 = ml.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])  # Форма: [2, 2]
```

---

### `ml.shape(tensor)`

Возвращает форму тензора.

**Аргументы:**
- `tensor` (tensor) - тензор

**Возвращает:** `array` - массив размеров по каждому измерению

**Примеры:**
```datacode
t = ml.tensor([[1, 2], [3, 4]])
shape = ml.shape(t)  # [2, 2]
# или
shape = t.shape # [2, 2]
```

---

### `ml.data(tensor)`

Возвращает данные тензора как плоский массив.

**Аргументы:**
- `tensor` (tensor) - тензор

**Возвращает:** `array` - массив чисел

**Примеры:**
```datacode
t = ml.tensor([[1, 2], [3, 4]])
data = ml.data(t)  # [1.0, 2.0, 3.0, 4.0]
```

---

### `ml.add(t1, t2)`

Поэлементное сложение двух тензоров.

**Аргументы:**
- `t1` (tensor) - первый тензор
- `t2` (tensor) - второй тензор

**Возвращает:** `tensor` - результат сложения

**Примеры:**
```datacode
t1 = ml.tensor([1, 2, 3])
t2 = ml.tensor([4, 5, 6])
result = ml.add(t1, t2)  # [5, 7, 9]
# или
result = t1 + t2 # [5, 7, 9]
```

---

### `ml.sub(t1, t2)`

Поэлементное вычитание двух тензоров.

**Аргументы:**
- `t1` (tensor) - первый тензор
- `t2` (tensor) - второй тензор

**Возвращает:** `tensor` - результат вычитания

**Примеры:**
```datacode
t1 = ml.tensor([5, 7, 9])
t2 = ml.tensor([1, 2, 3])
result = ml.sub(t1, t2)  # [4, 5, 6]
# или
result = t1 - t2 # [4, 5, 6]
```

---

### `ml.mul(t1, t2)`

Поэлементное умножение двух тензоров.

**Аргументы:**
- `t1` (tensor) - первый тензор
- `t2` (tensor) - второй тензор

**Возвращает:** `tensor` - результат умножения

**Примеры:**
```datacode
t1 = ml.tensor([2, 3, 4])
t2 = ml.tensor([5, 6, 7])
result = ml.mul(t1, t2)  # [10, 18, 28]
# или
result = t1 * t2 # [10, 18, 28]
```

---

### `ml.matmul(t1, t2)`

Матричное умножение двух тензоров.

**Аргументы:**
- `t1` (tensor) - первый тензор (форма `[n, m]`)
- `t2` (tensor) - второй тензор (форма `[m, k]`)

**Возвращает:** `tensor` - результат матричного умножения (форма `[n, k]`)

**Примеры:**
```datacode
t1 = ml.tensor([[1, 2], [3, 4]])  # [2, 2]
t2 = ml.tensor([[5, 6], [7, 8]])  # [2, 2]
result = ml.matmul(t1, t2)  # [[19, 22], [43, 50]]
# или
result = t1 @ t2 # [[19, 22], [43, 50]]
```

---

### `ml.transpose(tensor)`

Транспонирование тензора.

**Аргументы:**
- `tensor` (tensor) - тензор (форма `[n, m]`)

**Возвращает:** `tensor` - транспонированный тензор (форма `[m, n]`)

**Примеры:**
```datacode
t = ml.tensor([[1, 2, 3], [4, 5, 6]])  # [2, 3]
result = ml.transpose(t)  # [[1, 4], [2, 5], [3, 6]] - форма [3, 2]
# или
result = t.T # [[1, 4], [2, 5], [3, 6]] - форма [3, 2]
```

---

### `ml.sum(tensor)`

Сумма всех элементов тензора.

**Аргументы:**
- `tensor` (tensor) - тензор

**Возвращает:** `tensor` - сумма всех рядов элементов

**Примеры:**
```datacode
t = ml.tensor([[1, 2], [3, 4]])
s = sum(t)  # [[3], [7]]
```

---

### `ml.mean(tensor)`

Среднее значение всех элементов тензора.

**Аргументы:**
- `tensor` (tensor) - тензор

**Возвращает:** `number` - среднее значение для каждого ряда

**Примеры:**
```datacode
t = ml.tensor([[1, 2], [3, 4]])
m = average(t)  # [[1.5], [3.5]]
```

---

### `ml.max_idx(tensor)`

Находит индексы максимальных элементов. Для 1D тензоров возвращает один индекс, для многомерных - массив индексов для каждого среза по первому измерению.

**Аргументы:**
- `tensor` (tensor) - тензор

**Возвращает:** `array` - массив индексов

**Примеры:**
```datacode
t = ml.tensor([3, 1, 4, 1, 5])
idx = ml.max_idx(t)  # [4] - индекс максимального элемента
```

---

### `ml.min_idx(tensor)`

Находит индексы минимальных элементов. Для 1D тензоров возвращает один индекс, для многомерных - массив индексов для каждого среза по первому измерению.

**Аргументы:**
- `tensor` (tensor) - тензор

**Возвращает:** `array` - массив индексов

**Примеры:**
```datacode
t = ml.tensor([3, 1, 4, 1, 5])
idx = ml.min_idx(t)  # [1] или [3] - индекс минимального элемента
```

---

## Graph операции

Граф вычислений используется для автоматического дифференцирования (autograd).

### `ml.graph()`

Создает новый граф вычислений.

**Аргументы:** нет

**Возвращает:** `graph` - объект графа

**Примеры:**
```datacode
g = ml.graph()
```

---

### `ml.graph_add_input(graph)`

Добавляет входной узел в граф.

**Аргументы:**
- `graph` (graph) - граф вычислений

**Возвращает:** `number` - ID узла

**Примеры:**
```datacode
g = ml.graph()
input_id = ml.graph_add_input(g)
```

---

### `ml.graph_add_op(graph, op_name, input_node_ids)`

Добавляет операцию в граф.

**Аргументы:**
- `graph` (graph) - граф вычислений
- `op_name` (string) - имя операции: `"add"`, `"sub"`, `"mul"`, `"matmul"`, `"transpose"`, `"sum"`, `"mean"`
- `input_node_ids` (array) - массив ID входных узлов

**Возвращает:** `number` - ID нового узла

**Примеры:**
```datacode
g = ml.graph()
input1 = ml.graph_add_input(g)
input2 = ml.graph_add_input(g)
add_node = ml.graph_add_op(g, "add", [input1, input2])
```

---

### `ml.graph_forward(graph, input_tensors)`

Выполняет прямой проход через граф.

**Аргументы:**
- `graph` (graph) - граф вычислений
- `input_tensors` (array) - массив входных тензоров

**Возвращает:** `null`

**Примеры:**
```datacode
g = ml.graph()
input_id = ml.graph_add_input(g)
t1 = ml.tensor([1, 2, 3])
ml.graph_forward(g, [t1])
```

---

### `ml.graph_get_output(graph, node_id)`

Получает выходной тензор узла после прямого прохода.

**Аргументы:**
- `graph` (graph) - граф вычислений
- `node_id` (number) - ID узла

**Возвращает:** `tensor` - выходной тензор

**Примеры:**
```datacode
output = ml.graph_get_output(g, node_id)
```

---

### `ml.graph_backward(graph, output_node_id)`

Выполняет обратный проход (backpropagation) для вычисления градиентов.

**Аргументы:**
- `graph` (graph) - граф вычислений
- `output_node_id` (number) - ID выходного узла

**Возвращает:** `null`

**Примеры:**
```datacode
ml.graph_backward(g, output_node_id)
```

---

### `ml.graph_get_gradient(graph, node_id)`

Получает градиент узла после обратного прохода.

**Аргументы:**
- `graph` (graph) - граф вычислений
- `node_id` (number) - ID узла

**Возвращает:** `tensor` - градиент

**Примеры:**
```datacode
grad = ml.graph_get_gradient(g, node_id)
```

---

### `ml.graph_zero_grad(graph)`

Обнуляет все градиенты в графе.

**Аргументы:**
- `graph` (graph) - граф вычислений

**Возвращает:** `null`

**Примеры:**
```datacode
ml.graph_zero_grad(g)
```

---

### `ml.graph_set_requires_grad(graph, node_id, requires_grad)`

Устанавливает, требуется ли вычисление градиента для узла.

**Аргументы:**
- `graph` (graph) - граф вычислений
- `node_id` (number) - ID узла
- `requires_grad` (bool) - требуется ли градиент

**Возвращает:** `null`

**Примеры:**
```datacode
ml.graph_set_requires_grad(g, node_id, true)
```

---

## Linear Regression

### `ml.linear_regression(feature_count)`

Создает модель линейной регрессии.

**Аргументы:**
- `feature_count` (number) - количество признаков

**Возвращает:** `linear_regression` - объект модели

**Примеры:**
```datacode
model = ml.linear_regression(3)  # 3 признака
```

---

### `ml.lr_predict(model, features)`

Предсказание значений для линейной регрессии.

**Аргументы:**
- `model` (linear_regression) - модель
- `features` (tensor) - признаки (форма `[batch_size, feature_count]`)

**Возвращает:** `tensor` - предсказания (форма `[batch_size, 1]`)

**Примеры:**
```datacode
x = ml.tensor([[1, 2, 3], [4, 5, 6]])  # [2, 3]
predictions = ml.lr_predict(model, x)
```

---

### `ml.lr_train(model, x, y, epochs, lr)`

Обучение модели линейной регрессии.

**Аргументы:**
- `model` (linear_regression) - модель
- `x` (tensor) - признаки (форма `[batch_size, feature_count]`)
- `y` (tensor) - целевые значения (форма `[batch_size, 1]`)
- `epochs` (number) - количество эпох
- `lr` (number) - скорость обучения

**Возвращает:** `array` - история потерь

**Примеры:**
```datacode
loss_history = ml.lr_train(model, x_train, y_train, 100, 0.01)
```

---

### `ml.lr_evaluate(model, x, y)`

Оценка модели линейной регрессии (вычисляет MSE).

**Аргументы:**
- `model` (linear_regression) - модель
- `x` (tensor) - признаки
- `y` (tensor) - целевые значения

**Возвращает:** `number` - MSE (Mean Squared Error)

**Примеры:**
```datacode
mse = ml.lr_evaluate(model, x_test, y_test)
```

---

## Optimizers

### `ml.sgd(learning_rate)`

Создает оптимизатор SGD (Stochastic Gradient Descent).

**Аргументы:**
- `learning_rate` (number) - скорость обучения

**Возвращает:** `sgd` - объект оптимизатора

**Примеры:**
```datacode
optimizer = ml.sgd(0.001)
```

---

### `ml.sgd_step(optimizer, graph, param_node_ids)`

Выполняет один шаг оптимизации SGD.

**Аргументы:**
- `optimizer` (sgd) - оптимизатор
- `graph` (graph) - граф вычислений
- `param_node_ids` (array) - массив ID узлов параметров

**Возвращает:** `null`

**Примеры:**
```datacode
ml.sgd_step(optimizer, graph, [weight_id, bias_id])
```

---

### `ml.sgd_zero_grad(graph)`

Обнуляет градиенты в графе (удобная функция для SGD).

**Аргументы:**
- `graph` (graph) - граф вычислений

**Возвращает:** `null`

**Примеры:**
```datacode
ml.sgd_zero_grad(graph)
```

---

### `ml.adam(learning_rate, beta1?, beta2?, epsilon?)`

Создает оптимизатор Adam.

**Аргументы:**
- `learning_rate` (number) - скорость обучения
- `beta1` (number, опционально) - коэффициент для первого момента (по умолчанию 0.9)
- `beta2` (number, опционально) - коэффициент для второго момента (по умолчанию 0.999)
- `epsilon` (number, опционально) - малое значение для численной стабильности (по умолчанию 1e-8)

**Возвращает:** `adam` - объект оптимизатора

**Примеры:**
```datacode
optimizer = ml.adam(0.001)  # С параметрами по умолчанию
optimizer2 = ml.adam(0.001, 0.9, 0.999, 1e-8)  # С явными параметрами
```

---

### `ml.adam_step(optimizer, graph, param_node_ids)`

Выполняет один шаг оптимизации Adam.

**Аргументы:**
- `optimizer` (adam) - оптимизатор
- `graph` (graph) - граф вычислений
- `param_node_ids` (array) - массив ID узлов параметров

**Возвращает:** `null`

**Примеры:**
```datacode
ml.adam_step(optimizer, graph, [weight_id, bias_id])
```

---

## Loss функции

### `ml.mse_loss(y_pred, y_true)`

Вычисляет Mean Squared Error (среднеквадратичную ошибку).

**Аргументы:**
- `y_pred` (tensor) - предсказанные значения
- `y_true` (tensor) - истинные значения

**Возвращает:** `tensor` - тензор потерь

**Примеры:**
```datacode
loss = ml.mse_loss(predictions, targets)
```

---

### `ml.cross_entropy_loss(logits, class_indices)`

**УСТАРЕЛО**: Используйте `sparse_softmax_cross_entropy_loss` или `categorical_cross_entropy_loss`.

---

### `ml.binary_cross_entropy_loss(y_pred, y_true)`

Вычисляет Binary Cross Entropy Loss (для бинарной классификации).

**Аргументы:**
- `y_pred` (tensor) - предсказанные вероятности (форма `[batch_size, 1]`)
- `y_true` (tensor) - истинные метки (форма `[batch_size, 1]`, значения 0 или 1)

**Возвращает:** `tensor` - тензор потерь

**Примеры:**
```datacode
loss = ml.binary_cross_entropy_loss(predictions, targets)
```

---

### `ml.mae_loss(y_pred, y_true)`

Вычисляет Mean Absolute Error (среднюю абсолютную ошибку).

**Аргументы:**
- `y_pred` (tensor) - предсказанные значения
- `y_true` (tensor) - истинные значения

**Возвращает:** `tensor` - тензор потерь

**Примеры:**
```datacode
loss = ml.mae_loss(predictions, targets)
```

---

### `ml.huber_loss(y_pred, y_true, delta?)`

Вычисляет Huber Loss (комбинация MSE и MAE).

**Аргументы:**
- `y_pred` (tensor) - предсказанные значения
- `y_true` (tensor) - истинные значения
- `delta` (number, опционально) - пороговое значение (по умолчанию 1.0)

**Возвращает:** `tensor` - тензор потерь

**Примеры:**
```datacode
loss = ml.huber_loss(predictions, targets, 1.0)
```

---

### `ml.hinge_loss(y_pred, y_true)`

Вычисляет Hinge Loss (для SVM).

**Аргументы:**
- `y_pred` (tensor) - предсказанные значения
- `y_true` (tensor) - истинные значения (обычно -1 или 1)

**Возвращает:** `tensor` - тензор потерь

**Примеры:**
```datacode
loss = ml.hinge_loss(predictions, targets)
```

---

### `ml.kl_divergence(y_pred, y_true)`

Вычисляет Kullback-Leibler Divergence.

**Аргументы:**
- `y_pred` (tensor) - предсказанные распределения вероятностей
- `y_true` (tensor) - истинные распределения вероятностей

**Возвращает:** `tensor` - тензор потерь

**Примеры:**
```datacode
loss = ml.kl_divergence(predictions, targets)
```

---

### `ml.smooth_l1_loss(y_pred, y_true)`

Вычисляет Smooth L1 Loss (комбинация L1 и L2).

**Аргументы:**
- `y_pred` (tensor) - предсказанные значения
- `y_true` (tensor) - истинные значения

**Возвращает:** `tensor` - тензор потерь

**Примеры:**
```datacode
loss = ml.smooth_l1_loss(predictions, targets)
```

---

### `ml.categorical_cross_entropy_loss(logits, targets)`

Вычисляет Categorical Cross Entropy Loss для многоклассовой классификации с one-hot метками.

**Аргументы:**
- `logits` (tensor) - логиты модели (форма `[batch_size, num_classes]`)
- `targets` (tensor) - one-hot метки (форма `[batch_size, num_classes]`)

**Возвращает:** `tensor` - тензор потерь

**Примеры:**
```datacode
# Преобразование меток в one-hot
y_onehot = ml.onehots(y_train, 10)
loss = ml.categorical_cross_entropy_loss(logits, y_onehot)
```

**Примечание:** Для меток в виде индексов классов используйте `sparse_softmax_cross_entropy_loss` через `model.train()` с `loss="cross_entropy"`.

---

## Dataset функции

### `ml.dataset(table, feature_columns, target_columns)`

Создает датасет из таблицы.

**Аргументы:**
- `table` (table) - таблица данных
- `feature_columns` (array) - массив имен колонок признаков
- `target_columns` (array) - массив имен колонок целевых переменных

**Возвращает:** `dataset` - объект датасета

**Примеры:**
```datacode
ds = ml.dataset(data_table, ["feature1", "feature2"], ["target"])
```

---

### `ml.dataset_features(dataset)`

Извлекает тензор признаков из датасета.

**Аргументы:**
- `dataset` (dataset) - датасет

**Возвращает:** `tensor` - тензор признаков (форма `[num_samples, num_features]`)

**Примеры:**
```datacode
x = ml.dataset_features(dataset)
```

---

### `ml.dataset_targets(dataset)`

Извлекает тензор целевых переменных из датасета.

**Аргументы:**
- `dataset` (dataset) - датасет

**Возвращает:** `tensor` - тензор целевых переменных (форма `[num_samples, num_targets]`)

**Примеры:**
```datacode
y = ml.dataset_targets(dataset)
```

---

### `ml.onehots(labels, num_classes?)`

Преобразует метки классов в one-hot кодирование (батч).

**Аргументы:**
- `labels` (tensor) - метки классов (форма `[N]` или `[N, 1]`)
- `num_classes` (number, опционально) - количество классов (если не указано, определяется как `max(labels) + 1`)

**Возвращает:** `tensor` - one-hot тензор (форма `[N, num_classes]`)

**Примеры:**
```datacode
labels = ml.tensor([[0], [1], [2], [0]])  # [4, 1]
y_onehot = ml.onehots(labels, 3)  # [4, 3]
# Результат: [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]
```

---

### `ml.one_hot(class_index, num_classes)`

Одна строка one-hot: `1.0` в позиции `class_index`, остальные нули.

**Аргументы:**
- `class_index` (number) - индекс в `[0, num_classes)`, где ставится `1.0`
- `num_classes` (number) - длина вектора (число классов)

**Возвращает:** `tensor` - форма `[1, num_classes]`

**Примеры:**
```datacode
t = ml.one_hot(1, 10)  # [1, 10], единица во второй позиции
```

---

### `ml.load_mnist(split)`

Загружает датасет MNIST.

**Аргументы:**
- `split` (string) - раздел датасета: `"train"` или `"test"`

**Возвращает:** `dataset` - датасет MNIST

**Примеры:**
```datacode
train_dataset = ml.load_mnist("train")
test_dataset = ml.load_mnist("test")
```

**Примечание:** Файлы MNIST должны находиться в `src/lib/ml/datasets/mnist/`.

---

## Layer функции

Слои создаются через подмодуль `ml.layer`:

### `ml.layer.linear(in_features, out_features)`

Создает линейный (полносвязный) слой.

**Аргументы:**
- `in_features` (number) - количество входных признаков
- `out_features` (number) - количество выходных признаков

**Возвращает:** `layer` - объект слоя

**Примеры:**
```datacode
layer = ml.layer.linear(784, 128)  # 784 входа, 128 выходов
```

---

### `ml.layer.relu()`

Создает слой активации ReLU.

**Аргументы:** нет

**Возвращает:** `layer` - объект слоя

**Примеры:**
```datacode
relu_layer = ml.layer.relu()
```

---

### `ml.layer.softmax()`

Создает слой активации Softmax.

**Аргументы:** нет

**Возвращает:** `layer` - объект слоя

**Примеры:**
```datacode
softmax_layer = ml.layer.softmax()
```

---

### `ml.layer.flatten()`

Создает слой Flatten (преобразует многомерный тензор в одномерный).

**Аргументы:** нет

**Возвращает:** `layer` - объект слоя

**Примеры:**
```datacode
flatten_layer = ml.layer.flatten()
```

---

## Neural Network функции

### `ml.sequential(layers)`

Создает последовательный контейнер слоев.

**Аргументы:**
- `layers` (array) - массив слоев

**Возвращает:** `sequential` - объект последовательного контейнера

**Примеры:**
```datacode
layer1 = ml.layer.linear(784, 128)
layer2 = ml.layer.relu()
layer3 = ml.layer.linear(128, 10)
seq = ml.sequential([layer1, layer2, layer3])
```

---

### `ml.sequential_add(sequential, layer)`

Добавляет слой в последовательный контейнер.

**Аргументы:**
- `sequential` (sequential) - последовательный контейнер
- `layer` (layer) - слой для добавления

**Возвращает:** `null`

**Примеры:**
```datacode
ml.sequential_add(seq, ml.layer.softmax())
```

---

### `ml.neural_network(sequential)`

Создает нейронную сеть из последовательного контейнера.

**Аргументы:**
- `sequential` (sequential) - последовательный контейнер слоев

**Возвращает:** `neural_network` - объект нейронной сети

**Примеры:**
```datacode
model = ml.neural_network(seq)
```

---

### `ml.nn_forward(model, x)`

Выполняет прямой проход через нейронную сеть.

**Аргументы:**
- `model` (neural_network | sequential | linear_regression) - модель
- `x` (tensor) - входные данные (форма `[batch_size, input_features]`)

**Возвращает:** `tensor` - выходные данные

**Примеры:**
```datacode
output = ml.nn_forward(model, x)
```

---

### `ml.nn_train(model, x, y, epochs, batch_size, lr, loss_type, optimizer?, x_val?, y_val?)`

Обучает нейронную сеть.

**Аргументы:**
- `model` (neural_network) - модель
- `x` (tensor) - обучающие признаки (форма `[num_samples, num_features]`)
- `y` (tensor) - обучающие метки (форма `[num_samples, 1]` для `cross_entropy` или `[num_samples, num_classes]` для `categorical_cross_entropy`)
- `epochs` (number) - количество эпох
- `batch_size` (number) - размер батча
- `lr` (number) - скорость обучения
- `loss_type` (string) - тип функции потерь: `"cross_entropy"`, `"categorical_cross_entropy"`, `"mse"`, `"binary_cross_entropy"`
- `optimizer` (string, опционально) - оптимизатор: `"SGD"`, `"Adam"`, `"Momentum"`, `"NAG"`, `"Adagrad"`, `"RMSprop"`, `"AdamW"` (по умолчанию `"SGD"`)
- `x_val` (tensor, опционально) - валидационные признаки
- `y_val` (tensor, опционально) - валидационные метки

**Возвращает:** `array` - история потерь

**Примеры:**
```datacode
# Базовое обучение
loss_history = ml.nn_train(model, x_train, y_train, 10, 32, 0.001, "cross_entropy")

# С оптимизатором и валидацией
loss_history = ml.nn_train(model, x_train, y_train, 10, 32, 0.001, 
                           "cross_entropy", "Adam", x_val, y_val)
```

---

### `ml.nn_train_sh(model, x, y, epochs, batch_size, lr, loss_type, optimizer, monitor, patience, min_delta, restore_best, x_val?, y_val?)`

Обучает нейронную сеть с early stopping и планировщиком скорости обучения.

**Аргументы:**
- `model` (neural_network) - модель
- `x` (tensor) - обучающие признаки
- `y` (tensor) - обучающие метки
- `epochs` (number) - максимальное количество эпох
- `batch_size` (number) - размер батча
- `lr` (number) - начальная скорость обучения
- `loss_type` (string) - тип функции потерь
- `optimizer` (string | null) - оптимизатор (или `null` для SGD по умолчанию)
- `monitor` (string) - метрика для мониторинга: `"loss"`, `"val_loss"`, `"acc"`, `"val_acc"`
- `patience` (number) - количество эпох без улучшения перед остановкой
- `min_delta` (number) - минимальное изменение для улучшения
- `restore_best` (bool) - восстанавливать ли лучшие веса после остановки
- `x_val` (tensor, опционально) - валидационные признаки
- `y_val` (tensor, опционально) - валидационные метки

**Возвращает:** `object` - объект с историей обучения:
- `loss` (array) - история потерь на обучении
- `val_loss` (array | null) - история потерь на валидации
- `acc` (array) - история точности на обучении
- `val_acc` (array | null) - история точности на валидации
- `lr` (array) - история скорости обучения
- `best_metric` (number) - лучшее значение метрики
- `best_epoch` (number) - эпоха с лучшей метрикой
- `stopped_epoch` (number) - эпоха остановки

**Примеры:**
```datacode
history = ml.nn_train_sh(model, x_train, y_train, 50, 32, 0.001,
                         "cross_entropy", "Adam", "val_loss", 5, 0.0001, true,
                         x_val, y_val)
print("Лучшая эпоха:", history.best_epoch)
print("Лучшая метрика:", history.best_metric)
```

---

### `ml.nn_save(model, path)`

Сохраняет нейронную сеть в файл.

**Аргументы:**
- `model` (neural_network) - модель
- `path` (path | string) - путь к файлу

**Возвращает:** `null`

**Примеры:**
```datacode
ml.nn_save(model, "model.nn")
ml.nn_save(model, path("models/my_model.nn"))
```

---

### `ml.nn_load(path)`

Загружает нейронную сеть из файла.

**Аргументы:**
- `path` (path | string) - путь к файлу

**Возвращает:** `neural_network` - загруженная модель

**Примеры:**
```datacode
model = ml.nn_load("model.nn")
model = ml.nn_load(path("models/my_model.nn"))
```

---

### `ml.save_model(model, path)`

Альтернативное имя для `ml.nn_save()`.

**Аргументы:**
- `model` (neural_network) - модель
- `path` (path | string) - путь к файлу

**Возвращает:** `null`

---

### `ml.load(path)`

Альтернативное имя для `ml.nn_load()`.

**Аргументы:**
- `path` (path | string) - путь к файлу

**Возвращает:** `neural_network` - загруженная модель

---

### `ml.set_device(device_name)`

Устанавливает устройство по умолчанию для ML операций.

**Аргументы:**
- `device_name` (string) - имя устройства: `"cpu"`, `"cuda"`, `"metal"`, `"auto"`

**Возвращает:** `string` - имя установленного устройства

**Примеры:**
```datacode
ml.set_device("metal")  # macOS
ml.set_device("cuda")   # Linux/Windows с NVIDIA GPU
ml.set_device("cpu")    # CPU
```

**Примечание:** Если GPU устройство недоступно, автоматически выполняется откат на CPU.

---

### `ml.get_device()`

Возвращает текущее устройство по умолчанию.

**Аргументы:** нет

**Возвращает:** `string` - имя устройства

**Примеры:**
```datacode
device = ml.get_device()  # "cpu", "metal", "cuda"
```

---

### `ml.nn_set_device(model, device_name)`

Устанавливает устройство для конкретной модели.

**Аргументы:**
- `model` (neural_network) - модель
- `device_name` (string) - имя устройства

**Возвращает:** `string` - имя установленного устройства

**Примеры:**
```datacode
model.device("metal")
```

---

### `ml.nn_get_device(model)`

Возвращает устройство модели.

**Аргументы:**
- `model` (neural_network) - модель

**Возвращает:** `string` - имя устройства

**Примеры:**
```datacode
device = ml.nn_get_device(model)
```

---

### `ml.validate_model(model)`

Проверяет валидность модели.

**Аргументы:**
- `model` (neural_network) - модель

**Возвращает:** `bool` - `true` если модель валидна, `false` иначе

**Примеры:**
```datacode
if ml.validate_model(model) {
    print("Модель валидна")
}
```

---

### `ml.model_info(model, verbose?, format?, show_graph?)`

Выводит информацию о модели.

**Аргументы:**
- `model` (neural_network | linear_regression) - модель
- `verbose` (bool, опционально) - показывать ли детальную информацию (по умолчанию `false`)
- `format` (string, опционально) - формат вывода: `"text"` или `"json"` (по умолчанию `"text"`)
- `show_graph` (bool, опционально) - показывать ли граф вычислений (по умолчанию `false`)

**Возвращает:** `null` для формата `"text"`, `string` (JSON) для формата `"json"`

**Примеры:**
```datacode
ml.model_info(model)  # Текстовая информация
ml.model_info(model, true, "json")  # JSON формат с деталями
```

---

### `ml.model_get_layer(model, index)`

Получает слой модели по индексу.

**Аргументы:**
- `model` (neural_network) - модель
- `index` (number) - индекс слоя (начиная с 0)

**Возвращает:** `layer` - объект слоя

**Примеры:**
```datacode
first_layer = ml.model_get_layer(model, 0)
```

---

### `ml.layer_freeze(layer)`

Замораживает слой (отключает обновление параметров при обучении).

**Аргументы:**
- `layer` (layer) - слой

**Возвращает:** `null`

**Примеры:**
```datacode
ml.layer_freeze(first_layer)
```

---

### `ml.layer_unfreeze(layer)`

Размораживает слой (включает обновление параметров при обучении).

**Аргументы:**
- `layer` (layer) - слой

**Возвращает:** `null`

**Примеры:**
```datacode
ml.layer_unfreeze(first_layer)
```

---

## Методы объектов

### Tensor

Тензоры имеют следующие свойства и методы:

#### Свойства

- **`tensor.shape`** - возвращает форму тензора как массив
- **`tensor.data`** - возвращает данные тензора как плоский массив

#### Методы

- **`tensor.max_idx()`** - возвращает индекс максимального элемента
- **`tensor.min_idx()`** - возвращает индекс минимального элемента

#### Индексация

- **`tensor[i]`** - доступ к элементу по индексу:
  - Для 1D тензоров: возвращает скалярное значение
  - Для многомерных тензоров: возвращает срез по первому измерению

**Примеры:**
```datacode
t = ml.tensor([[1, 2, 3], [4, 5, 6]])

# Свойства
shape = t.shape  # [2, 3]
data = t.data    # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

# Методы
max_idx = t.max_idx()  # [5] - индекс максимального элемента
min_idx = t.min_idx()  # [0] - индекс минимального элемента

# Индексация
first_row = t[0]  # тензор [1, 2, 3]
second_row = t[1]  # тензор [4, 5, 6]
```

---

### NeuralNetwork

Нейронные сети имеют следующие методы:

#### Методы

- **`model.train(x, y, epochs, batch_size, lr, loss_type, optimizer?, x_val?, y_val?)`** - обучение модели
- **`model.train_sh(x, y, epochs, batch_size, lr, loss_type, optimizer, monitor, patience, min_delta, restore_best, x_val?, y_val?)`** - обучение с early stopping
- **`model.save(path)`** - сохранение модели
- **`model.device(device_name)`** - установка устройства (аналог `ml.nn_set_device()`)
- **`model.get_device()`** - получение устройства (аналог `ml.nn_get_device()`)

#### Свойства

- **`model.layers[i]`** - доступ к слою по индексу

**Примеры:**
```datacode
# Обучение
loss_history = model.train(x_train, y_train, 10, 32, 0.001, "cross_entropy")

# Обучение с валидацией и оптимизатором
loss_history = model.train(x_train, y_train, 10, 32, 0.001, 
                           "cross_entropy", "Adam", x_val, y_val)

# Сохранение
model.save("my_model.nn")

# Установка устройства
model.device("metal")

# Доступ к слоям
first_layer = model.layers[0]
```

---

### Dataset

Датасеты поддерживают индексацию:

- **`dataset[i]`** - возвращает `[features, target]` для i-го образца

**Примеры:**
```datacode
dataset = ml.load_mnist("train")
sample = dataset[0]  # [features_tensor, target_value]
features = sample[0]
target = sample[1]
```

---

## Примеры использования

### Полный пример обучения MLP на MNIST

```datacode
import ml

# Загрузка данных
train_dataset = ml.load_mnist("train")
test_dataset = ml.load_mnist("test")

x_train = ml.dataset_features(train_dataset)  # [60000, 784]
y_train = ml.dataset_targets(train_dataset)   # [60000, 1]

x_test = ml.dataset_features(test_dataset)    # [10000, 784]
y_test = ml.dataset_targets(test_dataset)     # [10000, 1]

# Создание модели
layer1 = ml.layer.linear(784, 128)
layer2 = ml.layer.relu()
layer3 = ml.layer.linear(128, 10)
model_seq = ml.sequential([layer1, layer2, layer3])
model = ml.neural_network(model_seq)

# Установка устройства (опционально)
model.device("metal")  # или "cuda" для Linux/Windows

# Обучение
loss_history = model.train(x_train, y_train, 
                           10, 64, 0.001, 
                           "cross_entropy", "Adam", x_test, y_test)

# Предсказание
predictions = ml.nn_forward(model, x_test)

# Сохранение модели
model.save("mnist_model.nn")
```

### Пример с early stopping

```datacode
import ml

# ... создание модели и загрузка данных ...

# Обучение с early stopping
history = model.train_sh(x_train, y_train, 50, 32, 0.001,
                         "cross_entropy", "Adam", "val_loss", 5, 0.0001, true,
                         x_val, y_val)

print("Обучение остановлено на эпохе:", history.stopped_epoch)
print("Лучшая эпоха:", history.best_epoch)
print("Лучшая метрика:", history.best_metric)
```

### Пример работы с тензорами

```datacode
import ml

# Создание тензоров
a = ml.tensor([[1, 2], [3, 4]])
b = ml.tensor([[5, 6], [7, 8]])

# Операции
c = ml.add(a, b)        # Поэлементное сложение
d = ml.matmul(a, b)     # Матричное умножение
e = ml.transpose(a)     # Транспонирование

# Свойства и методы
shape = a.shape         # [2, 2]
data = a.data           # [1.0, 2.0, 3.0, 4.0]
max_idx = a.max_idx()   # [3]
sum_val = ml.sum(a)     # 10.0
mean_val = ml.mean(a)   # 2.5
```

### Пример линейной регрессии

```datacode
import ml

# Создание модели
model = ml.linear_regression(3)  # 3 признака

# Обучение
x_train = ml.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y_train = ml.tensor([[6], [15], [24]])
loss_history = ml.lr_train(model, x_train, y_train, 100, 0.01)

# Предсказание
x_test = ml.tensor([[2, 3, 4]])
prediction = ml.lr_predict(model, x_test)

# Оценка
mse = ml.lr_evaluate(model, x_test, y_test)
```

---

## Связанные документы

- [Схема обучения нейронной сети](./training_flow.md) - подробное описание процесса обучения
- [Формат сохранения моделей](./model_save_format.md) - описание формата файлов моделей
- [Встроенные функции](./builtin_functions.md) - другие встроенные функции DataCode
- [Примеры MNIST MLP](../../examples/ru/11-mnist-mlp/) - практические примеры использования

---

**Примечание:** Данная документация создана на основе анализа исходного кода DataCode. Все функции протестированы и работают в текущей версии языка.

