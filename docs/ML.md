ML-модуль U-Ray

Справочник по math, linalg, nn и рецептам обучения.

Содержание

1. math
2. linalg
3. nn: shape
4. nn: активации
5. nn: слои и forward
6. nn: backprop-примитивы
7. nn: high-level API
8. nn: autodiff и составляемые слои
9. nn: загрузка наборов данных
10. nn: оптимизаторы
11. nn: сохранение моделей
12. Контракт skip_last_act
13. Состояние оптимизатора и save/load
14. Численная устойчивость
15. Формат JSON
16. Рецепты

---

math

Функция Пример Результат
math.pi  3.141592653589793
math.e  2.718281828459045
math.sqrt(x) sqrt(2) 1.4142135623730951
math.exp(x) exp(0) 1.0
math.log(x) log(1) 0.0
math.abs(x) abs(-7) 7.0
math.sin(x) sin(0) 0.0
math.cos(x) cos(0) 1.0
math.rand()  случайное [0, 1)

Все math.* возвращают float.

---

linalg

Функция Что делает
dot(a, b) Скалярное произведение двух векторов
matmul(A, B) Умножение матриц
transpose(A) Транспонирование
zeros(n) Вектор из n нулей (0.0)
ones(n) Вектор из n единиц (1.0)
sum(a) Сумма элементов
max(a) Максимум
min(a) Минимум

---

nn: shape

nn.shape(x)

Возвращает форму структуры:

· nn.shape([1, 2, 3]) → [3] (вектор)
· nn.shape([[1, 2], [3, 4]]) → [2, 2] (матрица 2×2)
· nn.shape([]) → [0] (пустой список)

Для матрицы проверяет, что все строки одинаковой длины. Используется в forward/dense для явной валидации.

---

nn: активации

Функция Что делает
nn.relu(x) max(0, x)
nn.leaky_relu(x) x если x > 0, иначе 0.1 * x
nn.sigmoid(x) 1 / (1 + exp(-x)), стабильная
nn.relu_vec(v) Поэлементно для вектора
nn.leaky_relu_vec(v) Поэлементно для вектора
nn.sigmoid_vec(v) Поэлементно для вектора
nn.softmax(v) Нормирует вектор в распределение (log-sum-exp)

---

nn: слои и forward

nn.dense(in_dim, out_dim)

Создаёт слой — словарь {"W": [[...]], "b": [...]}:

· W — матрица out_dim × in_dim, веса случайные (Xavier/He)
· b — вектор длины out_dim, нули

```
layer = nn.dense(2, 4)
print(layer)  // {"W": [[...], ...], "b": [0.0, 0.0, 0.0, 0.0]}
```

nn.forward(layer, x)

Прямой проход одного слоя: W·x + b. Возвращает вектор длины out_dim.

nn.mse(yhat, y)

Среднеквадратичная ошибка по двум векторам.

nn.bce(yhat, y)

Бинарная кросс-энтропия (для вероятностей после сигмоиды). Клиппинг eps = 1e-12.

nn.categorical_cross_entropy(yhat, y)

Категориальная кросс-энтропия (для вероятностей после softmax). Клиппинг eps = 1e-12.

---

nn: backprop-примитивы

nn.forward_dense(layer, x)

Прямой проход с кэшем. Возвращает [z, x].

nn.backward_dense(layer, x, grad_z)

Градиенты одного слоя. Возвращает [grad_W, grad_b, grad_x].

nn.mse_grad(yhat, y)

Градиент MSE по yhat: 2 * (yhat - y) / len(yhat).

nn.bce_grad(yhat, y)

Градиент BCE по z (для BCE+sigmoid): (yhat - y) / len(yhat).

nn.cce_grad(yhat, y)

Градиент CCE по z (для CCE+softmax): (yhat - y) / len(yhat).

nn.sigmoid_grad(z)

sigmoid(z) * (1 - sigmoid(z)).

nn.relu_grad(z)

1.0 если z > 0, иначе 0.0.

---

nn: high-level API

nn.mlp(sizes, act, out_act)

Создаёт многослойную сеть.

· sizes — список размеров: [in, h1, h2, ..., out]
· act — активация промежуточных слоёв: "sigmoid", "relu", "leaky_relu" (по умолчанию "leaky_relu")
· out_act — активация выходного слоя: "sigmoid", "softmax", "linear" (по умолчанию "sigmoid")

Возвращает DictValue с полями "layers", "act", "out_act".

```
model = nn.mlp([2, 8, 8, 1], "leaky_relu", "sigmoid")       // бинарная классификация
model = nn.mlp([2, 16, 16, 3], "leaky_relu", "softmax")     // многоклассовая
model = nn.mlp([2, 16, 1], "leaky_relu", "linear")          // регрессия
```

Перед исполнением U-Ray статически проверяет размерности, которые можно
вывести из литералов и известных переменных. Например, если `nn.dense(2, 4)`
передаёт вектор длины 4 в `nn.dense(5, 1)`, файл завершится ошибкой проверки
до выполнения первой инструкции. Для `nn.mlp` также проверяются известные
ширины `X` и `Y` в `nn.train` и `nn.evaluate`.

```uray
import nn

first = nn.dense(2, 4)
second = nn.dense(5, 1)
hidden = nn.forward(first, [0.0, 1.0])
output = nn.forward(second, hidden) // ошибка: слой ожидает 5, получено 4
```

Для отдельной проверки без запуска программы выполни `./uray check file.ur`.
Размерности, зависящие от пользовательского ввода или динамических вычислений,
остаются под runtime-проверками.

nn.forward_mlp(model, x)

Прямой проход всей сети. Возвращает cache:

```
cache = [x, z1, h1, z2, h2, ..., zN, out]
```

где x — вход, z_k — выход слоя k до активации, h_k — после, out — финальный выход.

nn.backward_mlp(model, cache, grad_out, skip_last_act)

Обратный проход. Возвращает grads — список [grad_W, grad_b] по слоям, в порядке от первого к последнему.

skip_last_act (по умолчанию false) — контракт (см. ниже).

nn.accuracy(yhat, y)

Доля правильных предсказаний (порог 0.5) для бинарной классификации.

nn.summary(model)

Печатает архитектуру: act, out_act, размеры и число параметров каждого слоя.

nn.rand_seed(seed)

Фиксирует random seed. Влияет на math.rand и инициализацию весов в nn.mlp/nn.dense.

nn.train(model, X, Y, epochs, lr[, batch_size, loss, optimizer, show_progress])

Обучает модель мини-батчами и возвращает средний loss по каждой эпохе.
Оптимизаторы: `"sgd"`, `"momentum"` и `"adam"`; функции потерь: `"mse"`,
`"bce"` и `"cce"`.

Последний аргумент `show_progress` необязателен и по умолчанию равен `false`.
Если передать `true`, U-Ray выводит прогресс-бар и средний loss после каждой
эпохи:

```
history = nn.train(model, X, Y, 100, 0.01, 32, "bce", "adam", true)
```

`nn.evaluate(model, X, Y, loss)` возвращает `[средний_loss, accuracy]`.

---

nn: autodiff и составляемые слои

Общий reverse-mode autodiff позволяет строить граф из тензорных операций;
градиенты считаются по графу, а не отдельной реализацией `backward` для каждой
модели.

```uray
weight = nn.parameter([[0.2, -0.4], [0.7, 0.3]])
x = nn.tensor([1.0, 0.0])
logits = nn.matmul(weight, x)
loss = nn.cross_entropy(logits, [1.0, 0.0])
nn.backward(loss)
grad_weight = nn.grad(weight)
nn.sgd_update([weight], 0.01)
nn.zero_grad([weight])
```

`nn.tensor(data[, requires_grad])` создаёт узел графа; `nn.parameter(data)`
создаёт обучаемый параметр. Основные операции: `nn.add`, `nn.sub`,
`nn.multiply`, `nn.divide`, `nn.matmul`, `nn.reshape`, `nn.reduce_sum`,
`nn.mean`, `nn.relu`, `nn.sigmoid`, `nn.tanh`, `nn.softmax`, `nn.conv2d`,
`nn.max_pool2d`, `nn.mse` и `nn.cross_entropy`. `nn.cross_entropy` принимает
logits и one-hot labels; `nn.cce` по-прежнему принимает вероятности, как и
прежний high-level API. Для `GradTensor` также работают `+`, `-`, `*`, `/`,
`**` с числовой степенью и унарный минус; матричное умножение вызывается как
`nn.matmul(a, b)`. `nn.backward(loss)` освобождает граф после прохода, чтобы
держать расход памяти под контролем; для нового шага нужно выполнить forward
заново.

Слои можно собирать в последовательную сеть:

```uray
model = nn.sequential([
    nn.conv2d_layer(1, 8, 3, 1, 1),
    nn.activation_layer("relu"),
    nn.max_pool2d_layer(2, 2),
    nn.flatten_layer(),
    nn.linear_layer(8 * 14 * 14, 10)
])
```

Доступны `linear_layer`, `conv2d_layer`, `activation_layer`,
`max_pool2d_layer`, `flatten_layer`, `batch_norm_layer`, `dropout_layer` и
`residual_layer`. `nn.forward_layers(model, x)` запускает режим оценки, а
`nn.forward_layers(model, x, true)` — обучающий режим. `nn.train_layers`
автоматически включает обучение; `evaluate_layers` и `predict_layers` используют
режим оценки.

`nn.batch_norm_layer(features[, momentum, epsilon, axis])` обучает параметры
масштаба и сдвига и хранит скользящие статистики. По умолчанию нормализуется
последняя ось признаков, например `[batch, features]`; для выхода свёртки
`[batch, channels, height, width]` укажи `axis = 1`. При Dropout с аргументом
`rate` случайные элементы обнуляются только во время обучения, а оставшиеся
масштабируются на `1 / (1 - rate)`.

`nn.residual_layer([...])` принимает непустую последовательность слоёв и
прибавляет её результат к входу. Вход и ветвь должны иметь одинаковую форму.
Вложенные слои входят в `nn.parameters`, обучение и сохранение модели.

Для последовательностей доступны `nn.self_attention_layer(embed_dim, heads,
causal)` и `nn.transformer_block(embed_dim, heads, hidden_dim, dropout, causal)`.
Self-attention работает с формами `[sequence, features]` и
`[batch, sequence, features]`; число признаков должно делиться на число голов.
Causal-маска запрещает позиции смотреть на будущие токены. Transformer-блок
содержит pre-LayerNorm, multi-head attention, feed-forward сеть и две residual
связи. Входом служат числовые embeddings; токенизатор и таблицу embeddings нужно
строить отдельно.

```uray
model = nn.sequential([
    nn.batch_norm_layer(32),
    nn.dropout_layer(0.2),
    nn.residual_layer([
        nn.linear_layer(32, 64),
        nn.activation_layer("relu"),
        nn.linear_layer(64, 32)
    ]),
    nn.linear_layer(32, 10)
])
```

```uray
sequence_model = nn.sequential([
    nn.transformer_block(128, 4, 512, 0.1, true),
    nn.linear_layer(128, 32)
])
```

U-Ray-функции также могут комбинировать операции, ветвить и повторно
использовать результаты; для новой дифференцируемой примитивы её прямой и
обратный проход добавляется в ядро.

```uray
history = nn.train_layers(
    model, X, Y, 5, 0.001, 32, "cce", "adam", true
)
metrics = nn.evaluate_layers(model, X, Y, "cce")
probabilities = nn.predict_layers(model, X, 10)
nn.save_layers(model, "model.json")
loaded = nn.load_layers("model.json")
```

`nn.train_layers` поддерживает loss `"cce"`, `"bce"` (логиты) и `"mse"`,
оптимизаторы `"adam"` и `"sgd"`. `show_progress` по умолчанию включён. Для
изображений укажи `input_shape`, например `[1, 28, 28]`; шкала обновляется по
мини-батчам. Встроенный `native` backend выполняет матричные умножения, Conv2D,
ReLU и max-pooling в прямом и обратном проходах через C-ядра; CNN можно обучать без
NumPy. Остальные операции autodiff остаются в обычном runtime U-Ray. На Python
backend удобно проверять небольшие датасеты и примеры.

Сохранение модели включает параметры BatchNorm, running statistics, Dropout и
вложенные residual-ветви; при загрузке восстановленные слои готовы к оценке
или дальнейшему обучению.

Пример всех трёх слоёв в обучаемом классификаторе:

```
./uray examples/residual_regularization.ur
```

Пример обучения Transformer-блока на последовательностях:

```
./uray examples/transformer_sequence.ur
```

Полный пример сверточной сети на MNIST:

```
python3 examples/download_mnist.py
./uray examples/mnist_cnn.ur --backend=native
```

---

nn: загрузка наборов данных

`nn.load_mnist(directory)` читает стандартные IDX-файлы MNIST — сжатые `.gz`
или уже распакованные. Второй аргумент включает скачивание недостающих
архивов с официально используемого зеркала:

```
data = nn.load_mnist("data/mnist", true)
```

Можно вызвать `nn.download_mnist(directory)`, чтобы скачать и проверить MNIST
отдельно от обучения. Загрузка атомарная: незавершённый файл не подменит архив,
а скачанный набор проверяется по заголовкам IDX и ожидаемым размерам выборок.
При стандартном вызове без `true` загрузчик работает офлайн и только читает
файлы из каталога.

`nn.load_csv(path, label_column[, test_ratio[, seed[, normalize[, one_hot[, header]]]]])`
загружает таблицу с числовыми признаками и создаёт воспроизводимые выборки
train/test. `label_column` — имя столбца при заголовке или индекс столбца
(индексы могут быть отрицательными). По умолчанию заголовок включён,
`test_ratio=0.2`, `seed=42`, нормализация и one-hot выключены. При
`normalize=true` признаки масштабируются по минимуму и максимуму train-части;
при `one_hot=true` строковые или числовые метки классов кодируются векторы
one-hot. Без one-hot метки должны быть числовыми и возвращаются как скалярные
векторы, что подходит для регрессии.

Результат содержит `train` и `test` с полями `"X"`, `"Y"` и `"count"`, а также
`"features"`, `"classes"` и, если есть заголовок, `"feature_names"`.
Обычные списки можно сразу передавать в `nn.train`, `nn.evaluate` и
`nn.predict`.

```
import nn

data = nn.load_csv(
    "examples/tabular_classification.csv",
    "class",
    0.2,
    42,
    true,
    true,
    true
)
model = nn.mlp([data["features"], 16, len(data["classes"])], "relu", "softmax")
history = nn.train(
    model, data["train"]["X"], data["train"]["Y"],
    20, 0.001, 16, "cce", "adam"
)
metrics = nn.evaluate(model, data["test"]["X"], data["test"]["Y"], "cce")
```

Для MNIST каталог содержит четыре стандартных файла:

```
train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz
```

Результат `nn.load_mnist` — словарь `{"train": split, "test": split}`. В каждом `split` доступны
`"X"` (изображения), `"Y"` (метки one-hot), `"count"`, `"rows"` и `"columns"`.
Изображения хранятся компактно и нормализуются в диапазон `[0, 1]` по мере
обработки мини-батчей, а не копируются целиком в Python-списки.

```
import nn

data = nn.load_mnist("data/mnist")
train = data["train"]
test = data["test"]

model = nn.mlp([784, 64, 10], "relu", "softmax")
history = nn.train(
    model, train["X"], train["Y"],
    5, 0.001, 128, "cce", "adam", true
)
metrics = nn.evaluate(model, test["X"], test["Y"], "cce")
image = nn.sample(test["X"], 0)
scores = nn.predict(model, [image])
```

`nn.sample(view, index)` возвращает одно нормализованное изображение или
вектор-метку. `nn.predict(model, X[, limit])` возвращает вероятности классов;
необязательный `limit` ограничивает число предсказаний.

Для обычной MLP можно обучать полный MNIST встроенным нативным CPU backend без
NumPy. При первом запуске нужен системный C-компилятор (`cc`, `clang` или
`gcc`); исходник ядра входит в U-Ray и кэшируется после сборки:

```
python3 examples/download_mnist.py
./uray examples/mnist.ur --backend=native
```

Этот backend ускоряет `nn.mlp`, `nn.train`, `nn.evaluate` и `nn.predict`, а
также `matmul`, Conv2D, ReLU и max-pooling в обучаемых `nn.sequential`-моделях.
AVX-512 и AVX2 выбираются автоматически, иначе используется переносимый
скалярный код. Пример CNN на полном MNIST также работает без NumPy:

```
./uray examples/mnist_cnn.ur --backend=native
```

Для сравнения полного обучения и оценки MLP на всех 60 000 train и 10 000 test
изображениях запусти:

```
python3 benchmarks/benchmark_mnist.py
```

Для контрольной стороны этого замера нужен NumPy; сам запуск U-Ray с
`--backend=native` его не использует. Скрипт сравнивает одинаковую модель и
один цикл train+evaluate, а также показывает отношение времени нативного
backend к NumPy на текущем компьютере.

Для замера Conv2D forward+backward на изображениях 28×28 и восьми фильтрах
3×3 используй:

```
python3 benchmarks/benchmark_conv2d.py
```

Этот сравнительный скрипт тоже использует NumPy только как контрольный backend.

Чтобы сравнить полный цикл train+evaluate для CNN с одинаковыми весами и
параметрами, используй:

```
python3 benchmarks/benchmark_cnn.py
```

По умолчанию скрипт берёт первые 1 024 train и test изображения для быстрой
проверки. Флаг `--full` запускает сравнение на всех 60 000 train и 10 000 test
изображениях. Он отдельно показывает время обучения и оценки, итоговое
отношение native/NumPy и точность обеих моделей.

---

nn: оптимизаторы

SGD

```
nn.sgd_step(model, grads, lr)
```

W -= lr * grad_W, b -= lr * grad_b. Обновляет модель на месте.

Momentum SGD

```
state = nn.momentum_init(model)
nn.momentum_step(model, grads, state, lr, momentum)
```

v = momentum * v - lr * grad, W += v. state — отдельный объект, должен передаваться при каждом шаге.

Adam

```
state = nn.adam_init(model)
nn.adam_step(model, grads, state, lr)
nn.adam_step(model, grads, state, lr, beta1, beta2, eps)  // с параметрами
```

С bias correction. По умолчанию: beta1=0.9, beta2=0.999, eps=1e-8. state — отдельный объект.

---

nn: сохранение моделей

nn.save(model, path)

Сохраняет модель в JSON. Возвращает абсолютный путь.

· Относительный путь ("model.json") → рядом с .ur-файлом
· Абсолютный ("/tmp/model.json") → как есть

```
saved = nn.save(model, "xor_model.json")
print(saved)  // /полный/путь/xor_model.json
```

nn.load(path)

Загружает модель из JSON. Пути — по тем же правилам.

```
loaded = nn.load("xor_model.json")
```

`nn.export_pytorch(model, path)` экспортирует модель в PyTorch checkpoint `.pt`.
Требуется необязательная зависимость PyTorch:

```
nn.export_pytorch(model, "model.pt")
```

Checkpoint содержит размеры слоёв, активации и `state_dict` с ключами
`layers.N.weight` и `layers.N.bias` в стандартной для `torch.nn.Linear` форме.
Чтобы загрузить его, создайте PyTorch-модуль с `torch.nn.ModuleList` слоёв
`layers`, затем вызовите `load_state_dict(checkpoint["state_dict"])`.
U-Ray не загружает `.pt` обратно; состояние оптимизатора также не экспортируется.

---

Контракт skip_last_act

nn.backward_mlp(model, cache, grad_out, skip_last_act) — четвёртый аргумент (по умолчанию false).

· skip_last_act=true — grad_out это dL/dz_N (градиент по входу последней активации). Активация последнего слоя не применяется повторно.
  · С bce_grad (BCE+sigmoid), cce_grad (CCE+softmax), mse_grad + out_act="linear".
· skip_last_act=false — grad_out это dL/dyhat. Активация применяется (цепное правило).
  · С mse_grad + out_act="sigmoid".

Правило: bce_grad/cce_grad — всегда true. mse_grad + sigmoid — false. mse_grad + linear — true.

---

Состояние оптимизатора и save/load

nn.save(model, path) сохраняет только модель (layers, act, out_act, version). Состояние оптимизатора (state от nn.adam_init/nn.momentum_init) не сохраняется.

При загрузке модели и продолжении обучения:

· state создаётся заново (m=0, v=0, t=0 для Adam).
· Adam «начинает с нуля» — корректно для нового обучения, но не для продолжения.

---

Численная устойчивость

· nn.softmax — вычитает максимум (log-sum-exp), не переполняется.
· nn.sigmoid — стабильная реализация, не переполняется.
· nn.bce/nn.cce — клиппинг eps = 1e-12, не -inf.
· nn.adam_step — eps = 1e-8, не NaN при grad = 0.

---

Формат JSON

```json
{
  "version": 1,
  "act": "leaky_relu",
  "out_act": "sigmoid",
  "layers": [
    {"W": [[...], ...], "b": [...]},
    ...
  ]
}
```

load принимает файлы без version (обратная совместимость) и отклоняет version > 1 с понятной ошибкой.

---

Рецепты

Линейный классификатор (перцептрон)

```
import nn

X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y = [0.0, 1.0, 1.0, 1.0]

layer = nn.dense(2, 1)
lr = 0.5

for epoch in range(200) {
    for i in range(len(X)) {
        z = nn.forward(layer, X[i])
        yhat = nn.sigmoid(z[0])
        err = yhat - Y[i]
        for j in range(2) {
            layer["W"][0][j] = layer["W"][0][j] - lr * err * X[i][j]
        }
        layer["b"][0] = layer["b"][0] - lr * err
    }
}
```

Многослойная сеть (XOR)

```
import nn

X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y = [0.0, 1.0, 1.0, 0.0]

model = nn.mlp([2, 4, 1], "sigmoid")
lr = 0.5

for epoch in range(2000) {
    for i in range(len(X)) {
        cache = nn.forward_mlp(model, X[i])
        yhat = cache[len(cache) - 1]
        grads = nn.backward_mlp(model, cache, nn.mse_grad(yhat, [Y[i]]))
        nn.sgd_step(model, grads, lr)
    }
}
```

Спирали (нелинейная классификация, Adam)

```
import math
import nn

nn.rand_seed(42)

X = []
Y = []
for i in range(100) {
    t = 3.0 * math.pi * i / 100
    r = 0.1 + 0.9 * i / 100
    push(X, [r * math.cos(t), r * math.sin(t)])
    push(Y, 0.0)
    push(X, [r * math.cos(t + math.pi), r * math.sin(t + math.pi)])
    push(Y, 1.0)
}

model = nn.mlp([2, 8, 8, 1], "leaky_relu")
state = nn.adam_init(model)
for epoch in range(2000) {
    for i in range(len(X)) {
        cache = nn.forward_mlp(model, X[i])
        yhat = cache[len(cache) - 1]
        grads = nn.backward_mlp(model, cache, nn.bce_grad(yhat, [Y[i]]), true)
        nn.adam_step(model, grads, state, 0.01)
    }
}
```

Многоклассовая классификация (softmax + Adam)

```
import nn

model = nn.mlp([2, 16, 16, 3], "leaky_relu", "softmax")
state = nn.adam_init(model)

for epoch in range(2000) {
    for i in range(len(X)) {
        cache = nn.forward_mlp(model, X[i])
        yhat = cache[len(cache) - 1]
        grads = nn.backward_mlp(model, cache, nn.cce_grad(yhat, Y[i]), true)
        nn.adam_step(model, grads, state, 0.01)
    }
}
```

Сохранение и загрузка

```
import nn

model = nn.mlp([2, 4, 1], "sigmoid")
// ... обучение ...

path = nn.save(model, "model.json")
print("сохранено в", path)

loaded = nn.load("model.json")
cache = nn.forward_mlp(loaded, [0.0, 1.0])
print(cache[len(cache) - 1])
```

Регрессия (linear + MSE)

```
import nn

X = [0.0, 1.0, 2.0, 3.0]
Y = [1.0, 3.0, 5.0, 7.0]   // y = 2x + 1

model = nn.mlp([1, 16, 1], "leaky_relu", "linear")
state = nn.adam_init(model)

for epoch in range(2000) {
    for i in range(len(X)) {
        cache = nn.forward_mlp(model, [X[i]])
        yhat = cache[len(cache) - 1]
        grads = nn.backward_mlp(model, cache, nn.mse_grad(yhat, [Y[i]]), true)
        nn.adam_step(model, grads, state, 0.01)
    }
}
```
