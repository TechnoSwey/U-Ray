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
8. nn: оптимизаторы
9. nn: сохранение моделей
10. Контракт skip_last_act
11. Состояние оптимизатора и save/load
12. Численная устойчивость
13. Формат JSON
14. Рецепты

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
