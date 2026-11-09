U-Ray

Минимальный интерпретируемый язык программирования с прицелом на машинное обучение.

U-Ray — это:

· полноценный скриптовый язык с функциями, списками, словарями, кортежами и циклами;
· встроенный ML-модуль nn с dense-слоями, backprop, SGD, Adam и обучением по батчам;
· ноль обязательных внешних зависимостей — только Python 3.10+ (NumPy и PyPy опциональны для ускорения);
· предсказуемая семантика, зафиксированная 19 автотестами.

Установка

```bash
git clone https://github.com/TechnoSwey/U-Ray.git
cd U-Ray
./install.sh
```

Запуск:

```bash
./uray examples/perceptron.ur
```

Или запусти пример прямо из репозитория (без установки):

```bash
python3 main.py examples/perceptron.ur
```

Требования: Python 3.10+. Никаких внешних зависимостей (NumPy и PyPy опциональны).

Быстрый старт

Hello, U-Ray

```
print("привет, U-Ray")
```

Функции и циклы

```
func factorial(n) {
    if n <= 1 {
        return 1
    }
    return n * factorial(n - 1)
}

for i in range(1, 6) {
    print(i, "! =", factorial(i))
}
```

Списки, кортежи и enumerate

```
numbers = [10, 20, 30]
for i, x in enumerate(numbers) {
    print("индекс", i, "значение", x)
}

pair = (1, 2)
print(pair[0], pair[1])
```

Строки

```
s = "Hello, World"
print(upper(s))             // HELLO, WORLD
print(split(s, ", "))       // ["Hello", "World"]
print(join(["a", "b"], "-")) // a-b
```

Модули

```
import math
import linalg

print(math.sqrt(2))
print(linalg.dot([1, 2, 3], [4, 5, 6]))
```

Нейросеть (полный цикл обучения)

```
import nn

// Данные: 2 входа, 1 выход
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y = [[0.0], [1.0], [1.0], [0.0]]  // метки — списки!

// Модель 2 -> 4 -> 1
model = nn.mlp([2, 4, 1], "sigmoid")

// Обучение "в одну строку" с батчами
history = nn.train(model, X, Y, epochs=2000, lr=0.5, batch_size=4, loss="mse", optimizer="sgd")

// Оценка
metrics = nn.evaluate(model, X, Y, "mse")
print("loss:", metrics[0])
print("accuracy:", metrics[1])

// Сохранение
nn.save(model, "xor_model.json")
```

Документация

· Язык — синтаксис, типы, операторы, семантика
· ML-модуль — nn, linalg, math, рецепты, nn.train
· Архитектура — как устроен интерпретатор, как расширять
· Тесты — как писать и запускать

Примеры

Файл Что показывает
examples/perceptron.ur Линейный классификатор (ручное обновление весов)
examples/mlp_xor.ur XOR через ручной backprop
examples/mlp_api.ur XOR через nn.mlp + nn.forward_mlp
examples/spirals.ur Нелинейная задача (спирали) с Adam
examples/multiclass.ur 3 класса, softmax, Adam
examples/train_batch_example.ur Обучение с батчами (nn.train)
examples/save_load.ur Сохранение и загрузка модели в JSON

Философия

1. Предсказуемость. Одинаковые вещи работают одинаково везде. == — всегда deep_equal. copy — всегда поверхностная. Слой — всегда словарь {"W": ..., "b": ...}.
2. Понятные ошибки. Не KeyError из Python, а Ключ 'b' не найден в 3:5.
3. Ноль обязательных зависимостей. Работает на голом Python. NumPy и PyPy — опциональны (см. ниже).
4. Расширяемость. Добавить nn-примитив — это 10 строк в modules.py, а не переписывание пол-интерпретатора.

Ускорение (NumPy и PyPy)

U-Ray работает без зависимостей, но использует ускорение, если оно есть:

· NumPy (pip install numpy или pkg install python-numpy): nn.forward, nn.train и linalg используют np.matmul (в 5–10 раз быстрее на больших матрицах).
· PyPy (pkg install pypy3): скрипт ./uray автоматически запускает через JIT-компилятор (в 5–15 раз быстрее Python-циклов).
· Если ни того, ни другого нет — работает чистый Python (медленнее, но без зависимостей).

Тесты и CI

https://github.com/TechnoSwey/uray/actions/workflows/ci.yml/badge.svg

Проект использует GitHub Actions для автоматического прогона тестов. Локально все тесты запускаются так:

```bash
./run_tests.sh
```

Быстрые тесты (без обучения моделей):

```bash
./run_tests.sh --fast
```

Лицензия

MIT.
