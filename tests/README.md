# Тесты U-Ray

Большинство тестов — `.ur`-файлы с ожидаемым выводом в `.expected`.
Дополнительно runner проверяет CLI, а Python unit-тесты — formatter, REPL
completion, статические размерности, VS Code manifest, PyTorch export,
пустые тензоры, MNIST IDX-загрузчик и autodiff (включая численные проверки
градиентов Conv2D). NumPy-тесты пропускаются, если зависимость не установлена.

## Как запускать

```bash
cd ~/uray
./run_tests.sh        # все тесты
./run_tests.sh --fast # только быстрые (без ml_*)
```

Как писать новый тест

1. Создай tests/my_test.ur.
2. Напиши код на U-Ray.
3. Создай tests/my_test.expected с точным выводом программы.
4. Запусти ./run_tests.sh. PASS засчитывается, только если совпали вывод и код завершения 0.

Структура

Файл Что проверяет
arith.ur Арифметика, приоритеты, степени
dicts.ur Словари, ключи, keys, has
edge.ur Циклы в структурах, copy, deep_equal
enumerate.ur enumerate и распаковка в for
funcs.ur Функции, рекурсия
hello.ur Базовый print
import_file.ur import из файлов
ml_xor.ur ML: XOR (accuracy = 1.0)
ml_spirals.ur ML: спирали (accuracy > 0.9)
ml_multiclass.ur ML: 3 класса, softmax
autograd_graph.ur Reverse-mode autodiff и обучение последовательной MLP
autograd_conv.ur Conv2D, max-pooling и общий backward
autograd_trainer.ur Универсальный trainer, evaluate и predict
autograd_cnn_trainer.ur Обучение сверточной модели на пакетах
test_autograd.py Градиенты, Conv2D, pooling, слои, optimizer и save/load
test_mnist.py IDX-загрузчик, выборки, train/evaluate и согласованность backend
evaluate_binary.ur Точность бинарной классификации
math_overflow.ur Обработка переполнения в math.exp
match_struct.ur Деструктуризация структур в match
nn_shapes.ur Проверка размерностей входа слоя
numerical.ur Численная устойчивость (NaN/overflow)
numpy_backend.ur Матрицы через backend NumPy
roundtrip.ur save → load → ==
shape.ur nn.shape
softmax_mse.ur Производная softmax для MSE
strings_builtin.ur Строковые методы
train_batch.ur nn.train с батчами
train_mse.ur Обучение с MSE и sigmoid
train_progress.ur Прогресс и loss по эпохам
train_loss_history.ur Средний loss по всем примерам
tensor_validation.ur Размерности и индексы тензоров
test_typechecker.py Предварительная проверка известных ML-размерностей
test_formatter.py Форматтер: комментарии, отступы, idempotence и CLI
test_repl_completion.py Подсказки REPL для имён и членов значений
test_vscode_extension.py Манифест и конфигурация локального VS Code расширения
tuples.ur Кортежи, (42,)
version.ur Версия JSON
zip.ur zip

Диагностические файлы (SKIP)

Файлы без .expected (например, ml_gradcheck.ur, bench.ur) — это инструменты отладки, а не тесты. Раннер их честно пропускает (SKIP).
