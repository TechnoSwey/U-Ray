# Тесты U-Ray

Все тесты — это `.ur`-файлы с ожидаемым выводом в `.expected`.

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
4. Запусти ./run_tests.sh. Если вывод совпадает — PASS, иначе — FAIL.

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
numerical.ur Численная устойчивость (NaN/overflow)
roundtrip.ur save → load → ==
shape.ur nn.shape
strings_builtin.ur Строковые методы
train_batch.ur nn.train с батчами
tuples.ur Кортежи, (42,)
version.ur Версия JSON
zip.ur zip

Диагностические файлы (SKIP)

Файлы без .expected (например, ml_gradcheck.ur, bench.ur) — это инструменты отладки, а не тесты. Раннер их честно пропускает (SKIP).
