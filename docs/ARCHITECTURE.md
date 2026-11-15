Архитектура U-Ray

Как устроен интерпретатор: конвейер, слои, точки расширения.

Конвейер

```
.ur-файл
   │
   ▼
lexer.py       → список токенов
   │
   ▼
parser.py      → AST (дерево)
   │
   ▼
typechecker.py → проверка известных форм тензоров
   │
   ▼
interpreter.py → выполнение
```

Файлы

Файл Ответственность
errors.py Иерархия ошибок (URayError → LexerError, ParseError, RuntimeErrorUray)
values.py Типы значений (DictValue, Builtin, Module, Function), deep_equal, format_*, URAY_BASE_DIR
lexer.py Текст → токены. Позиции (строка:колонка) у каждого токена
ast_nodes.py Узлы AST: Program, Assign, Print, If, While, For, FuncDecl, Return, Call, BinOp, UnaryOp, ListLit, DictLit, Index, Attr, ...
parser.py Токены → AST. Рекурсивный спуск, приоритеты, проверка break/continue вне цикла
typechecker.py Консервативно проверяет известные размерности MLP и dense-проходов до исполнения
formatter.py Проверяет синтаксис и форматирует код, сохраняя комментарии
repl_completion.py Подсказки для имён, модулей, словарей и структур в REPL
modules.py Модули math, linalg, nn. Все built-in функции
interpreter.py Обход AST, среда (Env), вызовы функций, apply_binop
main.py Точка входа: читает файл, ставит URAY_BASE_DIR, ловит URayError
uray Bash-скрипт для запуска без python main.py

Слоистость

```
errors.py       (ни от чего не зависит)
   ↑
values.py       (импортирует errors)
   ↑
modules.py      (импортирует errors, values)
   ↑
interpreter.py  (импортирует errors, values, modules, ast_nodes)
   ↑
main.py         (импортирует всё)
```

lexer.py, parser.py, ast_nodes.py — независимые слои. lexer.py импортирует errors, parser.py — errors, ast_nodes, lexer.

Нет циклических импортов. Это позволяет добавлять модули и типы без переделок.

Как добавить новую функцию в nn

1. Открой modules.py.
2. В _build_nn добавь функцию:

```python
def my_func(args):
    if len(args) != 1:
        raise R("nn.my_func ожидает 1 аргумент")
    # ... логика ...
    return result
```

1. Зарегистрируй в Module("nn", {...}):

```python
"my_func": Builtin("my_func", my_func),
```

Всё. Функция доступна как nn.my_func(...).

Как добавить новый модуль

1. В modules.py добавь метод _build_my_module:

```python
def _build_my_module(self):
    def my_fn(args):
        ...
    return Module("my_module", {
        "my_fn": Builtin("my_fn", my_fn),
    })
```

1. Зарегистрируй в _build:

```python
return {
    "math": self._build_math(),
    "linalg": self._build_linalg(),
    "nn": self._build_nn(),
    "my_module": self._build_my_module(),
}
```

1. В U-Ray: import my_module.

Как добавить новый оператор

1. lexer.py — добавь токен:

```python
OP_MY = "OP_MY"
```

И распознавание в tokenize:

```python
if ch == "?":
    self.advance()
    tokens.append(Token(OP_MY, "?", line, col))
    continue
```

1. ast_nodes.py — если нужен новый узел, добавь класс.
2. parser.py — добавь в нужный уровень приоритета.
3. interpreter.py — обработай в eval_expr или apply_binop.

Как добавить новый тип значения

1. values.py — добавь класс (например, Tensor).
2. values.py — добавь в format_value / _format_repr_impl:

```python
if isinstance(v, Tensor):
    return f"tensor({v.data})"
```

1. values.py — добавь в deep_equal, если нужно.
2. interpreter.py — обработай в apply_binop и eval_expr.
3. modules.py — добавь функции, возвращающие Tensor.

Тесты

tests/*.ur + tests/*.expected. Раннер run_tests.sh прогоняет все, сравнивает вывод.

```bash
./run_tests.sh        # все тесты
./run_tests.sh --fast # только быстрые (без ml_*)
```

Расширение

· nn.Sequential — обёртка над model["layers"], чтобы не писать [2, 4, 1].
· Оптимизаторы — nn.momentum, nn.adam. Требуют состояния (velocity), это DictValue.
· Функции потерь — nn.cross_entropy, nn.binary_cross_entropy.
· Батчи — nn.train(model, X, Y, epochs, lr, batch_size).
· import из файлов — import "mylib.ur" вместо встроенных модулей.
