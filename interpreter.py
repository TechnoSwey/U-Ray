# interpreter.py

import os

from errors import RuntimeErrorUray, URayError
from values import (
    DictValue, Builtin, Module, Function,
    check_key, deep_equal, format_value
)
from modules import ModuleRegistry
from ast_nodes import (
    Program, Import, Assign, Del, Print, Number, Str, Var, BinOp,
    If, While, For, Break, Continue, FuncDecl, Return, Block, Bool, Call,
    ListLit, DictLit, Index, Attr, UnaryOp, TupleLit
)
import values


class _ReturnSignal(Exception):
    def __init__(self, value):
        self.value = value


class _BreakSignal(Exception):
    pass


class _ContinueSignal(Exception):
    pass


class Env:
    def __init__(self, parent=None):
        self.vars = {}
        self.parent = parent

    def get(self, name):
        env = self
        while env is not None:
            if name in env.vars:
                return env.vars[name]
            env = env.parent
        raise RuntimeErrorUray(f"Переменная {name!r} не определена")

    def set(self, name, value):
        self.vars[name] = value

    def assign_existing(self, name, value):
        env = self
        while env is not None:
            if name in env.vars:
                env.vars[name] = value
                return
            env = env.parent
        self.vars[name] = value


class Interpreter:
    def __init__(self, loaded_files=None):
        self.globals = Env()
        self.env = self.globals
        self.registry = ModuleRegistry()
        self.loaded_files = loaded_files if loaded_files is not None else set()

        self.globals.set("len", Builtin("len", self._builtin_len))
        self.globals.set("push", Builtin("push", self._builtin_push))
        self.globals.set("pop", Builtin("pop", self._builtin_pop))
        self.globals.set("str", Builtin("str", self._builtin_str))
        self.globals.set("range", Builtin("range", self._builtin_range))
        self.globals.set("keys", Builtin("keys", self._builtin_keys))
        self.globals.set("has", Builtin("has", self._builtin_has))
        self.globals.set("copy", Builtin("copy", self._builtin_copy))
        self.globals.set("enumerate", Builtin("enumerate", self._builtin_enumerate))
        self.globals.set("zip", Builtin("zip", self._builtin_zip))

        # строковые методы
        self.globals.set("upper", Builtin("upper", self._str_upper))
        self.globals.set("lower", Builtin("lower", self._str_lower))
        self.globals.set("split", Builtin("split", self._str_split))
        self.globals.set("strip", Builtin("strip", self._str_strip))
        self.globals.set("replace", Builtin("replace", self._str_replace))
        self.globals.set("join", Builtin("join", self._str_join))
        self.globals.set("find", Builtin("find", self._str_find))
        self.globals.set("slice", Builtin("slice", self._str_slice))

    # ---------- базовые ----------
    def _builtin_len(self, args):
        if len(args) != 1:
            raise RuntimeErrorUray("len ожидает 1 аргумент")
        v = args[0]
        if isinstance(v, (list, DictValue, tuple, str)):
            return len(v)
        raise RuntimeErrorUray("len ожидает список, кортеж, строку или словарь")

    def _builtin_push(self, args):
        if len(args) != 2:
            raise RuntimeErrorUray("push ожидает 2 аргумента")
        if not isinstance(args[0], list):
            raise RuntimeErrorUray("push ожидает список первым аргументом")
        args[0].append(args[1])
        return args[0]

    def _builtin_pop(self, args):
        if len(args) != 1:
            raise RuntimeErrorUray("pop ожидает 1 аргумент")
        if not isinstance(args[0], list):
            raise RuntimeErrorUray("pop ожидает список")
        if not args[0]:
            raise RuntimeErrorUray("pop из пустого списка")
        return args[0].pop()

    def _builtin_str(self, args):
        if len(args) != 1:
            raise RuntimeErrorUray("str ожидает 1 аргумент")
        return format_value(args[0])

    def _builtin_range(self, args):
        if len(args) == 0 or len(args) > 3:
            raise RuntimeErrorUray("range ожидает 1, 2 или 3 аргумента")
        for a in args:
            if not isinstance(a, int) or isinstance(a, bool):
                raise RuntimeErrorUray("range ожидает целые аргументы")
        if len(args) == 1:
            return list(range(args[0]))
        if len(args) == 2:
            return list(range(args[0], args[1]))
        step = args[2]
        if step == 0:
            raise RuntimeErrorUray("range: шаг не может быть нулём")
        return list(range(args[0], args[1], step))

    def _builtin_keys(self, args):
        if len(args) != 1:
            raise RuntimeErrorUray("keys ожидает 1 аргумент")
        d = args[0]
        if not isinstance(d, DictValue):
            raise RuntimeErrorUray("keys ожидает словарь")
        return d.keys()

    def _builtin_has(self, args):
        if len(args) != 2:
            raise RuntimeErrorUray("has ожидает 2 аргумента")
        d, k = args
        if not isinstance(d, DictValue):
            raise RuntimeErrorUray("has ожидает словарь первым аргументом")
        check_key(k)
        return k in d

    def _builtin_copy(self, args):
        if len(args) != 1:
            raise RuntimeErrorUray("copy ожидает 1 аргумент")
        v = args[0]
        if isinstance(v, list):
            return list(v)
        if isinstance(v, DictValue):
            nd = DictValue()
            for k, val in v.items():
                nd[k] = val
            return nd
        return v

    def _builtin_enumerate(self, args):
        if len(args) != 1:
            raise RuntimeErrorUray("enumerate ожидает 1 аргумент")
        if not isinstance(args[0], list):
            raise RuntimeErrorUray("enumerate ожидает список")
        return [(i, x) for i, x in enumerate(args[0])]

    def _builtin_zip(self, args):
        if len(args) < 2:
            raise RuntimeErrorUray("zip ожидает минимум 2 аргумента")
        lists = []
        for a in args:
            if not isinstance(a, list):
                raise RuntimeErrorUray("zip ожидает списки")
            lists.append(a)
        return [tuple(t) for t in zip(*lists)]

    # ---------- строковые ----------
    def _check_str(self, name, x):
        if not isinstance(x, str):
            raise RuntimeErrorUray(f"{name} ожидает строку")
        return x

    def _str_upper(self, args):
        if len(args) != 1:
            raise RuntimeErrorUray("upper ожидает 1 аргумент")
        return self._check_str("upper", args[0]).upper()

    def _str_lower(self, args):
        if len(args) != 1:
            raise RuntimeErrorUray("lower ожидает 1 аргумент")
        return self._check_str("lower", args[0]).lower()

    def _str_split(self, args):
        if len(args) == 1:
            s = self._check_str("split", args[0])
            return s.split()
        if len(args) == 2:
            s = self._check_str("split", args[0])
            sep = self._check_str("split", args[1])
            if sep == "":
                raise RuntimeErrorUray("split: разделитель не может быть пустым")
            return s.split(sep)
        raise RuntimeErrorUray("split ожидает 1 или 2 аргумента")

    def _str_strip(self, args):
        if len(args) != 1:
            raise RuntimeErrorUray("strip ожидает 1 аргумент")
        return self._check_str("strip", args[0]).strip()

    def _str_replace(self, args):
        if len(args) != 3:
            raise RuntimeErrorUray("replace ожидает 3 аргумента")
        s = self._check_str("replace", args[0])
        old = self._check_str("replace", args[1])
        new = self._check_str("replace", args[2])
        if old == "":
            raise RuntimeErrorUray("replace: old не может быть пустым")
        return s.replace(old, new)

    def _str_join(self, args):
        if len(args) != 2:
            raise RuntimeErrorUray("join ожидает 2 аргумента")
        lst = args[0]
        sep = self._check_str("join", args[1])
        if not isinstance(lst, list):
            raise RuntimeErrorUray("join ожидает список первым аргументом")
        for x in lst:
            if not isinstance(x, str):
                raise RuntimeErrorUray("join: все элементы списка должны быть строками")
        return sep.join(lst)

    def _str_find(self, args):
        if len(args) != 2:
            raise RuntimeErrorUray("find ожидает 2 аргумента")
        s = self._check_str("find", args[0])
        sub = self._check_str("find", args[1])
        return s.find(sub)

    def _str_slice(self, args):
        if len(args) != 3:
            raise RuntimeErrorUray("slice ожидает 3 аргумента")
        s = self._check_str("slice", args[0])
        start = args[1]
        end = args[2]
        if not isinstance(start, int) or isinstance(start, bool):
            raise RuntimeErrorUray("slice: start должен быть целым")
        if not isinstance(end, int) or isinstance(end, bool):
            raise RuntimeErrorUray("slice: end должен быть целым")
        if start < 0 or end > len(s) or start > end:
            raise RuntimeErrorUray(
                f"slice: диапазон {start}:{end} вне границ строки (длина {len(s)})"
            )
        return s[start:end]

    # ---------- выполнение ----------
    def run(self, program):
        for stmt in program.statements:
            self.exec_statement(stmt)

    def exec_statement(self, node):
        if isinstance(node, Import):
            if node.path is not None:
                self._import_file(node.path)
                return
            module = self.registry.get(node.name)
            if module is None:
                raise RuntimeErrorUray(f"Модуль {node.name!r} не найден")
            self.env.set(node.name, module)
            return

        if isinstance(node, Assign):
            value = self.eval_expr(node.value)
            self.assign_to(node.target, value)
            return

        if isinstance(node, Del):
            self.delete_at(node.target)
            return

        if isinstance(node, Print):
            parts = [format_value(self.eval_expr(v)) for v in node.values]
            print(" ".join(parts))
            return

        if isinstance(node, If):
            cond = self.eval_expr(node.cond)
            if cond:
                self.exec_block(node.then_block)
            elif node.else_block is not None:
                self.exec_block(node.else_block)
            return

        if isinstance(node, While):
            while self.eval_expr(node.cond):
                try:
                    self.exec_block(node.body)
                except _ContinueSignal:
                    continue
                except _BreakSignal:
                    break
            return

        if isinstance(node, For):
            iterable = self.eval_expr(node.iterable)
            if isinstance(iterable, DictValue):
                items = iterable.keys()
            elif isinstance(iterable, (list, str)):
                items = list(iterable)
            else:
                raise RuntimeErrorUray("for ожидает список, строку или словарь")
            for item in items:
                if isinstance(node.var, list):
                    if not isinstance(item, tuple):
                        raise RuntimeErrorUray(
                            "for с распаковкой ожидает кортежи "
                            f"(получено {type(item).__name__})"
                        )
                    if len(node.var) != len(item):
                        raise RuntimeErrorUray(
                            f"for с распаковкой: ожидалось {len(node.var)} "
                            f"элементов, получено {len(item)}"
                        )
                    for name, val in zip(node.var, item):
                        self.env.assign_existing(name, val)
                else:
                    self.env.assign_existing(node.var, item)
                try:
                    self.exec_block(node.body)
                except _ContinueSignal:
                    continue
                except _BreakSignal:
                    break
            return

        if isinstance(node, Break):
            raise _BreakSignal()

        if isinstance(node, Continue):
            raise _ContinueSignal()

        if isinstance(node, FuncDecl):
            self.env.set(node.name, Function(node))
            return

        if isinstance(node, Return):
            value = 0 if node.value is None else self.eval_expr(node.value)
            raise _ReturnSignal(value)

        self.eval_expr(node)
        return

    def _import_file(self, rel_path):
        base = values.URAY_BASE_DIR
        full = os.path.normpath(os.path.join(base, rel_path))
        if full in self.loaded_files:
            raise RuntimeErrorUray(
                f"Циклический импорт: {rel_path!r} уже загружен"
            )
        if not os.path.isfile(full):
            raise RuntimeErrorUray(f"Файл для импорта не найден: {rel_path!r}")
        try:
            with open(full, "r", encoding="utf-8") as f:
                src = f.read()
        except OSError as e:
            raise RuntimeErrorUray(f"Не удалось прочитать {rel_path!r}: {e}")

        from parser import parse
        program = parse(src)
        sub = Interpreter(loaded_files=self.loaded_files | {full})
        sub.run(program)
        # Экспортируем всё, кроме встроенных функций
        for name, val in sub.globals.vars.items():
            if not isinstance(val, Builtin):
                self.globals.set(name, val)

    def assign_to(self, target, value):
        if isinstance(target, Var):
            self.env.assign_existing(target.name, value)
            return
        if isinstance(target, Index):
            container = self.eval_expr(target.target)
            key = self.eval_expr(target.index)
            if isinstance(container, DictValue):
                check_key(key)
                container[key] = value
                return
            if isinstance(container, list):
                self.check_index(container, key)
                container[key] = value
                return
            if isinstance(container, tuple):
                raise RuntimeErrorUray("Кортеж неизменяемый")
            raise RuntimeErrorUray(
                "Индексация возможна только для списков и словарей"
            )
        if isinstance(target, Attr):
            raise RuntimeErrorUray("Присваивание в поле модуля запрещено")
        raise RuntimeErrorUray(f"Недопустимая цель присваивания: {target}")

    def delete_at(self, target):
        container = self.eval_expr(target.target)
        key = self.eval_expr(target.index)
        if isinstance(container, DictValue):
            check_key(key)
            if key not in container:
                raise RuntimeErrorUray(f"Ключ {key!r} не найден")
            del container[key]
            return
        if isinstance(container, list):
            self.check_index(container, key)
            del container[key]
            return
        if isinstance(container, tuple):
            raise RuntimeErrorUray("Кортеж неизменяемый")
        raise RuntimeErrorUray("del работает только со списками и словарями")

    def check_index(self, container, index):
        if not isinstance(container, (list, tuple, str)):
            raise RuntimeErrorUray(
                "Индексация возможна только для списков, кортежей и строк"
            )
        if not isinstance(index, int) or isinstance(index, bool):
            raise RuntimeErrorUray("Индекс должен быть целым числом")
        if index < 0 or index >= len(container):
            raise RuntimeErrorUray(
                f"Индекс {index} вне диапазона (длина {len(container)})"
            )

    def exec_block(self, block):
        for stmt in block.statements:
            self.exec_statement(stmt)

    def eval_expr(self, node):
        if isinstance(node, Number):
            return node.value

        if isinstance(node, Str):
            return node.value

        if isinstance(node, Bool):
            return node.value

        if isinstance(node, Var):
            return self.env.get(node.name)

        if isinstance(node, ListLit):
            return [self.eval_expr(item) for item in node.items]

        if isinstance(node, TupleLit):
            return tuple(self.eval_expr(item) for item in node.items)

        if isinstance(node, DictLit):
            d = DictValue()
            for key_node, value_node in node.pairs:
                key = self.eval_expr(key_node)
                check_key(key)
                if key in d:
                    raise RuntimeErrorUray(f"Дублирующийся ключ {key!r}")
                d[key] = self.eval_expr(value_node)
            return d

        if isinstance(node, Index):
            container = self.eval_expr(node.target)
            key = self.eval_expr(node.index)
            if isinstance(container, DictValue):
                check_key(key)
                if key not in container:
                    raise RuntimeErrorUray(f"Ключ {key!r} не найден")
                return container[key]
            if isinstance(container, (list, tuple, str)):
                self.check_index(container, key)
                return container[key]
            raise RuntimeErrorUray(
                "Индексация возможна только для списков, кортежей, строк и словарей"
            )

        if isinstance(node, Attr):
            target = self.eval_expr(node.target)
            if not isinstance(target, Module):
                raise RuntimeErrorUray(
                    f"Доступ к полю {node.name!r} возможен только у модуля"
                )
            if node.name not in target.members:
                raise RuntimeErrorUray(
                    f"В модуле {target.name!r} нет {node.name!r}"
                )
            return target.members[node.name]

        if isinstance(node, Call):
            return self.eval_call(node)

        if isinstance(node, UnaryOp):
            val = self.eval_expr(node.operand)
            if isinstance(val, bool):
                raise RuntimeErrorUray(
                    f"Унарный оператор {node.op!r} недопустим для bool"
                )
            if not isinstance(val, (int, float)):
                raise RuntimeErrorUray(
                    f"Унарный оператор {node.op!r} недопустим для "
                    f"{type(val).__name__}"
                )
            if node.op == "-":
                return -val
            return +val

        if isinstance(node, BinOp):
            left = self.eval_expr(node.left)
            right = self.eval_expr(node.right)
            return self.apply_binop(node.op, left, right)

        raise RuntimeErrorUray(f"Неизвестное выражение: {node}")

    def apply_binop(self, op, left, right):
        if op == "==":
            return deep_equal(left, right)
        if op == "!=":
            return not deep_equal(left, right)

        if isinstance(left, str) or isinstance(right, str):
            if op == "+" and isinstance(left, str) and isinstance(right, str):
                return left + right
            raise RuntimeErrorUray(
                f"Операция {op!r} недопустима для строки и {type(right).__name__}"
            )

        if isinstance(left, (list, DictValue, tuple)) or isinstance(right, (list, DictValue, tuple)):
            if op == "+" and isinstance(left, list) and isinstance(right, list):
                return left + right
            raise RuntimeErrorUray(
                f"Операция {op!r} недопустима для списков/словарей/кортежей"
            )

        if isinstance(left, bool) or isinstance(right, bool):
            if op in ("<", ">", "<=", ">="):
                return self._cmp(op, int(left), int(right))
            raise RuntimeErrorUray(f"Операция {op!r} недопустима для булевых")

        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise RuntimeErrorUray(
                f"Операция {op!r} недопустима для "
                f"{type(left).__name__} и {type(right).__name__}"
            )

        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            if right == 0:
                raise RuntimeErrorUray("Деление на ноль")
            return left / right
        if op == "%":
            if right == 0:
                raise RuntimeErrorUray("Остаток от деления на ноль")
            return left % right
        if op == "**":
            return left ** right
        if op in ("<", ">", "<=", ">="):
            return self._cmp(op, left, right)
        raise RuntimeErrorUray(f"Неизвестный оператор {op!r}")

    def _cmp(self, op, a, b):
        if op == "<":
            return a < b
        if op == ">":
            return a > b
        if op == "<=":
            return a <= b
        if op == ">=":
            return a >= b
        raise RuntimeErrorUray(f"Неизвестное сравнение {op!r}")

    def eval_call(self, node):
        callee = self.eval_expr(node.callee)
        args = [self.eval_expr(a) for a in node.args]

        if isinstance(callee, Builtin):
            return callee.fn(args)

        if not isinstance(callee, Function):
            raise RuntimeErrorUray("Попытка вызвать не функцию")

        decl = callee.decl
        if len(args) != len(decl.params):
            raise RuntimeErrorUray(
                f"Функция {decl.name!r} ожидает {len(decl.params)} аргументов, "
                f"получено {len(args)}"
            )
        local = Env(parent=self.globals)
        for name, val in zip(decl.params, args):
            local.set(name, val)

        saved_env = self.env
        self.env = local
        try:
            self.exec_block(decl.body)
            result = 0
        except _ReturnSignal as sig:
            result = sig.value
        finally:
            self.env = saved_env
        return result


def run(program):
    Interpreter().run(program)
