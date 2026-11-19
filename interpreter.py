# interpreter.py

import os
import time

from errors import RuntimeErrorUray, URayError
from values import (
    DictValue, Builtin, Module, Function, LambdaFunc, StructInstance,
    check_key, deep_equal, format_value
)
from tensor import Tensor
from autograd import GradTensor, as_grad_tensor
from modules import ModuleRegistry
from ast_nodes import (
    Program, Import, Assign, Del, Print, Number, Str, Var, BinOp,
    If, While, For, Break, Continue, FuncDecl, MethodDecl, Lambda, Return, Block, Bool, Call,
    ListLit, DictLit, Index, Attr, UnaryOp, TupleLit,
    Try, Throw, Match, TypeDecl, Slice, Ternary,
    StructPattern, BindPattern, WildcardPattern, ExprStmt
)
import values


class _ReturnSignal(Exception):
    def __init__(self, value):
        self.value = value


class _BreakSignal(Exception):
    pass


class _ContinueSignal(Exception):
    pass


class _ThrowSignal(Exception):
    def __init__(self, value):
        self.value = value


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
        self.profiler = None

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
        
        self.globals.set("upper", Builtin("upper", self._str_upper))
        self.globals.set("lower", Builtin("lower", self._str_lower))
        self.globals.set("split", Builtin("split", self._str_split))
        self.globals.set("strip", Builtin("strip", self._str_strip))
        self.globals.set("replace", Builtin("replace", self._str_replace))
        self.globals.set("join", Builtin("join", self._str_join))
        self.globals.set("find", Builtin("find", self._str_find))
        self.globals.set("slice", Builtin("slice", self._str_slice))
        
        self.globals.set("zeros", Builtin("zeros", self._tensor_zeros))
        self.globals.set("ones", Builtin("ones", self._tensor_ones))
        self.globals.set("shape", Builtin("shape", self._tensor_shape))
        
        self.globals.set("read_file", Builtin("read_file", self._stl_read_file))
        self.globals.set("write_file", Builtin("write_file", self._stl_write_file))
        self.globals.set("append_file", Builtin("append_file", self._stl_append_file))

        self.globals.set("sort", Builtin("sort", self._stl_sort))
        self.globals.set("reverse", Builtin("reverse", self._stl_reverse))
        self.globals.set("map", Builtin("map", self._stl_map))
        self.globals.set("filter", Builtin("filter", self._stl_filter))
        self.globals.set("reduce", Builtin("reduce", self._stl_reduce))

        self.globals.set("progress", Builtin("progress", self._builtin_progress))

    def _builtin_progress(self, args):
        if len(args) not in (2, 3):
            raise RuntimeErrorUray("progress ожидает 2 или 3 аргумента: current, total[, loss]")
        cur = args[0]
        total = args[1]
        has_loss = len(args) == 3
        loss = args[2] if has_loss else None
        if not isinstance(cur, int) or isinstance(cur, bool):
            raise RuntimeErrorUray("progress: current должен быть целым")
        if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
            raise RuntimeErrorUray("progress: total должен быть положительным целым")
        if has_loss and (
            not isinstance(loss, (int, float)) or isinstance(loss, bool)
        ):
            raise RuntimeErrorUray("progress: loss должен быть числом")
        if cur < 0:
            cur = 0
        if cur > total:
            cur = total
        width = 20
        filled = int(width * cur / total)
        bar = "█" * filled + "░" * (width - filled)
        pct = int(100 * cur / total)
        loss_text = f" loss: {loss:.6g}" if has_loss else ""
        print(f"\r[{bar}] {pct}%{loss_text}", end="", flush=True)
        if cur == total:
            print()
        return 0

    # ---------- Тензоры ----------
    def _tensor_zeros(self, args):
        if len(args) != 1 or not isinstance(args[0], list):
            raise RuntimeErrorUray("zeros ожидает список размерностей")
        return Tensor.zeros(args[0])

    def _tensor_ones(self, args):
        if len(args) != 1 or not isinstance(args[0], list):
            raise RuntimeErrorUray("ones ожидает список размерностей")
        return Tensor.ones(args[0])

    def _tensor_shape(self, args):
        if len(args) != 1 or not isinstance(args[0], Tensor):
            raise RuntimeErrorUray("shape ожидает тензор")
        return args[0].shape()

    # ---------- STL: Файлы ----------
    def _resolve_path(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(values.URAY_BASE_DIR, path)

    def _stl_read_file(self, args):
        if len(args) != 1 or not isinstance(args[0], str):
            raise RuntimeErrorUray("read_file ожидает путь")
        full = self._resolve_path(args[0])
        try:
            with open(full, "r", encoding="utf-8") as f:
                return f.read()
        except OSError as e:
            raise RuntimeErrorUray(f"Не удалось прочитать {args[0]!r}: {e}")

    def _stl_write_file(self, args):
        if len(args) != 2 or not isinstance(args[0], str) or not isinstance(args[1], str):
            raise RuntimeErrorUray("write_file ожидает путь и текст")
        full = self._resolve_path(args[0])
        try:
            with open(full, "w", encoding="utf-8") as f:
                f.write(args[1])
            return 0
        except OSError as e:
            raise RuntimeErrorUray(f"Не удалось записать {args[0]!r}: {e}")

    def _stl_append_file(self, args):
        if len(args) != 2 or not isinstance(args[0], str) or not isinstance(args[1], str):
            raise RuntimeErrorUray("append_file ожидает путь и текст")
        full = self._resolve_path(args[0])
        try:
            with open(full, "a", encoding="utf-8") as f:
                f.write(args[1])
            return 0
        except OSError as e:
            raise RuntimeErrorUray(f"Не удалось дописать в {args[0]!r}: {e}")

    # ---------- STL: Коллекции ----------
    def _stl_sort(self, args):
        if len(args) != 1 or not isinstance(args[0], list):
            raise RuntimeErrorUray("sort ожидает список")
        try:
            args[0].sort()
        except TypeError:
            raise RuntimeErrorUray("sort: элементы списка должны быть сравнимы")
        return args[0]

    def _stl_reverse(self, args):
        if len(args) != 1 or not isinstance(args[0], list):
            raise RuntimeErrorUray("reverse ожидает список")
        args[0].reverse()
        return args[0]

    def _apply_func(self, fn, arg):
        if isinstance(fn, Builtin):
            return fn.fn([arg])
        if isinstance(fn, Function):
            local = Env(parent=self.globals)
            if len(fn.decl.params) != 1:
                raise RuntimeErrorUray(f"Функция {fn.decl.name!r} должна принимать 1 аргумент")
            local.set(fn.decl.params[0], arg)
            saved_env = self.env
            self.env = local
            try:
                self.exec_block(fn.decl.body)
                return 0
            except _ReturnSignal as sig:
                return sig.value
            finally:
                self.env = saved_env
        if isinstance(fn, LambdaFunc):
            local = Env(parent=self.globals)
            if len(fn.node.params) != 1:
                raise RuntimeErrorUray("Лямбда должна принимать 1 аргумент")
            local.set(fn.node.params[0], arg)
            saved_env = self.env
            self.env = local
            try:
                self.exec_block(fn.node.body)
                return 0
            except _ReturnSignal as sig:
                return sig.value
            finally:
                self.env = saved_env
        raise RuntimeErrorUray("Ожидается функция")

    def _stl_map(self, args):
        if len(args) != 2 or not isinstance(args[0], list):
            raise RuntimeErrorUray("map ожидает список и функцию")
        return [self._apply_func(args[1], x) for x in args[0]]

    def _stl_filter(self, args):
        if len(args) != 2 or not isinstance(args[0], list):
            raise RuntimeErrorUray("filter ожидает список и функцию")
        return [x for x in args[0] if self._apply_func(args[1], x)]

    def _stl_reduce(self, args):
        if len(args) != 3 or not isinstance(args[0], list):
            raise RuntimeErrorUray("reduce ожидает список, функцию и начальное значение")
        acc = args[2]
        for x in args[0]:
            if isinstance(args[1], Builtin):
                acc = args[1].fn([acc, x])
            elif isinstance(args[1], (Function, LambdaFunc)):
                fn = args[1]
                node = fn.decl if isinstance(fn, Function) else fn.node
                if len(node.params) != 2:
                    raise RuntimeErrorUray("reduce: функция должна принимать 2 аргумента")
                local = Env(parent=self.globals)
                local.set(node.params[0], acc)
                local.set(node.params[1], x)
                saved_env = self.env
                self.env = local
                try:
                    self.exec_block(node.body)
                    acc = 0
                except _ReturnSignal as sig:
                    acc = sig.value
                finally:
                    self.env = saved_env
            else:
                raise RuntimeErrorUray("reduce: ожидается функция")
        return acc

    # ---------- Базовые ----------
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

    # ---------- Строковые ----------
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
            return self._check_str("split", args[0]).split()
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
        return self._check_str("find", args[0]).find(self._check_str("find", args[1]))

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
            raise RuntimeErrorUray(f"slice: диапазон {start}:{end} вне границ строки (длина {len(s)})")
        return s[start:end]

    # ---------- Выполнение ----------
    def run(self, program):
        for stmt in program.statements:
            self.exec_statement(stmt)

    def exec_statement(self, node):
        if isinstance(node, ExprStmt):
            return self.eval_expr(node.expr)

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
            if node.start is not None:
                self.exec_statement(node.start)
                while self.eval_expr(node.iterable):
                    try:
                        self.exec_block(node.body)
                    except _ContinueSignal:
                        pass
                    except _BreakSignal:
                        break
                    self.exec_statement(node.step)
                return
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
                        raise RuntimeErrorUray("for с распаковкой ожидает кортежи")
                    if len(node.var) != len(item):
                        raise RuntimeErrorUray(f"for с распаковкой: ожидалось {len(node.var)} элементов, получено {len(item)}")
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

        if isinstance(node, TypeDecl):
            self.env.set(node.name, node)
            return

        if isinstance(node, Return):
            value = 0 if node.value is None else self.eval_expr(node.value)
            raise _ReturnSignal(value)

        if isinstance(node, Try):
            try:
                self.exec_block(node.try_block)
            except (_ThrowSignal, RuntimeErrorUray) as e:
                err_value = e.value if isinstance(e, _ThrowSignal) else str(e)
                self.env.assign_existing(node.catch_var, err_value)
                self.exec_block(node.catch_block)
            return

        if isinstance(node, Throw):
            value = self.eval_expr(node.value)
            raise _ThrowSignal(value)

        if isinstance(node, Match):
            value = self.eval_expr(node.expr)
            matched = False
            for pattern, block in node.cases:
                bindings = {}
                if self._match_pattern(pattern, value, bindings):
                    saved_env = self.env
                    self.env = Env(parent=saved_env)
                    for name, bound_value in bindings.items():
                        self.env.set(name, bound_value)
                    try:
                        self.exec_block(block)
                    finally:
                        self.env = saved_env
                    matched = True
                    break
            if not matched and node.default_block is not None:
                self.exec_block(node.default_block)
            return

        self.eval_expr(node)
        return

    def _match_pattern(self, pattern, value, bindings):
        if isinstance(pattern, WildcardPattern):
            return True
        if isinstance(pattern, BindPattern):
            if pattern.name in bindings:
                return deep_equal(bindings[pattern.name], value)
            bindings[pattern.name] = value
            return True
        if isinstance(pattern, StructPattern):
            if not isinstance(value, StructInstance) \
                    or value.type_name != pattern.type_name:
                return False
            if pattern.named:
                for field_name, subpattern in pattern.fields:
                    if field_name not in value.fields:
                        return False
                    if not self._match_pattern(
                        subpattern, value.fields[field_name], bindings
                    ):
                        return False
                return True
            field_names = value.fields.keys()
            if len(pattern.fields) != len(field_names):
                return False
            for subpattern, field_name in zip(pattern.fields, field_names):
                if not self._match_pattern(
                    subpattern, value.fields[field_name], bindings
                ):
                    return False
            return True
        return deep_equal(value, self.eval_expr(pattern))

    def _import_file(self, rel_path):
        base = values.URAY_BASE_DIR
        full = os.path.normpath(os.path.join(base, rel_path))
        if full in self.loaded_files:
            raise RuntimeErrorUray(f"Циклический импорт: {rel_path!r} уже загружен")
        if not os.path.isfile(full):
            raise RuntimeErrorUray(f"Файл для импорта не найден: {rel_path!r}")
        try:
            with open(full, "r", encoding="utf-8") as f:
                src = f.read()
        except OSError as e:
            raise RuntimeErrorUray(f"Не удалось прочитать {rel_path!r}: {e}")

        from parser import parse
        program = parse(src)
        from typechecker import StaticChecker
        StaticChecker().check(program)
        sub = Interpreter(loaded_files=self.loaded_files | {full})
        sub.profiler = self.profiler
        sub.run(program)
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
            if isinstance(container, Tensor):
                container.set(key, value)
                return
            if isinstance(container, list):
                self.check_index(container, key)
                container[key] = value
                return
            if isinstance(container, tuple):
                raise RuntimeErrorUray("Кортеж неизменяемый")
            raise RuntimeErrorUray("Индексация возможна только для списков, тензоров и словарей")
        if isinstance(target, Attr):
            container = self.eval_expr(target.target)
            if isinstance(container, StructInstance):
                if target.name not in container.fields:
                    raise RuntimeErrorUray(f"В структуре {container.type_name!r} нет поля {target.name!r}")
                container.fields[target.name] = value
                return
            if isinstance(container, DictValue):
                container[target.name] = value
                return
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
            raise RuntimeErrorUray("Индексация возможна только для списков, кортежей и строк")
        if not isinstance(index, int) or isinstance(index, bool):
            raise RuntimeErrorUray("Индекс должен быть целым числом")
        if index < 0 or index >= len(container):
            raise RuntimeErrorUray(f"Индекс {index} вне диапазона (длина {len(container)})")

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
        if isinstance(node, Lambda):
            return LambdaFunc(node)
        if isinstance(node, Ternary):
            cond = self.eval_expr(node.cond)
            if cond:
                return self.eval_expr(node.true_val)
            else:
                return self.eval_expr(node.false_val)
        if isinstance(node, Index):
            container = self.eval_expr(node.target)
            if isinstance(node.index, Slice):
                start = self.eval_expr(node.index.start) if node.index.start is not None else None
                end = self.eval_expr(node.index.end) if node.index.end is not None else None
                if isinstance(container, list):
                    return container[start:end]
                if isinstance(container, Tensor):
                    return container.slice(start, end)
                raise RuntimeErrorUray("Срезы поддерживаются только для списков и тензоров")
            
            key = self.eval_expr(node.index)
            if isinstance(container, Tensor):
                return container.get(key)

            if isinstance(container, DictValue):
                check_key(key)
                if key not in container:
                    raise RuntimeErrorUray(f"Ключ {key!r} не найден")
                return container[key]
            if isinstance(container, (list, tuple, str)):
                self.check_index(container, key)
                return container[key]
            raise RuntimeErrorUray("Индексация возможна только для списков, кортежей, строк, словарей и тензоров")
        if isinstance(node, Attr):
            target = self.eval_expr(node.target)
            if isinstance(target, Module):
                if node.name not in target.members:
                    raise RuntimeErrorUray(f"В модуле {target.name!r} нет {node.name!r}")
                return target.members[node.name]
            if isinstance(target, StructInstance):
                if node.name in target.fields:
                    return target.fields[node.name]
                for m in target.methods:
                    if m.name == node.name:
                        return ("method", target, m)
                raise RuntimeErrorUray(f"В структуре {target.type_name!r} нет поля или метода {node.name!r}")
            if isinstance(target, DictValue):
                if node.name not in target:
                    raise RuntimeErrorUray(f"Ключ {node.name!r} не найден")
                return target[node.name]
            raise RuntimeErrorUray(f"Доступ к полю {node.name!r} возможен только у модуля, структуры или словаря")
        if isinstance(node, Call):
            return self.eval_call(node)
        if isinstance(node, UnaryOp):
            val = self.eval_expr(node.operand)
            if isinstance(val, GradTensor):
                return val.scalar_mul(-1.0) if node.op == "-" else val
            if isinstance(val, bool):
                raise RuntimeErrorUray(f"Унарный оператор {node.op!r} недопустим для bool")
            if not isinstance(val, (int, float)):
                raise RuntimeErrorUray(f"Унарный оператор {node.op!r} недопустим для {type(val).__name__}")
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
            raise RuntimeErrorUray(f"Операция {op!r} недопустима для строки и {type(right).__name__}")
        if isinstance(left, (list, DictValue, tuple)) or isinstance(right, (list, DictValue, tuple)):
            if op == "+" and isinstance(left, list) and isinstance(right, list):
                return left + right
            raise RuntimeErrorUray(f"Операция {op!r} недопустима для списков/словарей/кортежей")
        if isinstance(left, bool) or isinstance(right, bool):
            if op in ("<", ">", "<=", ">="):
                return self._cmp(op, int(left), int(right))
            raise RuntimeErrorUray(f"Операция {op!r} недопустима для булевых")
        if isinstance(left, GradTensor) or isinstance(right, GradTensor):
            if op == "+":
                return as_grad_tensor(left).add(right)
            if op == "-":
                return as_grad_tensor(left).sub(right)
            if op == "*":
                return as_grad_tensor(left).mul(right)
            if op == "/":
                return as_grad_tensor(left).div(right)
            if op == "**":
                if not isinstance(left, GradTensor) or not isinstance(
                    right, (int, float)
                ):
                    raise RuntimeErrorUray("Степень тензора должна быть числом")
                return left.pow(right)
            raise RuntimeErrorUray(
                f"Операция {op!r} недопустима для тензоров"
            )
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise RuntimeErrorUray(f"Операция {op!r} недопустима для {type(left).__name__} и {type(right).__name__}")
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
            call_name = self._profile_call_name(node.callee)
            if self.profiler is not None and call_name.startswith("nn."):
                started = time.perf_counter_ns()
                try:
                    result = callee.fn(args)
                except BaseException:
                    self.profiler.record(
                        call_name, time.perf_counter_ns() - started,
                        args, failed=True,
                    )
                    raise
                self.profiler.record(
                    call_name, time.perf_counter_ns() - started, args, result
                )
                return result
            return callee.fn(args)

        if isinstance(callee, LambdaFunc):
            if len(args) != len(callee.node.params):
                raise RuntimeErrorUray(f"Лямбда ожидает {len(callee.node.params)} аргументов, получено {len(args)}")
            local = Env(parent=self.globals)
            for name, val in zip(callee.node.params, args):
                local.set(name, val)
            saved_env = self.env
            self.env = local
            try:
                self.exec_block(callee.node.body)
                result = 0
            except _ReturnSignal as sig:
                result = sig.value
            finally:
                self.env = saved_env
            return result

        if isinstance(callee, tuple) and len(callee) == 3 and callee[0] == "method":
            _, instance, method = callee
            if len(args) != len(method.params):
                raise RuntimeErrorUray(f"Метод {method.name!r} ожидает {len(method.params)} аргументов, получено {len(args)}")
            local = Env(parent=self.globals)
            local.set("self", instance)
            for name, val in zip(method.params, args):
                local.set(name, val)
            saved_env = self.env
            self.env = local
            try:
                self.exec_block(method.body)
                result = 0
            except _ReturnSignal as sig:
                result = sig.value
            finally:
                self.env = saved_env
            return result

        if isinstance(callee, TypeDecl):
            if len(args) != len(callee.fields):
                raise RuntimeErrorUray(f"Структура {callee.name!r} ожидает {len(callee.fields)} аргументов, получено {len(args)}")
            fields = DictValue()
            for fname, val in zip(callee.fields, args):
                fields[fname] = val
            return StructInstance(callee.name, fields, callee.methods)

        if not isinstance(callee, Function):
            raise RuntimeErrorUray("Попытка вызвать не функцию")

        decl = callee.decl
        if len(args) != len(decl.params):
            raise RuntimeErrorUray(f"Функция {decl.name!r} ожидает {len(decl.params)} аргументов, получено {len(args)}")
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

    @staticmethod
    def _profile_call_name(node):
        if isinstance(node, Var):
            return node.name
        if isinstance(node, Attr):
            parent = Interpreter._profile_call_name(node.target)
            return f"{parent}.{node.name}" if parent else ""
        return ""


def run(program):
    try:
        Interpreter().run(program)
    except _ThrowSignal as e:
        raise RuntimeErrorUray(f"Непойманная ошибка: {e.value}")
