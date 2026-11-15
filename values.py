# values.py

import os

from errors import RuntimeErrorUray
from tensor import Tensor

URAY_BASE_DIR = os.getcwd()


def check_key(k):
    if isinstance(k, bool):
        raise RuntimeErrorUray("Ключ словаря не может быть bool")
    if not isinstance(k, (str, int)):
        raise RuntimeErrorUray(
            f"Ключ словаря должен быть строкой или целым, получено {type(k).__name__}"
        )


def _key_tag(k):
    if isinstance(k, str):
        return ("s", k)
    return ("i", k)


class DictValue:
    def __init__(self):
        self._data = {}
        self._keys = []

    def __contains__(self, k):
        check_key(k)
        return _key_tag(k) in self._data

    def __getitem__(self, k):
        check_key(k)
        return self._data[_key_tag(k)]

    def __setitem__(self, k, v):
        check_key(k)
        tag = _key_tag(k)
        if tag not in self._data:
            self._keys.append(k)
        self._data[tag] = v

    def __delitem__(self, k):
        check_key(k)
        tag = _key_tag(k)
        if tag not in self._data:
            raise RuntimeErrorUray(f"Ключ {k!r} не найден")
        del self._data[tag]
        self._keys.remove(k)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._keys)

    def keys(self):
        return list(self._keys)

    def items(self):
        for k in self._keys:
            yield k, self._data[_key_tag(k)]

    def __repr__(self):
        return f"DictValue({self._data})"


class Builtin:
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn


class Module:
    def __init__(self, name, members):
        self.name = name
        self.members = members

    def __repr__(self):
        return f"<module {self.name}>"


class Function:
    def __init__(self, decl):
        self.decl = decl


class LambdaFunc:
    def __init__(self, node):
        self.node = node


class StructInstance:
    def __init__(self, type_name, fields, methods):
        self.type_name = type_name
        self.fields = fields
        self.methods = methods

    def __repr__(self):
        return f"<struct {self.type_name} {self.fields}>"


def deep_equal(a, b, seen=None):
    if seen is None:
        seen = set()

    if isinstance(a, Tensor) or isinstance(b, Tensor):
        if not isinstance(a, Tensor) or not isinstance(b, Tensor):
            return False
        return a.data == b.data

    a_is_container = isinstance(a, (list, DictValue, tuple, StructInstance))
    b_is_container = isinstance(b, (list, DictValue, tuple, StructInstance))
    if not a_is_container and not b_is_container:
        if isinstance(a, bool) != isinstance(b, bool):
            return False
        return a == b
    if a_is_container != b_is_container:
        return False
    if type(a) != type(b):
        return False

    pair = (id(a), id(b))
    if pair in seen:
        return a is b
    seen.add(pair)
    try:
        if isinstance(a, StructInstance):
            if a.type_name != b.type_name:
                return False
            return deep_equal(a.fields, b.fields, seen)
        if isinstance(a, (tuple, list)):
            if len(a) != len(b):
                return False
            return all(deep_equal(x, y, seen) for x, y in zip(a, b))
        if len(a) != len(b):
            return False
        for k in a.keys():
            if k not in b:
                return False
            if not deep_equal(a[k], b[k], seen):
                return False
        return True
    finally:
        seen.discard(pair)


_MAX_REPR_DEPTH = 200


def _escape_string(s):
    return (
        s.replace("\\", "\\\\")
         .replace('"', '\\"')
         .replace("\n", "\\n")
         .replace("\t", "\\t")
    )


def _format_repr_impl(v, seen, depth):
    if depth > _MAX_REPR_DEPTH:
        return "<too deep>"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, float):
        if v.is_integer():
            return f"{v:.1f}"
        return repr(v)
    if isinstance(v, Tensor):
        return repr(v)
    if isinstance(v, StructInstance):
        return f"<struct {v.type_name} {_format_repr_impl(v.fields, seen, depth + 1)}>"
    if isinstance(v, (list, DictValue, tuple)):
        vid = id(v)
        if vid in seen:
            return "<cycle>"
        seen.add(vid)
        try:
            if isinstance(v, DictValue):
                parts = []
                for k, val in v.items():
                    parts.append(
                        f"{_format_repr_impl(k, seen, depth + 1)}: "
                        f"{_format_repr_impl(val, seen, depth + 1)}"
                    )
                return "{" + ", ".join(parts) + "}"
            if isinstance(v, tuple):
                if len(v) == 0:
                    return "()"
                if len(v) == 1:
                    return "(" + _format_repr_impl(v[0], seen, depth + 1) + ",)"
                return "(" + ", ".join(
                    _format_repr_impl(x, seen, depth + 1) for x in v
                ) + ")"
            return "[" + ", ".join(
                _format_repr_impl(x, seen, depth + 1) for x in v
            ) + "]"
        finally:
            seen.discard(vid)
    if isinstance(v, Builtin):
        return f"<builtin {v.name}>"
    if isinstance(v, Module):
        return f"<module {v.name}>"
    if isinstance(v, Function):
        return f"<function {v.decl.name}>"
    if isinstance(v, LambdaFunc):
        return "<lambda>"
    if isinstance(v, str):
        return '"' + _escape_string(v) + '"'
    return str(v)


def format_repr(v):
    return _format_repr_impl(v, set(), 0)


def _format_value_impl(v, seen, depth):
    if isinstance(v, str):
        return v
    return _format_repr_impl(v, seen, depth)


def format_value(v):
    return _format_value_impl(v, set(), 0)
