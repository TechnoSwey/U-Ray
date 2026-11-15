"""Tab-completion helpers for the interactive U-Ray prompt."""

import re

from ast_nodes import TypeDecl
from lexer import KEYWORDS
from values import DictValue, Module, StructInstance

_PATH_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$", re.UNICODE)


def _visible_value(interpreter, name):
    env = interpreter.env
    while env is not None:
        if name in env.vars:
            return env.vars[name]
        env = env.parent
    return interpreter.registry.get(name)


def _member_names(value):
    if isinstance(value, Module):
        return value.members.keys()
    if isinstance(value, StructInstance):
        return list(value.fields.keys()) + [
            method.name for method in value.methods
        ]
    if isinstance(value, TypeDecl):
        return list(value.fields) + [method.name for method in value.methods]
    if isinstance(value, DictValue):
        return [key for key in value.keys() if isinstance(key, str)]
    return []


def _resolve_path(interpreter, path):
    parts = path.split(".")
    value = _visible_value(interpreter, parts[0])
    if value is None:
        return None
    for part in parts[1:]:
        if isinstance(value, Module):
            value = value.members.get(part)
        elif isinstance(value, StructInstance):
            if part in value.fields:
                value = value.fields[part]
            else:
                value = next(
                    (method for method in value.methods if method.name == part),
                    None,
                )
        elif isinstance(value, TypeDecl):
            if part in value.fields:
                value = None
            else:
                value = next(
                    (method for method in value.methods if method.name == part),
                    None,
                )
        elif isinstance(value, DictValue):
            value = value[part] if part in value else None
        else:
            return None
        if value is None:
            return None
    return value


def completion_candidates(text, interpreter):
    """Return full readline replacement candidates for a typed prefix."""
    if "." in text:
        path, prefix = text.rsplit(".", 1)
        if not _PATH_RE.fullmatch(path):
            return []
        value = _resolve_path(interpreter, path)
        return sorted(
            f"{path}.{name}"
            for name in _member_names(value)
            if name.startswith(prefix)
        ) if value is not None else []

    names = set(KEYWORDS)
    names.update(interpreter.registry.modules)
    env = interpreter.env
    while env is not None:
        names.update(env.vars)
        env = env.parent
    return sorted(name for name in names if name.startswith(text))
