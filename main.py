# main.py

import argparse
import os
import sys
from pathlib import Path

import values
from errors import URayError
from parser import parse
from interpreter import Interpreter
from tensor import set_backend
from ast_nodes import ExprStmt
from values import format_value
from formatter import format_source
from repl_completion import completion_candidates
from typechecker import StaticChecker

VERSION = "0.1.0"


def _needs_more_input(source):
    """Return true while a block, list, call, or dictionary is unfinished."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = set(pairs.values())
    stack = []
    in_string = False
    escaped = False
    i = 0
    while i < len(source):
        char = source[i]
        if in_string:
            if char == "\n":
                return False
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        else:
            if char == "/" and i + 1 < len(source) and source[i + 1] == "/":
                while i < len(source) and source[i] != "\n":
                    i += 1
                continue
            if char == '"':
                in_string = True
            elif char in pairs:
                stack.append(pairs[char])
            elif char in closing:
                if not stack or stack[-1] != char:
                    return False
                stack.pop()
        i += 1
    return bool(stack)


def _enable_tab_completion(interpreter):
    try:
        import readline
    except ImportError:
        return False

    state = {"matches": []}

    def complete(text, index):
        if index == 0:
            state["matches"] = completion_candidates(text, interpreter)
        if index < len(state["matches"]):
            return state["matches"][index]
        return None

    try:
        readline.set_completer_delims(" \t\n()[]{},;=+-*/%<>!")
        readline.set_completer(complete)
        readline.parse_and_bind("tab: complete")
    except (AttributeError, RuntimeError):
        return False
    return True


def repl():
    """Run a persistent line-oriented U-Ray prompt."""
    interpreter = Interpreter()
    checker = StaticChecker()
    interactive = sys.stdin.isatty()
    if interactive:
        _enable_tab_completion(interpreter)
    buffer = []
    while True:
        prompt = "uray... " if buffer else "uray> "
        try:
            line = input(prompt if interactive else "")
        except EOFError:
            if buffer:
                print("Ввод завершён до закрытия блока", file=sys.stderr)
                return 1
            if interactive:
                print()
            return 0
        except KeyboardInterrupt:
            if interactive:
                print()
            buffer.clear()
            continue

        if not buffer and not line.strip():
            continue
        buffer.append(line)
        source = "\n".join(buffer)
        if _needs_more_input(source):
            continue
        buffer.clear()
        try:
            program = parse(source)
            checked_symbols = checker.check(program)
            for statement in program.statements:
                result = interpreter.exec_statement(statement)
                if isinstance(statement, ExprStmt):
                    print(format_value(result))
            checker.symbols = checked_symbols
        except URayError as e:
            print(f"ошибка U-Ray: {e}", file=sys.stderr)
        except (EOFError, KeyboardInterrupt):
            if interactive:
                print()


def _build_argument_parser():
    parser = argparse.ArgumentParser(
        prog="uray",
        description="Интерпретатор языка программирования U-Ray.",
        epilog=(
            "Без файла запускается интерактивный режим. Встроенное "
            "нативное ядро ускоряет MLP без NumPy при --backend=native; "
            "--backend=numpy доступен как дополнительный backend. "
            "Команды: uray fmt FILE, uray check FILE."
        ),
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="файл исходного кода .ur",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"U-Ray {VERSION}",
    )
    parser.add_argument(
        "--repl",
        action="store_true",
        help="запустить интерактивный режим",
    )
    parser.add_argument(
        "--backend",
        choices=("python", "native", "numpy"),
        default="python",
        help=(
            "backend тензорных и пакетных ML-операций: python, "
            "native (встроенное C-ядро) или numpy"
        ),
    )
    return parser


def _format_command(argv):
    parser = argparse.ArgumentParser(
        prog="uray fmt",
        description="Форматирует исходный файл U-Ray.",
    )
    parser.add_argument("file", help="путь к .ur файлу")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--write",
        action="store_true",
        help="перезаписать файл отформатированным кодом",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="проверить форматирование без записи",
    )
    args = parser.parse_args(argv)
    path = Path(args.file)
    try:
        source = path.read_text(encoding="utf-8")
        formatted = format_source(source)
    except (OSError, URayError) as error:
        print(f"uray fmt: {error}", file=sys.stderr)
        return 1

    if args.check:
        if formatted != source:
            print(f"{path}: требуется форматирование", file=sys.stderr)
            return 1
        print(f"{path}: форматирование проверено")
        return 0
    if args.write:
        try:
            path.write_text(formatted, encoding="utf-8", newline="\n")
        except OSError as error:
            print(f"uray fmt: не удалось записать {path}: {error}", file=sys.stderr)
            return 1
        return 0
    sys.stdout.write(formatted)
    return 0


def _check_command(argv):
    parser = argparse.ArgumentParser(
        prog="uray check",
        description="Проверяет синтаксис и статически известные размерности.",
    )
    parser.add_argument("file", help="путь к .ur файлу")
    args = parser.parse_args(argv)
    try:
        source = Path(args.file).read_text(encoding="utf-8")
        program = parse(source)
        StaticChecker().check(program)
    except (OSError, URayError) as error:
        print(f"{args.file}: ошибка U-Ray: {error}", file=sys.stderr)
        return 1
    print(f"{args.file}: статическая проверка пройдена")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "fmt":
        return _format_command(argv[1:])
    if argv and argv[0] == "check":
        return _check_command(argv[1:])
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    if args.repl and args.file:
        parser.error("нельзя одновременно указать файл и --repl")

    try:
        set_backend(args.backend)
    except URayError as e:
        print(f"ошибка U-Ray: {e}", file=sys.stderr)
        return 1

    if args.repl or args.file is None:
        values.URAY_BASE_DIR = os.getcwd()
        return repl()

    path = args.file
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except OSError as e:
        print(f"Не удалось прочитать {path}: {e}", file=sys.stderr)
        return 1

    # Базовая директория — директория .ur-файла.
    # Относительные пути import, nn.save и nn.load считаются от неё.
    values.URAY_BASE_DIR = os.path.dirname(os.path.abspath(path))
    try:
        program = parse(src)
        StaticChecker().check(program)
        Interpreter().run(program)
    except URayError as e:
        print(f"{path}: ошибка U-Ray: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
