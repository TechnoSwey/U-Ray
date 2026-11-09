# main.py

import os
import sys

import values
from errors import URayError
from parser import parse
from interpreter import run


def main():
    if len(sys.argv) < 2:
        print("Использование: python main.py <файл.ur>")
        sys.exit(1)

    path = sys.argv[1]
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except FileNotFoundError:
        print(f"Файл не найден: {path}")
        sys.exit(1)

    # Базовая директория — директория .ur-файла.
    # nn.save/nn.load будут склеивать относительные пути с ней.
    values.URAY_BASE_DIR = os.path.dirname(os.path.abspath(path))

    try:
        program = parse(src)
        run(program)
    except URayError as e:
        print(f"{path}: ошибка U-Ray: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
