#!/bin/bash
set -e

echo "Установка U-Ray..."

# Проверка Python
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "Ошибка: Python не найден. Установите Python 3.10+."
    exit 1
fi

# Установка прав на исполнение
chmod +x uray
chmod +x run_tests.sh

echo "Установка завершена!"
echo "Запустить пример: $PYTHON_CMD main.py examples/perceptron.ur"
echo "Или используйте: ./uray examples/perceptron.ur"
