#!/bin/bash
# Раннер тестов U-Ray.
# Ищет tests/*.ur, гоняет через uray, сравнивает с tests/*.expected.
#
# Флаги:
#   --fast  — пропустить ML-тесты (ml_*), только быстрые

set -u

URAY_DIR="$(cd "$(dirname "$0")" && pwd)"
TESTS_DIR="$URAY_DIR/tests"
PASS=0
FAIL=0
FAST=0

for arg in "$@"; do
    case "$arg" in
        --fast) FAST=1 ;;
    esac
done

if [ ! -d "$TESTS_DIR" ]; then
    echo "Нет папки tests/"
    exit 1
fi

for ur in "$TESTS_DIR"/*.ur; do
    [ -e "$ur" ] || continue
    base="$(basename "$ur" .ur)"
    if [ "$FAST" -eq 1 ] && [[ "$base" == ml_* ]]; then
        continue
    fi
    expected="$TESTS_DIR/$base.expected"
    if [ ! -f "$expected" ]; then
        echo "SKIP $base (нет .expected)"
        continue
    fi
    actual="$("$URAY_DIR/uray" "$ur" 2>&1)"
    status=$?
    actual="${actual//$'\r'/}"
    if [ "$status" -eq 0 ] && [ "$actual" == "$(cat "$expected")" ]; then
        echo "PASS $base"
        PASS=$((PASS + 1))
    else
        echo "FAIL $base"
        if [ "$status" -ne 0 ]; then
            echo "--- exit status ---"
            echo "$status"
        fi
        echo "--- expected ---"
        cat "$expected"
        echo "--- actual ---"
        echo "$actual"
        echo "---------------"
        FAIL=$((FAIL + 1))
    fi
done

if (cd "$URAY_DIR" && python3 -m unittest discover \
    -s "$TESTS_DIR" -p 'test_*.py'); then
    echo "PASS pytorch_export_unit"
    PASS=$((PASS + 1))
else
    echo "FAIL pytorch_export_unit"
    FAIL=$((FAIL + 1))
fi

check_cli() {
    name="$1"
    actual="$2"
    expected="$3"
    if [ "$actual" == "$expected" ]; then
        echo "PASS $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL $name"
        echo "--- expected ---"
        printf '%s\n' "$expected"
        echo "--- actual ---"
        printf '%s\n' "$actual"
        FAIL=$((FAIL + 1))
    fi
}

check_cli "cli_version" "$("$URAY_DIR/uray" --version 2>&1)" "U-Ray 0.1.0"

help_output="$("$URAY_DIR/uray" --help 2>&1)"
if [[ "$help_output" == *"--backend"* \
    && "$help_output" == *"--repl"* \
    && "$help_output" == *"uray fmt"* \
    && "$help_output" == *"uray check"* ]]; then
    echo "PASS cli_help"
    PASS=$((PASS + 1))
else
    echo "FAIL cli_help"
    echo "$help_output"
    FAIL=$((FAIL + 1))
fi

repl_output="$(
    printf 'func add(a, b) {\n    return a + b\n}\nadd(2, 3)\n' \
        | "$URAY_DIR/uray" --repl 2>&1
)"
check_cli "cli_repl" "$repl_output" "5"

stdin_repl_output="$(
    printf '2 + 2\n' | "$URAY_DIR/uray" 2>&1
)"
check_cli "cli_stdin_repl" "$stdin_repl_output" "4"

numpy_output="$(
    "$URAY_DIR/uray" --backend=numpy "$TESTS_DIR/numpy_backend.ur" 2>&1
)"
numpy_status=$?
if [ "$numpy_status" -eq 0 ]; then
    check_cli "cli_numpy_backend" "${numpy_output//$'\r'/}" \
        "$(cat "$TESTS_DIR/numpy_backend.expected")"
elif [[ "$numpy_output" == *"Backend numpy требует NumPy"* ]]; then
    echo "SKIP cli_numpy_backend (NumPy не установлен)"
else
    echo "FAIL cli_numpy_backend"
    echo "$numpy_output"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "PASS: $PASS, FAIL: $FAIL"
[ "$FAIL" -eq 0 ]
