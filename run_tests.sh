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
    actual="${actual//$'\r'/}"
    if [ "$actual" == "$(cat "$expected")" ]; then
        echo "PASS $base"
        PASS=$((PASS + 1))
    else
        echo "FAIL $base"
        echo "--- expected ---"
        cat "$expected"
        echo "--- actual ---"
        echo "$actual"
        echo "---------------"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "PASS: $PASS, FAIL: $FAIL"
[ "$FAIL" -eq 0 ]
