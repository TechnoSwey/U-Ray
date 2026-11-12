func check(n) {
    if n < 0 {
        throw "отрицательное число"
    }
    return n
}

try {
    print(check(5))
    print(check(0 - 1))
} catch e {
    print("ошибка:", e)
}
