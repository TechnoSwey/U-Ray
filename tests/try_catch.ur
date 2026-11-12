try {
    print(1 / 0)
} catch e {
    print("поймали:", e)
}

try {
    x = 10
    print(x)
} catch e {
    print("не выполнится")
}

try {
    throw "своя ошибка"
} catch e {
    print("throw:", e)
}
