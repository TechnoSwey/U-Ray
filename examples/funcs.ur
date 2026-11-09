func add(a, b) {
    return a + b
}

func greet(name) {
    print("привет, " + name)
}

func fact(n) {
    if n <= 1 {
        return 1
    }
    return n * fact(n - 1)
}

func fib(n) {
    if n < 2 {
        return n
    }
    return fib(n - 1) + fib(n - 2)
}

func no_return() {
    x = 1
}

print(add(2, 3))
greet("мир")
print(fact(5))
print(fib(10))
print(no_return())
