print("=== for по списку ===")
for x in [10, 20, 30] {
    print(x)
}

print("=== for по range ===")
for i in range(5) {
    print(i)
}

print("=== range с началом и концом ===")
for i in range(2, 5) {
    print(i)
}

print("=== range с шагом ===")
for i in range(0, 10, 3) {
    print(i)
}

print("=== break ===")
for x in range(10) {
    if x == 3 {
        break
    }
    print(x)
}

print("=== continue (пропускаем чётные) ===")
for x in range(6) {
    if x % 2 == 0 {
        continue
    }
    print(x)
}

print("=== while + break/continue ===")
i = 0
while i < 10 {
    i = i + 1
    if i == 2 {
        continue
    }
    if i == 5 {
        break
    }
    print(i)
}

print("=== вложенные циклы ===")
for i in range(3) {
    for j in range(3) {
        if j > i {
            break
        }
        print(i, j)
    }
}

print("=== сумма через for ===")
sum = 0
for x in range(1, 11) {
    sum = sum + x
}
print("sum =", sum)
