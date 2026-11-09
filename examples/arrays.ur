a = [1, 2, 3]
print(a)
print(a[0])
print(a[2])

a[0] = 10
print(a)

push(a, 4)
print(a)
print(len(a))
print(pop(a))
print(a)

m = [[1, 2], [3, 4]]
print(m)
print(m[1][0])
m[1][0] = 99
print(m)

s = ["a", "b", "c"]
print(s[1])
print(len(s))

mixed = [1, "два", true, 3.5]
print(mixed)

sum = 0
i = 0
while i < len(a) {
    sum = sum + a[i]
    i = i + 1
}
print(sum)
