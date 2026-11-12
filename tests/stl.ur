func double(x) {
    return x * 2
}

func is_even(x) {
    return x % 2 == 0
}

func add(a, b) {
    return a + b
}

a = [3, 1, 2]
sort(a)
print(a)

b = [1, 2, 3]
reverse(b)
print(b)

c = [1, 2, 3]
d = map(c, double)
print(d)

e = [1, 2, 3, 4, 5, 6]
f = filter(e, is_even)
print(f)

g = [1, 2, 3, 4]
sum = reduce(g, add, 0)
print(sum)

write_file("test_stl.txt", "hello")
append_file("test_stl.txt", " world")
print(read_file("test_stl.txt"))
