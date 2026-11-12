type Point {
    x
    y
}

p = Point(10, 20)
print(p.x)
print(p.y)

p.x = 99
print(p.x)
print(p)

type Person {
    name
    age
}

bob = Person("Bob", 42)
print(bob.name, bob.age)
