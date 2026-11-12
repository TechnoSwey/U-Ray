x = 2
match x {
    1 { print("один") }
    2 { print("два") }
    3 { print("три") }
    _ { print("другое") }
}

y = "hello"
match y {
    "world" { print("мир") }
    "hello" { print("привет") }
}

z = 99
match z {
    1 { print("один") }
    _ { print("другое") }
}
