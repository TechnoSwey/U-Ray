type Point {
    x
    y
}

type LocatedPoint {
    label
    point
}

p = Point(3, 4)
item = LocatedPoint("home", p)

match item {
    LocatedPoint { label: name, point: Point { x: x, y: _ } } {
        print(name, x)
    }
    _ {
        print("no match")
    }
}

match p {
    Point(0, _) {
        print("origin")
    }
    Point { x: x, y } {
        print("coordinates", x, y)
    }
}
