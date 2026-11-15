type Point {
    x
    y
}

type LocatedPoint {
    label
    point
}

item = LocatedPoint("home", Point(3, 4))
match item {
    LocatedPoint { label: name, point: Point { x: x, y: _ } } {
        print("point", name, "x =", x)
    }
    _ {
        print("not a located point")
    }
}
