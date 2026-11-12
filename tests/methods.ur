import math

type Point {
    x
    y
    func dist() {
        return math.sqrt(self.x * self.x + self.y * self.y)
    }
}

p = Point(3, 4)
print(p.dist())
