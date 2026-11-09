import math
import linalg

print(math.sqrt(2))
print(math.pi)
print(math.e)
print(math.exp(0))
print(math.abs(0 - 7))

print(linalg.dot([1, 2, 3], [4, 5, 6]))
print(linalg.matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]))
print(linalg.transpose([[1, 2, 3], [4, 5, 6]]))
print(linalg.zeros(4))
print(linalg.ones(3))
print(linalg.sum([1, 2, 3, 4]))
print(linalg.max([1, 5, 2]))
print(linalg.min([1, 5, 2]))

r = math.rand()
print(r >= 0)
print(r < 1)
