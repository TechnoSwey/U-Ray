# tensor.py

class Tensor:
    """Тензор без зависимостей. Внутри — вложенные списки float.
    Оптимизирован через встроенные zip/sum."""

    def __init__(self, data):
        self._data = self._to_float_list(data)

    @staticmethod
    def _to_float_list(data):
        if isinstance(data, list):
            return [Tensor._to_float_list(x) for x in data]
        return float(data)

    @property
    def data(self):
        return self._data

    def to_list(self):
        return self._data

    def shape(self):
        return self._shape_list(self._data)

    @staticmethod
    def _shape_list(data):
        if not isinstance(data, list):
            return []
        if not data:
            return [0]
        return [len(data)] + Tensor._shape_list(data[0])

    @staticmethod
    def zeros(shape):
        return Tensor(Tensor._fill(shape, 0.0))

    @staticmethod
    def ones(shape):
        return Tensor(Tensor._fill(shape, 1.0))

    @staticmethod
    def _fill(shape, val):
        if len(shape) == 0:
            return val
        return [Tensor._fill(shape[1:], val) for _ in range(shape[0])]

    def get(self, idx):
        if isinstance(idx, tuple):
            cur = self._data
            for i in idx:
                cur = cur[i]
            return cur
        return self._data[idx]

    def set(self, idx, value):
        value = float(value)
        if isinstance(idx, tuple):
            cur = self._data
            for i in idx[:-1]:
                cur = cur[i]
            cur[idx[-1]] = value
            return
        self._data[idx] = value

    def slice(self, start, end):
        return Tensor(self._data[start:end])

    def add(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(Tensor._add(self._data, other._data))

    @staticmethod
    def _add(a, b):
        if isinstance(a, list):
            return [Tensor._add(x, y) for x, y in zip(a, b)]
        return a + b

    def sub(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(Tensor._sub(self._data, other._data))

    @staticmethod
    def _sub(a, b):
        if isinstance(a, list):
            return [Tensor._sub(x, y) for x, y in zip(a, b)]
        return a - b

    def mul(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(Tensor._mul(self._data, other._data))

    @staticmethod
    def _mul(a, b):
        if isinstance(a, list):
            return [Tensor._mul(x, y) for x, y in zip(a, b)]
        return a * b

    def matmul(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(Tensor._matmul(self._data, other._data))

    @staticmethod
    def _matmul(A, B):
        B_T = list(zip(*B))
        result = []
        for row in A:
            result_row = []
            for col in B_T:
                result_row.append(sum(x * y for x, y in zip(row, col)))
            result.append(result_row)
        return result

    def T(self):
        return Tensor(list(map(list, zip(*self._data))))

    def scalar_mul(self, s):
        return Tensor(Tensor._scalar_mul(self._data, s))

    @staticmethod
    def _scalar_mul(a, s):
        if isinstance(a, list):
            return [Tensor._scalar_mul(x, s) for x in a]
        return a * s

    def scalar_add(self, s):
        return Tensor(Tensor._scalar_add(self._data, s))

    @staticmethod
    def _scalar_add(a, s):
        if isinstance(a, list):
            return [Tensor._scalar_add(x, s) for x in a]
        return a + s

    def map(self, fn):
        return Tensor(Tensor._map(self._data, fn))

    @staticmethod
    def _map(a, fn):
        if isinstance(a, list):
            return [Tensor._map(x, fn) for x in a]
        return fn(a)

    def outer(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor([[x * y for y in other._data] for x in self._data])

    def sum(self):
        return Tensor._sum(self._data)

    @staticmethod
    def _sum(a):
        if isinstance(a, list):
            return sum(Tensor._sum(x) for x in a)
        return a

    def max(self):
        return Tensor._max(self._data)

    @staticmethod
    def _max(a):
        if isinstance(a, list):
            return max(Tensor._max(x) for x in a)
        return a

    def __repr__(self):
        return f"<Tensor shape={self.shape()}>"
