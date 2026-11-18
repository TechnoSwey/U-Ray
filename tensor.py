# tensor.py

import importlib

from errors import RuntimeErrorUray

_backend = "python"
_numpy = None
_native = None


def get_numpy():
    """Return the configured NumPy module, or None for the Python backend."""
    return _numpy


def get_native():
    """Return the loaded U-Ray native engine, or None for another backend."""
    return _native


def set_backend(name):
    """Select the implementation used for tensor math operations."""
    global _backend, _numpy, _native
    if name == "python":
        _backend = "python"
        _numpy = None
        _native = None
        return
    if name == "native":
        from native_backend import load_native
        _native = load_native()
        _numpy = None
        _backend = "native"
        return
    if name != "numpy":
        raise RuntimeErrorUray(f"Неизвестный backend тензоров {name!r}")
    try:
        _numpy = importlib.import_module("numpy")
    except ImportError as e:
        raise RuntimeErrorUray(
            "Backend numpy требует NumPy. Установите его командой: "
            "python3 -m pip install numpy"
        ) from e
    _native = None
    _backend = "numpy"

class Tensor:
    """Тензор без зависимостей. Внутри — вложенные списки float.
    Оптимизирован через встроенные zip/sum."""

    def __init__(self, data):
        self._data = self._to_float_list(data)

    @classmethod
    def _from_native_data(cls, data):
        """Wrap a result produced by U-Ray's native kernels without recasting it."""
        tensor = cls.__new__(cls)
        tensor._data = data
        return tensor

    @staticmethod
    def _to_float_list(data):
        if isinstance(data, list):
            converted = [Tensor._to_float_list(x) for x in data]
            if converted:
                child_shape = Tensor._shape_list(converted[0])
                if any(Tensor._shape_list(x) != child_shape for x in converted[1:]):
                    raise RuntimeErrorUray("Тензор должен быть прямоугольным")
            return converted
        try:
            return float(data)
        except (TypeError, ValueError, OverflowError) as e:
            raise RuntimeErrorUray("Элементы тензора должны быть числами") from e

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
        Tensor._validate_shape(shape)
        if _backend == "numpy":
            return Tensor(_numpy.zeros(shape, dtype=float).tolist())
        return Tensor(Tensor._fill(shape, 0.0))

    @staticmethod
    def ones(shape):
        Tensor._validate_shape(shape)
        if _backend == "numpy":
            return Tensor(_numpy.ones(shape, dtype=float).tolist())
        return Tensor(Tensor._fill(shape, 1.0))

    @staticmethod
    def _validate_shape(shape):
        if not isinstance(shape, list) or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 0
            for dim in shape
        ):
            raise RuntimeErrorUray(
                "Размерности тензора должны быть неотрицательными целыми числами"
            )

    @staticmethod
    def _fill(shape, val):
        if len(shape) == 0:
            return val
        return [Tensor._fill(shape[1:], val) for _ in range(shape[0])]

    def get(self, idx):
        indices = idx if isinstance(idx, tuple) else (idx,)
        cur = self._data
        for axis, index in enumerate(indices):
            if not isinstance(cur, list):
                raise RuntimeErrorUray("Слишком много индексов для тензора")
            self._check_index(index, len(cur), axis)
            cur = cur[index]
        return cur

    def set(self, idx, value):
        indices = idx if isinstance(idx, tuple) else (idx,)
        if not indices:
            raise RuntimeErrorUray("Для присваивания тензору нужен индекс")
        cur = self._data
        for axis, index in enumerate(indices):
            if not isinstance(cur, list):
                raise RuntimeErrorUray("Слишком много индексов для тензора")
            self._check_index(index, len(cur), axis)
            if axis == len(indices) - 1:
                if isinstance(cur[index], list):
                    raise RuntimeErrorUray(
                        "Для присваивания укажите индекс каждого измерения тензора"
                    )
                try:
                    cur[index] = float(value)
                except (TypeError, ValueError, OverflowError) as e:
                    raise RuntimeErrorUray("Значение тензора должно быть числом") from e
                return
            cur = cur[index]

    @staticmethod
    def _check_index(index, length, axis):
        if not isinstance(index, int) or isinstance(index, bool):
            raise RuntimeErrorUray("Индекс тензора должен быть целым числом")
        if index < 0 or index >= length:
            raise RuntimeErrorUray(
                f"Тензорный индекс {index} вне диапазона измерения {axis} "
                f"(размер {length})"
            )

    def slice(self, start, end):
        return Tensor(self._data[start:end])

    def add(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        self._require_same_shape(other, "сложения")
        if _backend == "numpy":
            return self._numpy_binary(other, _numpy.add, "сложения")
        return Tensor(Tensor._add(self._data, other._data))

    def _numpy_binary(self, other, operation, label):
        try:
            result = operation(
                _numpy.asarray(self._data, dtype=float),
                _numpy.asarray(other._data, dtype=float),
            )
        except ValueError as e:
            raise RuntimeErrorUray(
                f"Несовместимые размерности тензоров для {label}"
            ) from e
        return Tensor(result.tolist())

    def _require_same_shape(self, other, label):
        if self.shape() != other.shape():
            raise RuntimeErrorUray(
                f"Несовместимые размерности тензоров для {label}: "
                f"{self.shape()} и {other.shape()}"
            )

    @staticmethod
    def _add(a, b):
        if isinstance(a, list):
            return [Tensor._add(x, y) for x, y in zip(a, b)]
        return a + b

    def sub(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        self._require_same_shape(other, "вычитания")
        if _backend == "numpy":
            return self._numpy_binary(other, _numpy.subtract, "вычитания")
        return Tensor(Tensor._sub(self._data, other._data))

    @staticmethod
    def _sub(a, b):
        if isinstance(a, list):
            return [Tensor._sub(x, y) for x, y in zip(a, b)]
        return a - b

    def mul(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        self._require_same_shape(other, "умножения")
        if _backend == "numpy":
            return self._numpy_binary(other, _numpy.multiply, "умножения")
        return Tensor(Tensor._mul(self._data, other._data))

    @staticmethod
    def _mul(a, b):
        if isinstance(a, list):
            return [Tensor._mul(x, y) for x, y in zip(a, b)]
        return a * b

    def matmul(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        left_shape = self.shape()
        right_shape = other.shape()
        if len(left_shape) != 2 or len(right_shape) != 2:
            raise RuntimeErrorUray("matmul ожидает два двумерных тензора")
        if left_shape[1] != right_shape[0]:
            raise RuntimeErrorUray(
                f"Несовместимые размерности для matmul: "
                f"{left_shape} и {right_shape}"
            )
        if _backend == "numpy":
            try:
                result = _numpy.matmul(
                    _numpy.asarray(self._data, dtype=float),
                    _numpy.asarray(other._data, dtype=float),
                )
            except ValueError as e:
                raise RuntimeErrorUray(
                    f"Несовместимые размерности тензоров для matmul: {e}"
                ) from e
            return Tensor(result.tolist())
        if _backend == "native":
            # Tensor has already enforced rectangular numeric inputs and
            # matmul has checked the inner dimensions, so avoid rebuilding
            # nested lists just to validate them a second time.
            return Tensor._from_native_data(_native.matmul(
                self._data, other._data, validated=True
            ))
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
        if len(self.shape()) != 2:
            raise RuntimeErrorUray("Транспонирование ожидает матрицу")
        if _backend == "numpy":
            return Tensor(
                _numpy.transpose(_numpy.asarray(self._data, dtype=float)).tolist()
            )
        return Tensor(list(map(list, zip(*self._data))))

    def scalar_mul(self, s):
        if _backend == "numpy":
            return Tensor(
                (_numpy.asarray(self._data, dtype=float) * s).tolist()
            )
        return Tensor(Tensor._scalar_mul(self._data, s))

    @staticmethod
    def _scalar_mul(a, s):
        if isinstance(a, list):
            return [Tensor._scalar_mul(x, s) for x in a]
        return a * s

    def scalar_add(self, s):
        if _backend == "numpy":
            return Tensor(
                (_numpy.asarray(self._data, dtype=float) + s).tolist()
            )
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
        if len(self.shape()) != 1 or len(other.shape()) != 1:
            raise RuntimeErrorUray("outer ожидает два вектора")
        if _backend == "numpy":
            return Tensor(
                _numpy.outer(
                    _numpy.asarray(self._data, dtype=float),
                    _numpy.asarray(other._data, dtype=float),
                ).tolist()
            )
        return Tensor([[x * y for y in other._data] for x in self._data])

    def sum(self):
        if _backend == "numpy":
            return float(_numpy.sum(_numpy.asarray(self._data, dtype=float)))
        return Tensor._sum(self._data)

    @staticmethod
    def _sum(a):
        if isinstance(a, list):
            return sum(Tensor._sum(x) for x in a)
        return a

    def max(self):
        if any(dim == 0 for dim in self.shape()):
            raise RuntimeErrorUray("max от пустого тензора")
        if _backend == "numpy":
            try:
                return float(_numpy.max(_numpy.asarray(self._data, dtype=float)))
            except ValueError as e:
                raise RuntimeErrorUray("max от пустого тензора") from e
        return Tensor._max(self._data)

    @staticmethod
    def _max(a):
        if isinstance(a, list):
            return max(Tensor._max(x) for x in a)
        return a

    def __repr__(self):
        return f"<Tensor shape={self.shape()}>"
