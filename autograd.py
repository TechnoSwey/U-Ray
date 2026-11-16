"""Reverse-mode automatic differentiation for U-Ray tensors.

The existing Tensor class remains the lightweight, non-differentiable tensor
used by the language runtime. GradTensor adds an operation graph and is
intentionally composable: model code is built from tensor operations rather
than a trainer that knows one particular architecture.
"""

import math

from errors import RuntimeErrorUray
from tensor import Tensor, get_numpy


def _shape(value):
    return Tensor._shape_list(value)


def _map(value, fn):
    if isinstance(value, list):
        return [_map(item, fn) for item in value]
    return fn(value)


def _zip_map(left, right, fn):
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise RuntimeErrorUray("Несовместимые размерности тензоров")
        return [_zip_map(a, b, fn) for a, b in zip(left, right)]
    return fn(left, right)


def _zeros_like(value):
    return _map(value, lambda _: 0.0)


def _ones_like(value):
    return _map(value, lambda _: 1.0)


def _add_data(left, right):
    np = get_numpy()
    if np is not None:
        return _array_to_data(
            np.asarray(left, dtype=float) + np.asarray(right, dtype=float)
        )
    return _zip_map(left, right, lambda a, b: a + b)


def _scale_data(value, scalar):
    np = get_numpy()
    if np is not None:
        return _array_to_data(np.asarray(value, dtype=float) * scalar)
    return _map(value, lambda x: x * scalar)


def _sum_data(value):
    np = get_numpy()
    if np is not None:
        return float(np.sum(np.asarray(value, dtype=float)))
    if isinstance(value, list):
        return sum(_sum_data(item) for item in value)
    return float(value)


def _count(value):
    np = get_numpy()
    if np is not None:
        return int(np.asarray(value).size)
    if isinstance(value, list):
        return sum(_count(item) for item in value)
    return 1


def _array_to_data(value):
    if getattr(value, "ndim", 0) == 0:
        return float(value)
    return value.tolist()


def _unbroadcast(value, target_shape):
    np = get_numpy()
    if np is None:
        return _sum_to_shape(value, target_shape)
    result = np.asarray(value, dtype=float)
    if not target_shape:
        return float(result.sum())
    while result.ndim > len(target_shape):
        result = result.sum(axis=0)
    for axis, target_dim in enumerate(target_shape):
        if target_dim == 1 and result.shape[axis] != 1:
            result = result.sum(axis=axis, keepdims=True)
    if tuple(result.shape) != tuple(target_shape):
        raise RuntimeErrorUray(
            f"Невозможно свести градиент формы {result.shape} к {target_shape}"
        )
    return _array_to_data(result)


def _sum_axis0(value):
    if not value:
        return []
    if not isinstance(value[0], list):
        return sum(value)
    return [
        _sum_axis0([row[index] for row in value])
        for index in range(len(value[0]))
    ]


def _sum_to_shape(value, target_shape):
    current_shape = _shape(value)
    if current_shape == target_shape:
        return value
    if not target_shape:
        return _sum_data(value)
    if (
        len(target_shape) == 1
        and len(current_shape) >= 2
        and current_shape[-1] == target_shape[0]
    ):
        reduced = value
        for _ in range(len(current_shape) - 1):
            reduced = _sum_axis0(reduced)
        return reduced
    raise RuntimeErrorUray(
        f"Невозможно свести градиент формы {current_shape} к {target_shape}"
    )


def _broadcast_binary(left, right, operation):
    left_shape = _shape(left)
    right_shape = _shape(right)
    if left_shape == right_shape:
        return _zip_map(left, right, operation)
    if not left_shape:
        return _map(right, lambda x: operation(left, x))
    if not right_shape:
        return _map(left, lambda x: operation(x, right))
    if (
        len(left_shape) >= 2
        and len(right_shape) == 1
        and left_shape[-1] == right_shape[0]
    ):
        def add_last(row, vector):
            if isinstance(row, list) and row and isinstance(row[0], list):
                return [add_last(item, vector) for item in row]
            return _zip_map(row, vector, operation)
        return add_last(left, right)
    if (
        len(right_shape) >= 2
        and len(left_shape) == 1
        and right_shape[-1] == left_shape[0]
    ):
        return _broadcast_binary(right, left, lambda a, b: operation(b, a))
    raise RuntimeErrorUray(
        f"Несовместимые размерности для broadcasting: {left_shape} и {right_shape}"
    )


def _flatten(value):
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return [value]


def _from_flat(values, shape):
    if not shape:
        if len(values) != 1:
            raise RuntimeErrorUray("Неверное число элементов при reshape")
        return values[0]
    size = shape[0]
    tail_size = math.prod(shape[1:]) if len(shape) > 1 else 1
    return [
        _from_flat(values[index * tail_size:(index + 1) * tail_size], shape[1:])
        for index in range(size)
    ]


def _transpose(data):
    if not isinstance(data, list) or not data or not isinstance(data[0], list):
        raise RuntimeErrorUray("Транспонирование ожидает двумерный тензор")
    if any(len(row) != len(data[0]) for row in data):
        raise RuntimeErrorUray("Транспонирование ожидает прямоугольную матрицу")
    return [list(column) for column in zip(*data)]


def _matmul(left, right):
    left_shape = _shape(left)
    right_shape = _shape(right)
    if len(left_shape) not in (1, 2) or len(right_shape) not in (1, 2):
        raise RuntimeErrorUray("matmul поддерживает векторы и матрицы")
    if left_shape[-1] != right_shape[0]:
        raise RuntimeErrorUray(
            f"Несовместимые размерности для matmul: {left_shape} и {right_shape}"
        )
    np = get_numpy()
    if np is not None:
        try:
            result = np.matmul(
                np.asarray(left, dtype=float), np.asarray(right, dtype=float)
            )
            return result.item() if result.ndim == 0 else result.tolist()
        except (TypeError, ValueError) as error:
            raise RuntimeErrorUray(f"matmul: {error}") from error
    if len(left_shape) == 1 and len(right_shape) == 1:
        return sum(a * b for a, b in zip(left, right))
    if len(left_shape) == 2 and len(right_shape) == 1:
        return [sum(a * b for a, b in zip(row, right)) for row in left]
    if len(left_shape) == 1 and len(right_shape) == 2:
        return [
            sum(left[k] * right[k][column] for k in range(left_shape[0]))
            for column in range(right_shape[1])
        ]
    columns = list(zip(*right))
    return [
        [sum(a * b for a, b in zip(row, column)) for column in columns]
        for row in left
    ]


def _outer(left, right):
    return [[a * b for b in right] for a in left]


def _softmax_rows(data):
    if isinstance(data, list) and data and isinstance(data[0], list):
        return [_softmax_rows(row) for row in data]
    values = [float(x) for x in data]
    if not values:
        raise RuntimeErrorUray("softmax от пустого тензора")
    maximum = max(values)
    exp_values = [math.exp(x - maximum) for x in values]
    total = sum(exp_values)
    return [x / total for x in exp_values]


def _rowwise(data, fn):
    if isinstance(data, list) and data and isinstance(data[0], list):
        return [_rowwise(row, fn) for row in data]
    return fn(data)


class GradTensor(Tensor):
    """Tensor with a reverse-mode computation graph.

    Gradients are accumulated within one backward pass. Call zero_grad before
    the next optimizer step, as with the common ML framework convention.
    """

    def __init__(self, data, requires_grad=False, parents=(), backward_fn=None):
        super().__init__(data)
        self.requires_grad = bool(requires_grad)
        self.grad = None
        self._parents = tuple(parents)
        self._backward_fn = backward_fn
        self._graph_consumed = False

    def _result(self, data, parents, backward_fn):
        return GradTensor(
            data,
            any(parent.requires_grad for parent in parents),
            parents,
            backward_fn,
        )

    def add(self, other):
        other = as_grad_tensor(other)
        left_shape, right_shape = self.shape(), other.shape()
        np = get_numpy()
        if np is not None:
            result = _array_to_data(
                np.asarray(self.data, dtype=float)
                + np.asarray(other.data, dtype=float)
            )
        else:
            result = _broadcast_binary(self.data, other.data, lambda a, b: a + b)

        def backward(grad):
            return (
                _unbroadcast(grad, left_shape),
                _unbroadcast(grad, right_shape),
            )

        return self._result(result, (self, other), backward)

    def sub(self, other):
        other = as_grad_tensor(other)
        left_shape, right_shape = self.shape(), other.shape()
        np = get_numpy()
        if np is not None:
            result = _array_to_data(
                np.asarray(self.data, dtype=float)
                - np.asarray(other.data, dtype=float)
            )
        else:
            result = _broadcast_binary(self.data, other.data, lambda a, b: a - b)

        def backward(grad):
            return (
                _unbroadcast(grad, left_shape),
                _unbroadcast(_scale_data(grad, -1.0), right_shape),
            )

        return self._result(result, (self, other), backward)

    def mul(self, other):
        other = as_grad_tensor(other)
        left_shape, right_shape = self.shape(), other.shape()
        np = get_numpy()
        if np is not None:
            left_array = np.asarray(self.data, dtype=float)
            right_array = np.asarray(other.data, dtype=float)
            result = _array_to_data(left_array * right_array)
        else:
            left_array = right_array = None
            result = _broadcast_binary(self.data, other.data, lambda a, b: a * b)

        def backward(grad):
            if np is not None:
                grad_array = np.asarray(grad, dtype=float)
                left_grad = _array_to_data(grad_array * right_array)
                right_grad = _array_to_data(grad_array * left_array)
            else:
                left_grad = _broadcast_binary(grad, other.data, lambda g, b: g * b)
                right_grad = _broadcast_binary(grad, self.data, lambda g, a: g * a)
            return (
                _unbroadcast(left_grad, left_shape),
                _unbroadcast(right_grad, right_shape),
            )

        return self._result(result, (self, other), backward)

    def div(self, other):
        other = as_grad_tensor(other)
        left_shape, right_shape = self.shape(), other.shape()
        np = get_numpy()
        if np is not None:
            left_array = np.asarray(self.data, dtype=float)
            right_array = np.asarray(other.data, dtype=float)
            if np.any(right_array == 0):
                raise RuntimeErrorUray("Деление тензоров на ноль")
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                result = _array_to_data(left_array / right_array)
        elif _sum_data(_map(other.data, lambda x: 1.0 if x == 0 else 0.0)):
            raise RuntimeErrorUray("Деление тензоров на ноль")
        else:
            left_array = right_array = None
            result = _broadcast_binary(self.data, other.data, lambda a, b: a / b)

        def backward(grad):
            if np is not None:
                grad_array = np.asarray(grad, dtype=float)
                left_grad = _array_to_data(grad_array / right_array)
                right_grad = _array_to_data(
                    -grad_array * left_array / (right_array * right_array)
                )
            else:
                left_local = _broadcast_binary(
                    self.data, other.data, lambda a, b: 1.0 / b
                )
                right_local = _broadcast_binary(
                    self.data, other.data, lambda a, b: -a / (b * b)
                )
                left_grad = _broadcast_binary(
                    grad, left_local, lambda g, x: g * x
                )
                right_grad = _broadcast_binary(
                    grad, right_local, lambda g, x: g * x
                )
            return (
                _unbroadcast(left_grad, left_shape),
                _unbroadcast(right_grad, right_shape),
            )

        return self._result(result, (self, other), backward)

    def pow(self, exponent):
        if not isinstance(exponent, (int, float)) or isinstance(exponent, bool):
            raise RuntimeErrorUray("Степень тензора должна быть числом")
        exponent = float(exponent)
        np = get_numpy()
        try:
            if np is not None:
                array = np.asarray(self.data, dtype=float)
                result = _array_to_data(np.power(array, exponent))
                gradient = (
                    np.zeros_like(array)
                    if exponent == 0.0
                    else exponent * np.power(array, exponent - 1.0)
                )
            else:
                result = _map(self.data, lambda value: float(value) ** exponent)
                gradient = (
                    _zeros_like(self.data)
                    if exponent == 0.0
                    else _map(
                        self.data,
                        lambda value: exponent
                        * (float(value) ** (exponent - 1.0)),
                    )
                )
        except (ValueError, OverflowError, ZeroDivisionError) as error:
            raise RuntimeErrorUray(f"Степень тензора: {error}") from error

        def backward(grad):
            if np is not None:
                return (
                    _array_to_data(np.asarray(grad, dtype=float) * gradient),
                )
            return (
                _broadcast_binary(grad, gradient, lambda g, local: g * local),
            )

        return self._result(
            result,
            (self,),
            backward,
        )

    def matmul(self, other):
        other = as_grad_tensor(other)
        left_shape, right_shape = self.shape(), other.shape()
        result = _matmul(self.data, other.data)

        def backward(grad):
            if len(left_shape) == 1 and len(right_shape) == 1:
                return (
                    [grad * value for value in other.data],
                    [grad * value for value in self.data],
                )
            if len(left_shape) == 2 and len(right_shape) == 1:
                return (
                    _outer(grad, other.data),
                    _matmul(_transpose(self.data), grad),
                )
            if len(left_shape) == 1 and len(right_shape) == 2:
                return (
                    _matmul(other.data, grad),
                    _outer(self.data, grad),
                )
            return (
                _matmul(grad, _transpose(other.data)),
                _matmul(_transpose(self.data), grad),
            )

        return self._result(result, (self, other), backward)

    def transpose(self):
        data = _transpose(self.data)
        return self._result(data, (self,), lambda grad: (_transpose(grad),))

    def reshape(self, shape):
        if not isinstance(shape, list) or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < -1
            for dim in shape
        ) or shape.count(-1) > 1:
            raise RuntimeErrorUray(
                "reshape ожидает размеры >= 0 и не более одного -1"
            )
        values = None
        np = get_numpy()
        size = _count(self.data)
        if -1 in shape:
            known = math.prod(dim for dim in shape if dim != -1)
            if known == 0 or size % known:
                raise RuntimeErrorUray("reshape: невозможно вывести размер -1")
            shape = [size // known if dim == -1 else dim for dim in shape]
        if size != math.prod(shape):
            raise RuntimeErrorUray(
                f"reshape: {size} элементов нельзя разместить в форме {shape}"
            )
        old_shape = self.shape()
        if np is not None:
            data = _array_to_data(
                np.asarray(self.data, dtype=float).reshape(shape)
            )
            backward = lambda grad: (
                _array_to_data(np.asarray(grad).reshape(old_shape)),
            )
        else:
            values = _flatten(self.data)
            data = _from_flat(values, shape)
            backward = lambda grad: (_from_flat(_flatten(grad), old_shape),)
        return self._result(
            data,
            (self,),
            backward,
        )

    def flatten(self, preserve_batch=True):
        if preserve_batch and len(self.shape()) >= 2:
            return self.reshape([self.shape()[0], -1])
        return self.reshape([_count(self.data)])

    def sum_all(self):
        data = _sum_data(self.data)

        def backward(grad):
            np = get_numpy()
            if np is not None:
                return (
                    _array_to_data(
                        np.full_like(np.asarray(self.data, dtype=float), grad)
                    ),
                )
            return (_map(self.data, lambda _: grad),)

        return self._result(
            data,
            (self,),
            backward,
        )

    def mean(self):
        size = _count(self.data)
        if size == 0:
            raise RuntimeErrorUray("mean от пустого тензора")
        return self.sum_all().scalar_mul(1.0 / size)

    def scalar_mul(self, scalar):
        if not isinstance(scalar, (int, float)) or isinstance(scalar, bool):
            raise RuntimeErrorUray("Умножение требует число")
        data = _scale_data(self.data, float(scalar))
        return self._result(
            data, (self,), lambda grad: (_scale_data(grad, float(scalar)),)
        )

    def relu(self):
        np = get_numpy()
        if np is not None:
            array = np.asarray(self.data, dtype=float)
            output = _array_to_data(np.maximum(array, 0.0))
        else:
            array = None
            output = _map(self.data, lambda x: max(0.0, x))

        def backward(grad):
            if np is not None:
                return (_array_to_data(np.asarray(grad) * (array > 0.0)),)
            return (
                _zip_map(
                    grad,
                    self.data,
                    lambda g, x: g if x > 0.0 else 0.0,
                ),
            )

        return self._result(
            output,
            (self,),
            backward,
        )

    def sigmoid(self):
        np = get_numpy()
        if np is not None:
            array = np.asarray(self.data, dtype=float)
            clipped = np.clip(array, -700.0, 700.0)
            output = _array_to_data(1.0 / (1.0 + np.exp(-clipped)))
        else:
            output = _map(
                self.data,
                lambda x: 1.0 / (1.0 + math.exp(-x))
                if x >= 0
                else math.exp(x) / (1.0 + math.exp(x)),
            )

        def backward(grad):
            if np is not None:
                grad_array = np.asarray(grad, dtype=float)
                output_array = np.asarray(output, dtype=float)
                return (
                    _array_to_data(
                        grad_array * output_array * (1.0 - output_array)
                    ),
                )
            return (
                _zip_map(
                    _zip_map(grad, output, lambda g, y: g * y),
                    output,
                    lambda gy, y: gy * (1.0 - y),
                ),
            )

        return self._result(
            output,
            (self,),
            backward,
        )

    def tanh(self):
        np = get_numpy()
        if np is not None:
            array = np.asarray(self.data, dtype=float)
            output = _array_to_data(np.tanh(array))
        else:
            output = _map(self.data, math.tanh)

        def backward(grad):
            if np is not None:
                grad_array = np.asarray(grad, dtype=float)
                output_array = np.asarray(output, dtype=float)
                return (
                    _array_to_data(grad_array * (1.0 - output_array * output_array)),
                )
            return (
                _zip_map(
                    _zip_map(grad, output, lambda g, y: g * y),
                    output,
                    lambda gy, y: gy * (1.0 - y * y),
                ),
            )

        return self._result(
            output,
            (self,),
            backward,
        )

    def softmax(self):
        np = get_numpy()
        if np is not None:
            array = np.asarray(self.data, dtype=float)
            shifted = array - np.max(array, axis=-1, keepdims=True)
            exp_values = np.exp(shifted)
            output = _array_to_data(
                exp_values / np.sum(exp_values, axis=-1, keepdims=True)
            )
        else:
            output = _softmax_rows(self.data)

        def backward(grad):
            if np is not None:
                grad_array = np.asarray(grad, dtype=float)
                output_array = np.asarray(output, dtype=float)
                dot = np.sum(grad_array * output_array, axis=-1, keepdims=True)
                return (
                    _array_to_data(output_array * (grad_array - dot)),
                )
            def rows_grad(g_row, y_row):
                if (
                    isinstance(g_row, list)
                    and g_row
                    and isinstance(g_row[0], list)
                ):
                    return [
                        rows_grad(g, y)
                        for g, y in zip(g_row, y_row)
                    ]
                dot = sum(g * y for g, y in zip(g_row, y_row))
                return [y * (g - dot) for g, y in zip(g_row, y_row)]
            return (rows_grad(grad, output),)

        return self._result(output, (self,), backward)

    def mse_loss(self, target):
        target = as_grad_tensor(target)
        left_shape, right_shape = self.shape(), target.shape()
        if left_shape != right_shape:
            raise RuntimeErrorUray(
                f"MSE ожидает одинаковые формы, получено {left_shape} и {right_shape}"
            )
        size = _count(self.data)
        if size == 0:
            raise RuntimeErrorUray("MSE от пустого тензора")
        diffs = _zip_map(self.data, target.data, lambda a, b: a - b)
        loss = _sum_data(_map(diffs, lambda x: x * x)) / size

        def backward(grad):
            scale = 2.0 * grad / size
            return (
                _scale_data(diffs, scale),
                _scale_data(diffs, -scale),
            )

        return self._result(loss, (self, target), backward)

    def cross_entropy(self, target):
        target = as_grad_tensor(target)
        if self.shape() != target.shape() or len(self.shape()) not in (1, 2):
            raise RuntimeErrorUray(
                "cross_entropy ожидает logits и labels одинаковой формы [C] или [N,C]"
            )
        probabilities = _softmax_rows(self.data)
        single = len(self.shape()) == 1
        logits_rows = [self.data] if single else self.data
        target_rows = [target.data] if single else target.data
        probability_rows = [probabilities] if single else probabilities
        eps = 1e-12
        loss = 0.0
        for p_row, y_row in zip(probability_rows, target_rows):
            loss -= sum(float(y) * math.log(max(float(p), eps)) for p, y in zip(p_row, y_row))
        loss /= len(logits_rows)

        def backward(grad):
            result = [
                [(p - y) * grad / len(logits_rows) for p, y in zip(p_row, y_row)]
                for p_row, y_row in zip(probability_rows, target_rows)
            ]
            return (result[0] if single else result, None)

        # Labels are data, not trainable parameters; the second returned
        # gradient is ignored by the graph engine.
        return self._result(loss, (self, target), backward)

    def categorical_cross_entropy(self, target):
        target = as_grad_tensor(target)
        if self.shape() != target.shape() or len(self.shape()) not in (1, 2):
            raise RuntimeErrorUray(
                "categorical_cross_entropy ожидает вероятности и labels формы [C] или [N,C]"
            )
        single = len(self.shape()) == 1
        probabilities = [self.data] if single else self.data
        targets = [target.data] if single else target.data
        classes = len(probabilities[0]) if probabilities else 0
        rows = len(probabilities)
        if not classes or not rows:
            raise RuntimeErrorUray("categorical_cross_entropy от пустого тензора")
        eps = 1e-12
        loss = 0.0
        for probability_row, target_row in zip(probabilities, targets):
            loss -= sum(
                float(label) * math.log(max(float(probability), eps))
                for probability, label in zip(probability_row, target_row)
            )
        loss /= rows * classes

        def backward(grad):
            result = [
                [
                    -float(label) * grad / (rows * classes * float(probability))
                    if float(probability) > eps else 0.0
                    for probability, label in zip(probability_row, target_row)
                ]
                for probability_row, target_row in zip(probabilities, targets)
            ]
            return (result[0] if single else result, None)

        return self._result(loss, (self, target), backward)

    def binary_cross_entropy_probabilities(self, target):
        target = as_grad_tensor(target)
        if self.shape() != target.shape() or not self.shape():
            raise RuntimeErrorUray(
                "binary_cross_entropy ожидает вероятности и labels одинаковой формы"
            )
        size = _count(self.data)
        eps = 1e-12
        loss = 0.0
        gradients = []
        for probability, label in zip(_flatten(self.data), _flatten(target.data)):
            p = min(max(float(probability), eps), 1.0 - eps)
            y = float(label)
            loss -= y * math.log(p) + (1.0 - y) * math.log(1.0 - p)
            gradients.append(
                0.0
                if probability <= eps or probability >= 1.0 - eps
                else (p - y) / (p * (1.0 - p) * size)
            )
        loss /= size
        gradient = _from_flat(gradients, self.shape())
        return self._result(
            loss,
            (self, target),
            lambda grad: (_scale_data(gradient, grad), None),
        )

    def binary_cross_entropy(self, target):
        target = as_grad_tensor(target)
        if self.shape() != target.shape() or not self.shape():
            raise RuntimeErrorUray(
                "binary_cross_entropy ожидает logits и labels одинаковой ненулевой формы"
            )
        size = _count(self.data)
        loss = _zip_map(
            self.data,
            target.data,
            lambda x, y: max(x, 0.0) - x * y + math.log1p(math.exp(-abs(x))),
        )
        loss = _sum_data(loss) / size

        def backward(grad):
            def sigmoid(x):
                if x >= 0:
                    return 1.0 / (1.0 + math.exp(-x))
                exp_x = math.exp(x)
                return exp_x / (1.0 + exp_x)
            values = _zip_map(
                self.data,
                target.data,
                lambda x, y: (sigmoid(x) - y) * grad / size,
            )
            return (values, None)

        return self._result(loss, (self, target), backward)

    def conv2d(self, weight, bias=None, stride=1, padding=0):
        weight = as_grad_tensor(weight)
        bias_tensor = as_grad_tensor(bias) if bias is not None else None
        parents = (self, weight) + ((bias_tensor,) if bias_tensor is not None else ())
        x_shape, w_shape = self.shape(), weight.shape()
        squeeze_batch = len(x_shape) == 3
        if squeeze_batch:
            x_data = [self.data]
            x_shape = [1] + x_shape
        elif len(x_shape) == 4:
            x_data = self.data
        else:
            raise RuntimeErrorUray("conv2d ожидает вход CHW или NCHW")
        if len(w_shape) != 4:
            raise RuntimeErrorUray("conv2d: веса должны иметь форму [out,in,kh,kw]")
        if (
            not isinstance(stride, int) or isinstance(stride, bool) or stride <= 0
            or not isinstance(padding, int) or isinstance(padding, bool) or padding < 0
        ):
            raise RuntimeErrorUray("conv2d: stride должен быть >0, padding >=0")
        batches, in_channels, height, width = x_shape
        out_channels, weight_channels, kernel_h, kernel_w = w_shape
        if in_channels != weight_channels:
            raise RuntimeErrorUray("conv2d: число входных каналов не совпадает с весами")
        out_h = (height + 2 * padding - kernel_h) // stride + 1
        out_w = (width + 2 * padding - kernel_w) // stride + 1
        if out_h <= 0 or out_w <= 0:
            raise RuntimeErrorUray("conv2d: ядро больше входного изображения")
        if bias_tensor is not None and bias_tensor.shape() != [out_channels]:
            raise RuntimeErrorUray("conv2d: bias должен иметь форму [out_channels]")

        np = get_numpy()
        if np is not None:
            x_array = np.asarray(x_data, dtype=float)
            w_array = np.asarray(weight.data, dtype=float)
            padded = np.pad(
                x_array,
                ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            )
            windows = np.lib.stride_tricks.sliding_window_view(
                padded, (kernel_h, kernel_w), axis=(2, 3)
            )[:, :, ::stride, ::stride, :, :]
            output = np.einsum("nchwkl,ockl->nohw", windows, w_array)
            if bias_tensor is not None:
                output += np.asarray(bias_tensor.data, dtype=float)[None, :, None, None]
            output_data = output.tolist()

            def backward(grad):
                grad_array = np.asarray(grad, dtype=float)
                grad_weight = np.einsum("nchwkl,nohw->ockl", windows, grad_array)
                grad_input_padded = np.zeros_like(padded)
                for row in range(kernel_h):
                    for col in range(kernel_w):
                        contribution = np.einsum(
                            "nohw,oc->nchw",
                            grad_array,
                            w_array[:, :, row, col],
                        )
                        grad_input_padded[
                            :, :, row:row + out_h * stride:stride,
                            col:col + out_w * stride:stride,
                        ] += contribution
                if padding:
                    grad_input = grad_input_padded[
                        :, :, padding:-padding, padding:-padding
                    ]
                else:
                    grad_input = grad_input_padded
                grad_values = [grad_input[0].tolist() if squeeze_batch else grad_input.tolist(),
                               grad_weight.tolist()]
                if bias_tensor is not None:
                    grad_values.append(grad_array.sum(axis=(0, 2, 3)).tolist())
                return tuple(grad_values)
        else:
            padded = [
                [
                    [[0.0] * (width + 2 * padding) for _ in range(height + 2 * padding)]
                    for _ in range(in_channels)
                ]
                for _ in range(batches)
            ]
            for n in range(batches):
                for c in range(in_channels):
                    for row in range(height):
                        for col in range(width):
                            padded[n][c][row + padding][col + padding] = x_data[n][c][row][col]
            output_data = [
                [
                    [[0.0] * out_w for _ in range(out_h)]
                    for _ in range(out_channels)
                ]
                for _ in range(batches)
            ]
            for n in range(batches):
                for oc in range(out_channels):
                    for oh in range(out_h):
                        for ow in range(out_w):
                            value = bias_tensor.data[oc] if bias_tensor is not None else 0.0
                            for ic in range(in_channels):
                                for kh in range(kernel_h):
                                    for kw in range(kernel_w):
                                        value += (
                                            padded[n][ic][oh * stride + kh][ow * stride + kw]
                                            * weight.data[oc][ic][kh][kw]
                                        )
                            output_data[n][oc][oh][ow] = value

            def backward(grad):
                grad_input_pad = _zeros_like(padded)
                grad_weight = _zeros_like(weight.data)
                grad_bias = [0.0] * out_channels
                grad_batches = [grad] if squeeze_batch else grad
                for n in range(batches):
                    for oc in range(out_channels):
                        for oh in range(out_h):
                            for ow in range(out_w):
                                g = grad_batches[n][oc][oh][ow]
                                grad_bias[oc] += g
                                for ic in range(in_channels):
                                    for kh in range(kernel_h):
                                        for kw in range(kernel_w):
                                            ih = oh * stride + kh
                                            iw = ow * stride + kw
                                            grad_weight[oc][ic][kh][kw] += padded[n][ic][ih][iw] * g
                                            grad_input_pad[n][ic][ih][iw] += weight.data[oc][ic][kh][kw] * g
                if padding:
                    grad_input = [
                        [
                            [row[padding:-padding] for row in channel[padding:-padding]]
                            for channel in sample
                        ]
                        for sample in grad_input_pad
                    ]
                else:
                    grad_input = grad_input_pad
                result = [grad_input[0] if squeeze_batch else grad_input, grad_weight]
                if bias_tensor is not None:
                    result.append(grad_bias)
                return tuple(result)
        if squeeze_batch:
            output_data = output_data[0]
        return self._result(output_data, parents, backward)

    def max_pool2d(self, kernel_size=2, stride=None):
        if stride is None:
            stride = kernel_size
        if (
            not isinstance(kernel_size, int) or isinstance(kernel_size, bool) or kernel_size <= 0
            or not isinstance(stride, int) or isinstance(stride, bool) or stride <= 0
        ):
            raise RuntimeErrorUray("max_pool2d: kernel и stride должны быть >0")
        original_shape = self.shape()
        squeeze_batch = len(original_shape) == 3
        if squeeze_batch:
            x_data = [self.data]
            n, channels, height, width = [1] + original_shape
        elif len(original_shape) == 4:
            x_data = self.data
            n, channels, height, width = original_shape
        else:
            raise RuntimeErrorUray("max_pool2d ожидает вход CHW или NCHW")
        out_h = (height - kernel_size) // stride + 1
        out_w = (width - kernel_size) // stride + 1
        if out_h <= 0 or out_w <= 0:
            raise RuntimeErrorUray("max_pool2d: окно больше входного изображения")
        np = get_numpy()
        if np is not None:
            array = np.asarray(x_data, dtype=float)
            windows = np.lib.stride_tricks.sliding_window_view(
                array, (kernel_size, kernel_size), axis=(2, 3)
            )[:, :, ::stride, ::stride, :, :]
            flat = windows.reshape(n, channels, out_h, out_w, -1)
            indices = np.argmax(flat, axis=-1)
            output = np.max(flat, axis=-1)
            output_data = output.tolist()

            def backward(grad):
                grad_array = np.asarray(grad, dtype=float)
                grad_input = np.zeros_like(array)
                ni, ci, hi, wi = np.indices(indices.shape)
                rows = hi * stride + indices // kernel_size
                cols = wi * stride + indices % kernel_size
                np.add.at(grad_input, (ni, ci, rows, cols), grad_array)
                return (grad_input[0].tolist() if squeeze_batch else grad_input.tolist(),)
        else:
            output_data = [
                [
                    [[0.0] * out_w for _ in range(out_h)]
                    for _ in range(channels)
                ]
                for _ in range(n)
            ]
            indices = [
                [
                    [[(0, 0)] * out_w for _ in range(out_h)]
                    for _ in range(channels)
                ]
                for _ in range(n)
            ]
            for batch in range(n):
                for channel in range(channels):
                    for oh in range(out_h):
                        for ow in range(out_w):
                            window = [
                                x_data[batch][channel][oh * stride + kh][ow * stride + kw]
                                for kh in range(kernel_size)
                                for kw in range(kernel_size)
                            ]
                            best = max(range(len(window)), key=window.__getitem__)
                            output_data[batch][channel][oh][ow] = window[best]
                            indices[batch][channel][oh][ow] = (
                                best // kernel_size, best % kernel_size
                            )

            def backward(grad):
                grad_input = _zeros_like(x_data)
                grad_batches = [grad] if squeeze_batch else grad
                for batch in range(n):
                    for channel in range(channels):
                        for oh in range(out_h):
                            for ow in range(out_w):
                                kh, kw = indices[batch][channel][oh][ow]
                                grad_input[batch][channel][oh * stride + kh][ow * stride + kw] += (
                                    grad_batches[batch][channel][oh][ow]
                                )
                return (grad_input[0] if squeeze_batch else grad_input,)
        if squeeze_batch:
            output_data = output_data[0]
        return self._result(output_data, (self,), backward)

    def batch_norm(
        self,
        weight,
        bias,
        running_mean,
        running_var,
        training=True,
        momentum=0.1,
        epsilon=1e-5,
        axis=-1,
    ):
        """Normalize each feature over every dimension except ``axis``."""
        weight = as_grad_tensor(weight)
        bias = as_grad_tensor(bias)
        shape = self.shape()
        if not shape or not isinstance(axis, int) or isinstance(axis, bool):
            raise RuntimeErrorUray("batch_norm ожидает тензор и целую ось признаков")
        if axis < 0:
            axis += len(shape)
        if axis < 0 or axis >= len(shape):
            raise RuntimeErrorUray("batch_norm: ось вне диапазона")
        feature_count = shape[axis]
        if (
            weight.shape() != [feature_count]
            or bias.shape() != [feature_count]
            or not isinstance(running_mean, list)
            or not isinstance(running_var, list)
            or len(running_mean) != feature_count
            or len(running_var) != feature_count
        ):
            raise RuntimeErrorUray(
                "batch_norm: размерность параметров должна совпадать с числом признаков"
            )
        if not isinstance(training, bool):
            raise RuntimeErrorUray("batch_norm: training должен быть bool")
        if not isinstance(momentum, (int, float)) or not 0.0 <= momentum <= 1.0:
            raise RuntimeErrorUray("batch_norm: momentum должен быть в диапазоне [0, 1]")
        if not isinstance(epsilon, (int, float)) or epsilon <= 0:
            raise RuntimeErrorUray("batch_norm: epsilon должен быть > 0")

        np = get_numpy()
        if np is not None:
            array = np.asarray(self.data, dtype=float)
            reduction_axes = tuple(
                index for index in range(len(shape)) if index != axis
            )
            if training:
                mean = array.mean(axis=reduction_axes)
                variance = array.var(axis=reduction_axes)
                for index in range(feature_count):
                    running_mean[index] = (
                        (1.0 - momentum) * running_mean[index]
                        + momentum * float(mean[index])
                    )
                    running_var[index] = (
                        (1.0 - momentum) * running_var[index]
                        + momentum * float(variance[index])
                    )
            else:
                mean = np.asarray(running_mean, dtype=float)
                variance = np.asarray(running_var, dtype=float)
            broadcast_shape = [1] * len(shape)
            broadcast_shape[axis] = feature_count
            mean_view = mean.reshape(broadcast_shape)
            inverse_std = 1.0 / np.sqrt(
                variance.reshape(broadcast_shape) + float(epsilon)
            )
            normalized = (array - mean_view) * inverse_std
            output = (
                normalized * np.asarray(weight.data, dtype=float).reshape(broadcast_shape)
                + np.asarray(bias.data, dtype=float).reshape(broadcast_shape)
            )
            reduction_count = math.prod(
                dimension for index, dimension in enumerate(shape) if index != axis
            )

            def backward(gradient):
                grad = np.asarray(gradient, dtype=float)
                grad_weight = np.sum(grad * normalized, axis=reduction_axes)
                grad_bias = np.sum(grad, axis=reduction_axes)
                scaled_weight = np.asarray(weight.data, dtype=float).reshape(
                    broadcast_shape
                )
                if training:
                    grad_sum = np.sum(grad, axis=reduction_axes).reshape(
                        broadcast_shape
                    )
                    normalized_grad_sum = np.sum(
                        grad * normalized, axis=reduction_axes
                    ).reshape(broadcast_shape)
                    grad_input = (
                        scaled_weight * inverse_std / reduction_count
                        * (
                            reduction_count * grad
                            - grad_sum
                            - normalized * normalized_grad_sum
                        )
                    )
                else:
                    grad_input = scaled_weight * inverse_std * grad
                return (
                    _array_to_data(grad_input),
                    _array_to_data(grad_weight),
                    _array_to_data(grad_bias),
                )

            return self._result(
                _array_to_data(output), (self, weight, bias), backward
            )

        flat = _flatten(self.data)
        inner_size = math.prod(shape[axis + 1:]) if axis + 1 < len(shape) else 1
        groups = [[] for _ in range(feature_count)]
        for index, value in enumerate(flat):
            feature = (index // inner_size) % feature_count
            groups[feature].append(float(value))

        if training:
            means = [sum(group) / len(group) for group in groups]
            variances = [
                sum((value - means[i]) ** 2 for value in group) / len(group)
                for i, group in enumerate(groups)
            ]
            for index in range(feature_count):
                running_mean[index] = (
                    (1.0 - momentum) * running_mean[index]
                    + momentum * means[index]
                )
                running_var[index] = (
                    (1.0 - momentum) * running_var[index]
                    + momentum * variances[index]
                )
        else:
            means = [float(value) for value in running_mean]
            variances = [float(value) for value in running_var]

        inverse_std = [
            1.0 / math.sqrt(variance + epsilon) for variance in variances
        ]
        normalized_groups = [
            [
                (value - means[index]) * inverse_std[index]
                for value in group
            ]
            for index, group in enumerate(groups)
        ]
        output_groups = [
            [
                normalized * weight.data[feature] + bias.data[feature]
                for normalized in normalized_groups[feature]
            ]
            for feature in range(feature_count)
        ]
        offsets = [0] * feature_count
        output_flat = []
        for index in range(len(flat)):
            feature = (index // inner_size) % feature_count
            output_flat.append(output_groups[feature][offsets[feature]])
            offsets[feature] += 1
        output_data = _from_flat(output_flat, shape)
        parents = (self, weight, bias)

        def backward(gradient):
            grad_flat = _flatten(gradient)
            grad_groups = [[] for _ in range(feature_count)]
            offsets = [0] * feature_count
            for index, value in enumerate(grad_flat):
                feature = (index // inner_size) % feature_count
                grad_groups[feature].append(float(value))
                offsets[feature] += 1

            input_grad_groups = []
            weight_grad = []
            bias_grad = []
            for feature, grad_group in enumerate(grad_groups):
                normalized = normalized_groups[feature]
                grad_sum = sum(grad_group)
                normalized_grad_sum = sum(
                    grad * value for grad, value in zip(grad_group, normalized)
                )
                weight_grad.append(normalized_grad_sum)
                bias_grad.append(grad_sum)
                if training:
                    count = len(grad_group)
                    input_grad_groups.append([
                        weight.data[feature] * inverse_std[feature] / count
                        * (
                            count * grad
                            - grad_sum
                            - value * normalized_grad_sum
                        )
                        for grad, value in zip(grad_group, normalized)
                    ])
                else:
                    input_grad_groups.append([
                        grad * weight.data[feature] * inverse_std[feature]
                        for grad in grad_group
                    ])

            offsets = [0] * feature_count
            input_grad_flat = []
            for index in range(len(flat)):
                feature = (index // inner_size) % feature_count
                input_grad_flat.append(
                    input_grad_groups[feature][offsets[feature]]
                )
                offsets[feature] += 1
            return (
                _from_flat(input_grad_flat, shape),
                weight_grad,
                bias_grad,
            )

        return self._result(output_data, parents, backward)

    def layer_norm(self, weight, bias, epsilon=1e-5):
        """Normalize the last feature axis independently for every token."""
        weight = as_grad_tensor(weight)
        bias = as_grad_tensor(bias)
        shape = self.shape()
        if not shape or shape[-1] <= 0:
            raise RuntimeErrorUray("layer_norm ожидает непустую ось признаков")
        features = shape[-1]
        if weight.shape() != [features] or bias.shape() != [features]:
            raise RuntimeErrorUray(
                "layer_norm: размерность параметров не совпадает с последней осью"
            )
        if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or epsilon <= 0:
            raise RuntimeErrorUray("layer_norm: epsilon должен быть > 0")

        np = get_numpy()
        if np is not None:
            array = np.asarray(self.data, dtype=float)
            mean = array.mean(axis=-1, keepdims=True)
            variance = array.var(axis=-1, keepdims=True)
            inverse_std = 1.0 / np.sqrt(variance + float(epsilon))
            normalized = (array - mean) * inverse_std
            output = (
                normalized * np.asarray(weight.data, dtype=float)
                + np.asarray(bias.data, dtype=float)
            )

            def backward(gradient):
                grad = np.asarray(gradient, dtype=float)
                grad_weight = np.sum(grad * normalized, axis=tuple(
                    range(grad.ndim - 1)
                ))
                grad_bias = np.sum(grad, axis=tuple(range(grad.ndim - 1)))
                grad_sum = grad.sum(axis=-1, keepdims=True)
                normalized_grad_sum = (grad * normalized).sum(
                    axis=-1, keepdims=True
                )
                grad_input = (
                    np.asarray(weight.data, dtype=float) * inverse_std / features
                    * (
                        features * grad
                        - grad_sum
                        - normalized * normalized_grad_sum
                    )
                )
                return (
                    _array_to_data(grad_input),
                    _array_to_data(grad_weight),
                    _array_to_data(grad_bias),
                )

            return self._result(
                _array_to_data(output), (self, weight, bias), backward
            )

        flat = _flatten(self.data)
        grad_groups = None
        groups = [flat[index:index + features] for index in range(0, len(flat), features)]
        means = [sum(group) / features for group in groups]
        variances = [
            sum((value - means[index]) ** 2 for value in group) / features
            for index, group in enumerate(groups)
        ]
        inverse_std = [
            1.0 / math.sqrt(variance + float(epsilon))
            for variance in variances
        ]
        normalized_groups = [
            [
                (value - means[index]) * inverse_std[index]
                for value in group
            ]
            for index, group in enumerate(groups)
        ]
        output_groups = [
            [
                normalized_groups[group_index][feature] * weight.data[feature]
                + bias.data[feature]
                for feature in range(features)
            ]
            for group_index in range(len(groups))
        ]
        output_data = _from_flat(
            [value for group in output_groups for value in group], shape
        )

        def backward(gradient):
            grad_groups = [
                _flatten(gradient)[index:index + features]
                for index in range(0, len(flat), features)
            ]
            input_groups = []
            weight_grad = [0.0] * features
            bias_grad = [0.0] * features
            for group_index, group_grad in enumerate(grad_groups):
                normalized = normalized_groups[group_index]
                grad_sum = sum(group_grad)
                norm_grad_sum = sum(
                    grad * value for grad, value in zip(group_grad, normalized)
                )
                input_groups.append([
                    weight.data[feature] * inverse_std[group_index] / features
                    * (
                        features * group_grad[feature]
                        - grad_sum
                        - normalized[feature] * norm_grad_sum
                    )
                    for feature in range(features)
                ])
                for feature in range(features):
                    weight_grad[feature] += (
                        group_grad[feature] * normalized[feature]
                    )
                    bias_grad[feature] += group_grad[feature]
            return (
                _from_flat(
                    [value for group in input_groups for value in group], shape
                ),
                weight_grad,
                bias_grad,
            )

        return self._result(
            output_data, (self, weight, bias), backward
        )

    def multihead_attention(
        self,
        weights,
        biases,
        heads=1,
        causal=False,
    ):
        """Scaled dot-product self-attention for [sequence, features] inputs."""
        if len(self.shape()) not in (2, 3):
            raise RuntimeErrorUray(
                "multihead_attention ожидает [sequence, features] "
                "или [batch, sequence, features]"
            )
        if not isinstance(heads, int) or isinstance(heads, bool) or heads <= 0:
            raise RuntimeErrorUray("multihead_attention: heads должен быть > 0")
        if not isinstance(causal, bool):
            raise RuntimeErrorUray("multihead_attention: causal должен быть bool")
        if len(weights) != 4 or len(biases) != 4:
            raise RuntimeErrorUray("multihead_attention ожидает 4 матрицы и 4 bias")
        weights = [as_grad_tensor(value) for value in weights]
        biases = [as_grad_tensor(value) for value in biases]
        original_shape = self.shape()
        squeeze_batch = len(original_shape) == 2
        x_data = [self.data] if squeeze_batch else self.data
        batch_count, sequence, features = (
            [1] + original_shape if squeeze_batch else original_shape
        )
        if sequence <= 0 or features <= 0 or features % heads:
            raise RuntimeErrorUray(
                "multihead_attention: features должны делиться на heads"
            )
        for weight, bias in zip(weights, biases):
            if weight.shape() != [features, features] or bias.shape() != [features]:
                raise RuntimeErrorUray(
                    "multihead_attention: проекции должны иметь форму "
                    "[features, features] и [features]"
                )
        depth = features // heads
        scale = 1.0 / math.sqrt(depth)
        parents = (self, *weights, *biases)
        np = get_numpy()

        if np is not None:
            x_array = np.asarray(x_data, dtype=float)
            weight_arrays = [
                np.asarray(value.data, dtype=float) for value in weights
            ]
            bias_arrays = [
                np.asarray(value.data, dtype=float) for value in biases
            ]
            projected = [
                x_array @ weight_arrays[index] + bias_arrays[index]
                for index in range(3)
            ]
            q, k, v = [
                value.reshape(batch_count, sequence, heads, depth).transpose(
                    0, 2, 1, 3
                )
                for value in projected
            ]
            scores = (q @ k.transpose(0, 1, 3, 2)) * scale
            future_mask = None
            if causal:
                future_mask = np.triu(
                    np.ones((sequence, sequence), dtype=bool), k=1
                )
                scores = np.where(future_mask[None, None, :, :], -1e30, scores)
            scores -= scores.max(axis=-1, keepdims=True)
            probabilities = np.exp(scores)
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            context = probabilities @ v
            merged = context.transpose(0, 2, 1, 3).reshape(
                batch_count, sequence, features
            )
            output = merged @ weight_arrays[3] + bias_arrays[3]

            def backward(gradient):
                grad = np.asarray(
                    [gradient] if squeeze_batch else gradient, dtype=float
                )
                grad_weights = [None] * 4
                grad_biases = [None] * 4
                grad_weights[3] = np.einsum("bse,bsd->ed", merged, grad)
                grad_biases[3] = grad.sum(axis=(0, 1))
                grad_merged = grad @ weight_arrays[3].T
                grad_context = grad_merged.reshape(
                    batch_count, sequence, heads, depth
                ).transpose(0, 2, 1, 3)
                grad_q = np.zeros_like(q)
                grad_k = np.zeros_like(k)
                grad_v = np.zeros_like(v)
                for head in range(heads):
                    probability = probabilities[:, head]
                    grad_head = grad_context[:, head]
                    value_head = v[:, head]
                    query_head = q[:, head]
                    key_head = k[:, head]
                    grad_v[:, head] = probability.transpose(0, 2, 1) @ grad_head
                    grad_probability = grad_head @ value_head.transpose(0, 2, 1)
                    grad_score = probability * (
                        grad_probability
                        - (grad_probability * probability).sum(
                            axis=-1, keepdims=True
                        )
                    )
                    if future_mask is not None:
                        grad_score = np.where(
                            future_mask[None, :, :], 0.0, grad_score
                        )
                    grad_q[:, head] = (grad_score @ key_head) * scale
                    grad_k[:, head] = (
                        grad_score.transpose(0, 2, 1) @ query_head
                    ) * scale
                grad_projected = [
                    value.transpose(0, 2, 1, 3).reshape(
                        batch_count, sequence, features
                    )
                    for value in (grad_q, grad_k, grad_v)
                ]
                grad_input = np.zeros_like(x_array)
                for index in range(3):
                    grad_weights[index] = np.einsum(
                        "bse,bsd->ed", x_array, grad_projected[index]
                    )
                    grad_biases[index] = grad_projected[index].sum(axis=(0, 1))
                    grad_input += (
                        grad_projected[index] @ weight_arrays[index].T
                    )
                result = [
                    _array_to_data(grad_input[0] if squeeze_batch else grad_input)
                ]
                result.extend(_array_to_data(value) for value in grad_weights)
                result.extend(_array_to_data(value) for value in grad_biases)
                return tuple(result)

            return self._result(
                _array_to_data(output[0] if squeeze_batch else output),
                parents,
                backward,
            )

        projected_batches = []
        caches = []
        for sample in x_data:
            projected = [
                _broadcast_binary(_matmul(sample, weights[index].data),
                                  biases[index].data, lambda a, b: a + b)
                for index in range(3)
            ]
            query_heads = [
                [row[head * depth:(head + 1) * depth] for row in projected[0]]
                for head in range(heads)
            ]
            key_heads = [
                [row[head * depth:(head + 1) * depth] for row in projected[1]]
                for head in range(heads)
            ]
            value_heads = [
                [row[head * depth:(head + 1) * depth] for row in projected[2]]
                for head in range(heads)
            ]
            probabilities = []
            contexts = []
            for head in range(heads):
                score = _scale_data(
                    _matmul(query_heads[head], _transpose(key_heads[head])),
                    scale,
                )
                if causal:
                    for row in range(sequence):
                        for column in range(row + 1, sequence):
                            score[row][column] = -1e30
                probability = _softmax_rows(score)
                context = _matmul(probability, value_heads[head])
                probabilities.append(probability)
                contexts.append(context)
            merged = [
                [
                    contexts[head][position][feature]
                    for head in range(heads)
                    for feature in range(depth)
                ]
                for position in range(sequence)
            ]
            output = _broadcast_binary(
                _matmul(merged, weights[3].data),
                biases[3].data,
                lambda a, b: a + b,
            )
            projected_batches.append(output)
            caches.append((
                sample, query_heads, key_heads, value_heads, probabilities, merged
            ))

        def backward(gradient):
            grad_batches = [gradient] if squeeze_batch else gradient
            grad_input_batches = []
            grad_weights = [_zeros_like(value.data) for value in weights]
            grad_biases = [_zeros_like(value.data) for value in biases]
            for batch_index, grad_output in enumerate(grad_batches):
                sample, query_heads, key_heads, value_heads, probabilities, merged = (
                    caches[batch_index]
                )
                grad_weights[3] = _add_data(
                    grad_weights[3],
                    _matmul(_transpose(merged), grad_output),
                )
                grad_biases[3] = _add_data(
                    grad_biases[3], _sum_axis0(grad_output)
                )
                grad_merged = _matmul(
                    grad_output, _transpose(weights[3].data)
                )
                grad_query_heads, grad_key_heads, grad_value_heads = [], [], []
                for head in range(heads):
                    start = head * depth
                    end = start + depth
                    grad_context = [
                        row[start:end] for row in grad_merged
                    ]
                    probability = probabilities[head]
                    value_head = value_heads[head]
                    query_head = query_heads[head]
                    key_head = key_heads[head]
                    grad_value_heads.append(
                        _matmul(_transpose(probability), grad_context)
                    )
                    grad_probability = _matmul(
                        grad_context, _transpose(value_head)
                    )
                    grad_score = []
                    for row in range(sequence):
                        dot = sum(
                            grad_probability[row][column]
                            * probability[row][column]
                            for column in range(sequence)
                        )
                        grad_score.append([
                            0.0 if causal and column > row else
                            probability[row][column]
                            * (grad_probability[row][column] - dot)
                            for column in range(sequence)
                        ])
                    grad_query_heads.append(_scale_data(
                        _matmul(grad_score, key_head), scale
                    ))
                    grad_key_heads.append(_scale_data(
                        _matmul(_transpose(grad_score), query_head), scale
                    ))
                grad_projected = []
                for head_groups in (
                    grad_query_heads, grad_key_heads, grad_value_heads
                ):
                    grad_projected.append([
                        [
                            head_groups[head][position][feature]
                            for head in range(heads)
                            for feature in range(depth)
                        ]
                        for position in range(sequence)
                    ])
                grad_input = _zeros_like(sample)
                for index in range(3):
                    grad_weights[index] = _add_data(
                        grad_weights[index],
                        _matmul(
                            _transpose(sample), grad_projected[index]
                        ),
                    )
                    grad_biases[index] = _add_data(
                        grad_biases[index], _sum_axis0(grad_projected[index])
                    )
                    grad_input = _add_data(
                        grad_input,
                        _matmul(
                            grad_projected[index],
                            _transpose(weights[index].data),
                        ),
                    )
                grad_input_batches.append(grad_input)
            result = [grad_input_batches[0] if squeeze_batch else grad_input_batches]
            result.extend(grad_weights)
            result.extend(grad_biases)
            return tuple(result)

        output_data = projected_batches[0] if squeeze_batch else projected_batches
        return self._result(output_data, parents, backward)

    def backward(self, gradient=None):
        if not self.requires_grad:
            raise RuntimeErrorUray("Нельзя вызвать backward у тензора без градиента")
        if self._graph_consumed:
            raise RuntimeErrorUray(
                "Граф уже использован; выполните forward заново перед backward"
            )
        if gradient is None:
            if self.shape():
                raise RuntimeErrorUray(
                    "Для backward у нескалярного тензора нужно передать начальный градиент"
                )
            gradient = 1.0
        elif isinstance(gradient, (Tensor, GradTensor)):
            gradient = gradient.data
        elif not isinstance(gradient, (int, float, list)):
            raise RuntimeErrorUray("Начальный градиент должен быть числом или тензором")
        if _shape(gradient) != self.shape():
            raise RuntimeErrorUray(
                f"Форма начального градиента {_shape(gradient)} не совпадает с {self.shape()}"
            )

        topology = []
        visited = set()

        def visit(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._parents:
                if parent.requires_grad:
                    visit(parent)
            topology.append(node)

        visit(self)
        gradients = {id(self): gradient}
        for node in reversed(topology):
            node_gradient = gradients.get(id(node))
            if node_gradient is None or node._backward_fn is None:
                continue
            parent_gradients = node._backward_fn(node_gradient)
            for parent, parent_gradient in zip(node._parents, parent_gradients):
                if not parent.requires_grad or parent_gradient is None:
                    continue
                key = id(parent)
                gradients[key] = (
                    parent_gradient
                    if key not in gradients
                    else _add_data(gradients[key], parent_gradient)
                )
        for node in topology:
            node.grad = gradients.get(id(node))
            if node._parents:
                node._graph_consumed = True
                node._parents = ()
                node._backward_fn = None
        return self

    def zero_grad(self):
        self.grad = None

    def apply_sgd(self, learning_rate):
        if not self.requires_grad:
            raise RuntimeErrorUray("SGD ожидает параметр с requires_grad=true")
        if self.grad is None:
            return self
        self._data = _zip_map(
            self.data,
            self.grad,
            lambda value, grad: value - learning_rate * grad,
        )
        return self

    def apply_adam(
        self, learning_rate, state, beta1=0.9, beta2=0.999, epsilon=1e-8
    ):
        if not self.requires_grad:
            raise RuntimeErrorUray("Adam ожидает параметр с requires_grad=true")
        if self.grad is None:
            return self
        state["step"] = state.get("step", 0) + 1
        if "first" not in state:
            state["first"] = _zeros_like(self.data)
            state["second"] = _zeros_like(self.data)
        step = state["step"]
        first, second = state["first"], state["second"]

        def update(value, grad, moment1, moment2):
            if isinstance(value, list):
                return [
                    update(v, g, m1, m2)
                    for v, g, m1, m2 in zip(value, grad, moment1, moment2)
                ]
            m1 = beta1 * moment1 + (1.0 - beta1) * grad
            m2 = beta2 * moment2 + (1.0 - beta2) * grad * grad
            corrected1 = m1 / (1.0 - beta1 ** step)
            corrected2 = m2 / (1.0 - beta2 ** step)
            return (
                value - learning_rate * corrected1 / (math.sqrt(corrected2) + epsilon),
                m1,
                m2,
            )

        def update_tree(value, grad, moment1, moment2):
            if isinstance(value, list):
                updated, next1, next2 = [], [], []
                for v, g, m1, m2 in zip(value, grad, moment1, moment2):
                    nv, nm1, nm2 = update_tree(v, g, m1, m2)
                    updated.append(nv)
                    next1.append(nm1)
                    next2.append(nm2)
                return updated, next1, next2
            return update(value, grad, moment1, moment2)

        self._data, state["first"], state["second"] = update_tree(
            self.data, self.grad, first, second
        )
        return self

    def item(self):
        if self.shape():
            raise RuntimeErrorUray("item ожидает скалярный тензор")
        return float(self.data)


def as_grad_tensor(value, requires_grad=False):
    if isinstance(value, GradTensor):
        return value
    if isinstance(value, Tensor):
        value = value.data
    return GradTensor(value, requires_grad=requires_grad)
