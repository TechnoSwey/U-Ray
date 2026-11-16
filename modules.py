# modules.py

import json
import os
import math as _pymath
import random as _pyrandom

import values
from errors import RuntimeErrorUray as R
from values import DictValue, Builtin, Module, Function
from tensor import Tensor, get_numpy
from autograd import GradTensor, as_grad_tensor
from mnist import MNISTView, load_mnist_directory


class ModuleRegistry:
    def __init__(self):
        self.modules = self._build()

    def get(self, name):
        return self.modules.get(name)

    def _build(self):
        return {
            "math": self._build_math(),
            "linalg": self._build_linalg(),
            "nn": self._build_nn(),
        }

    # ---------- math ----------
    def _build_math(self):
        def unary(name, fn):
            def impl(args):
                if len(args) != 1:
                    raise R(f"math.{name} ожидает 1 аргумент")
                x = args[0]
                if not isinstance(x, (int, float)) or isinstance(x, bool):
                    raise R(f"math.{name} ожидает число")
                try:
                    return float(fn(x))
                except (ValueError, OverflowError) as e:
                    raise R(f"math.{name}: {e}")
            return Builtin(name, impl)

        def rand(args):
            if args:
                raise R("math.rand не принимает аргументов")
            return _pyrandom.random()

        return Module("math", {
            "pi": _pymath.pi,
            "e": _pymath.e,
            "sqrt": unary("sqrt", _pymath.sqrt),
            "exp": unary("exp", _pymath.exp),
            "log": unary("log", _pymath.log),
            "abs": unary("abs", abs),
            "sin": unary("sin", _pymath.sin),
            "cos": unary("cos", _pymath.cos),
            "rand": Builtin("rand", rand),
        })

    # ---------- linalg ----------
    def _build_linalg(self):
        def ensure_flat(name, v):
            if not isinstance(v, list):
                raise R(f"linalg.{name} ожидает список")
            for x in v:
                if not isinstance(x, (int, float)) or isinstance(x, bool):
                    raise R(f"linalg.{name}: элементы должны быть числами")

        def ensure_matrix(name, m):
            if not isinstance(m, list) or not m:
                raise R(f"linalg.{name} ожидает непустую матрицу")
            cols = None
            for row in m:
                if not isinstance(row, list):
                    raise R(f"linalg.{name}: строки должны быть списками")
                if cols is None:
                    cols = len(row)
                elif len(row) != cols:
                    raise R(f"linalg.{name}: строки разной длины")
                for x in row:
                    if not isinstance(x, (int, float)) or isinstance(x, bool):
                        raise R(f"linalg.{name}: элементы должны быть числами")

        def dot(args):
            if len(args) != 2:
                raise R("linalg.dot ожидает 2 аргумента")
            a, b = args
            ensure_flat("dot", a)
            ensure_flat("dot", b)
            if len(a) != len(b):
                raise R(f"linalg.dot: длины не совпадают ({len(a)} и {len(b)})")
            return sum(x * y for x, y in zip(a, b))

        def matmul(args):
            if len(args) != 2:
                raise R("linalg.matmul ожидает 2 аргумента")
            A, B = args
            ensure_matrix("matmul", A)
            ensure_matrix("matmul", B)
            if len(A[0]) != len(B):
                raise R(f"linalg.matmul: размерности несовместимы ({len(A[0])} != {len(B)})")
            tA = Tensor(A)
            tB = Tensor(B)
            res = tA.matmul(tB)
            return res.to_list()

        def transpose(args):
            if len(args) != 1:
                raise R("linalg.transpose ожидает 1 аргумент")
            A = args[0]
            ensure_matrix("transpose", A)
            return Tensor(A).T().to_list()

        def zeros(args):
            if len(args) != 1:
                raise R("linalg.zeros ожидает 1 аргумент")
            n = args[0]
            if not isinstance(n, int) or isinstance(n, bool) or n < 0:
                raise R("linalg.zeros ожидает неотрицательное целое")
            return [0.0] * n

        def ones(args):
            if len(args) != 1:
                raise R("linalg.ones ожидает 1 аргумент")
            n = args[0]
            if not isinstance(n, int) or isinstance(n, bool) or n < 0:
                raise R("linalg.ones ожидает неотрицательное целое")
            return [1.0] * n

        def lsum(args):
            if len(args) != 1:
                raise R("linalg.sum ожидает 1 аргумент")
            a = args[0]
            ensure_flat("sum", a)
            return sum(a)

        def lmax(args):
            if len(args) != 1:
                raise R("linalg.max ожидает 1 аргумент")
            a = args[0]
            ensure_flat("max", a)
            if not a:
                raise R("linalg.max от пустого списка")
            return max(a)

        def lmin(args):
            if len(args) != 1:
                raise R("linalg.min ожидает 1 аргумент")
            a = args[0]
            ensure_flat("min", a)
            if not a:
                raise R("linalg.min от пустого списка")
            return min(a)

        return Module("linalg", {
            "dot": Builtin("dot", dot),
            "matmul": Builtin("matmul", matmul),
            "transpose": Builtin("transpose", transpose),
            "zeros": Builtin("zeros", zeros),
            "ones": Builtin("ones", ones),
            "sum": Builtin("sum", lsum),
            "max": Builtin("max", lmax),
            "min": Builtin("min", lmin),
        })

    # ---------- nn ----------
    def _build_nn(self):
        LEAKY_SLOPE = 0.1

        def ensure_number(name, x):
            if not isinstance(x, (int, float)) or isinstance(x, bool):
                raise R(f"nn.{name} ожидает число")

        def ensure_vector(name, v):
            if not isinstance(v, list):
                raise R(f"nn.{name} ожидает список")
            for x in v:
                if not isinstance(x, (int, float)) or isinstance(x, bool):
                    raise R(f"nn.{name}: элементы должны быть числами")

        def as_vector(name, value):
            vector = value.to_list() if isinstance(value, Tensor) else value
            ensure_vector(name, vector)
            return vector

        def grad_tensor(args):
            if len(args) not in (1, 2):
                raise R("nn.tensor ожидает data и необязательный requires_grad")
            requires_grad = args[1] if len(args) == 2 else False
            if not isinstance(requires_grad, bool):
                raise R("nn.tensor: requires_grad должен быть bool")
            try:
                return as_grad_tensor(args[0], requires_grad=requires_grad)
            except (TypeError, ValueError, OverflowError) as e:
                raise R(f"nn.tensor: некорректные данные: {e}")

        def parameter(args):
            if len(args) != 1:
                raise R("nn.parameter ожидает данные")
            try:
                return as_grad_tensor(args[0], requires_grad=True)
            except (TypeError, ValueError, OverflowError) as e:
                raise R(f"nn.parameter: некорректные данные: {e}")

        def _grad_operand(name, value):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return as_grad_tensor(float(value))
            if isinstance(value, (Tensor, list)):
                try:
                    return as_grad_tensor(value)
                except (TypeError, ValueError, OverflowError) as e:
                    raise R(f"nn.{name}: некорректный тензор: {e}")
            raise R(f"nn.{name}: ожидался тензор или число")

        def grad_add(args):
            if len(args) != 2: raise R("nn.add ожидает 2 аргумента")
            return _grad_operand("add", args[0]).add(args[1])

        def grad_sub(args):
            if len(args) != 2: raise R("nn.sub ожидает 2 аргумента")
            return _grad_operand("sub", args[0]).sub(args[1])

        def grad_mul(args):
            if len(args) != 2: raise R("nn.multiply ожидает 2 аргумента")
            return _grad_operand("multiply", args[0]).mul(args[1])

        def grad_div(args):
            if len(args) != 2: raise R("nn.divide ожидает 2 аргумента")
            return _grad_operand("divide", args[0]).div(args[1])

        def grad_matmul(args):
            if len(args) != 2: raise R("nn.matmul ожидает 2 аргумента")
            return _grad_operand("matmul", args[0]).matmul(args[1])

        def grad_reshape(args):
            if len(args) != 2: raise R("nn.reshape ожидает tensor и shape")
            if not isinstance(args[1], list):
                raise R("nn.reshape: shape должен быть списком")
            return _grad_operand("reshape", args[0]).reshape(args[1])

        def grad_sum(args):
            if len(args) != 1: raise R("nn.reduce_sum ожидает 1 аргумент")
            return _grad_operand("reduce_sum", args[0]).sum_all()

        def grad_mean(args):
            if len(args) != 1: raise R("nn.mean ожидает 1 аргумент")
            return _grad_operand("mean", args[0]).mean()

        def grad_cross_entropy(args):
            if len(args) != 2:
                raise R("nn.cross_entropy ожидает logits и labels")
            return _grad_operand("cross_entropy", args[0]).cross_entropy(args[1])

        def grad_binary_cross_entropy(args):
            if len(args) != 2:
                raise R("nn.binary_cross_entropy ожидает logits и labels")
            return _grad_operand("binary_cross_entropy", args[0]).binary_cross_entropy(args[1])

        def grad_backward(args):
            if len(args) not in (1, 2):
                raise R("nn.backward ожидает loss и необязательный градиент")
            value = args[0]
            if not isinstance(value, GradTensor):
                raise R("nn.backward ожидает тензор с графом градиентов")
            value.backward(args[1] if len(args) == 2 else None)
            return 0

        def grad_value(args):
            if len(args) != 1 or not isinstance(args[0], GradTensor):
                raise R("nn.grad ожидает параметр GradTensor")
            return Tensor(args[0].grad) if args[0].grad is not None else None

        def grad_item(args):
            if len(args) != 1 or not isinstance(args[0], GradTensor):
                raise R("nn.item ожидает скалярный GradTensor")
            return args[0].item()

        def _collect_parameters(value, result=None, seen=None):
            if result is None:
                result, seen = [], set()
            if isinstance(value, GradTensor):
                if value.requires_grad and id(value) not in seen:
                    seen.add(id(value))
                    result.append(value)
            elif isinstance(value, DictValue):
                for _, child in value.items():
                    _collect_parameters(child, result, seen)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    _collect_parameters(child, result, seen)
            return result

        def grad_parameters(args):
            if len(args) != 1:
                raise R("nn.parameters ожидает модель или список параметров")
            return _collect_parameters(args[0])

        def grad_zero(args):
            if len(args) != 1:
                raise R("nn.zero_grad ожидает модель или список параметров")
            for param in _collect_parameters(args[0]):
                param.zero_grad()
            return 0

        def grad_sgd(args):
            if len(args) != 2:
                raise R("nn.sgd_update ожидает параметры и learning_rate")
            params, learning_rate = args
            ensure_number("sgd_update", learning_rate)
            if learning_rate <= 0:
                raise R("nn.sgd_update: learning_rate должен быть > 0")
            for param in _collect_parameters(params):
                param.apply_sgd(float(learning_rate))
            return params

        def grad_conv2d(args):
            if len(args) < 2 or len(args) > 5:
                raise R("nn.conv2d ожидает input, weight, bias?, stride?, padding?")
            source = _grad_operand("conv2d", args[0])
            bias = args[2] if len(args) >= 3 else None
            stride = args[3] if len(args) >= 4 else 1
            padding = args[4] if len(args) >= 5 else 0
            return source.conv2d(args[1], bias, stride, padding)

        def grad_max_pool2d(args):
            if len(args) not in (2, 3):
                raise R("nn.max_pool2d ожидает input, kernel_size и stride?")
            stride = args[2] if len(args) == 3 else None
            return _grad_operand("max_pool2d", args[0]).max_pool2d(args[1], stride)

        def _positive_int(name, value):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise R(f"nn.{name}: значение должно быть положительным целым")

        def linear_layer(args):
            if len(args) != 2:
                raise R("nn.linear_layer ожидает input_size и output_size")
            in_dim, out_dim = args
            _positive_int("linear_layer", in_dim)
            _positive_int("linear_layer", out_dim)
            initialized = _init_layer_weights(in_dim, out_dim, "relu")
            layer = DictValue()
            layer["kind"] = "linear"
            layer["W"] = as_grad_tensor(initialized["W"], requires_grad=True)
            layer["b"] = as_grad_tensor(initialized["b"], requires_grad=True)
            return layer

        def conv2d_layer(args):
            if len(args) not in (3, 4, 5):
                raise R("nn.conv2d_layer ожидает in_channels, out_channels, kernel, stride?, padding?")
            in_channels, out_channels, kernel = args[:3]
            _positive_int("conv2d_layer", in_channels)
            _positive_int("conv2d_layer", out_channels)
            if isinstance(kernel, int) and not isinstance(kernel, bool):
                kernel_h = kernel_w = kernel
            elif isinstance(kernel, list) and len(kernel) == 2:
                kernel_h, kernel_w = kernel
            else:
                raise R("nn.conv2d_layer: kernel должен быть числом или [height, width]")
            _positive_int("conv2d_layer", kernel_h)
            _positive_int("conv2d_layer", kernel_w)
            stride = args[3] if len(args) >= 4 else 1
            padding = args[4] if len(args) >= 5 else 0
            _positive_int("conv2d_layer stride", stride)
            if not isinstance(padding, int) or isinstance(padding, bool) or padding < 0:
                raise R("nn.conv2d_layer: padding должен быть целым >= 0")
            fan_in = in_channels * kernel_h * kernel_w
            limit = _pymath.sqrt(6.0 / fan_in)
            weights = [
                [
                    [
                        [(_pyrandom.random() * 2.0 - 1.0) * limit for _ in range(kernel_w)]
                        for _ in range(kernel_h)
                    ]
                    for _ in range(in_channels)
                ]
                for _ in range(out_channels)
            ]
            layer = DictValue()
            layer["kind"] = "conv2d"
            layer["W"] = as_grad_tensor(weights, requires_grad=True)
            layer["b"] = as_grad_tensor([0.0] * out_channels, requires_grad=True)
            layer["stride"] = stride
            layer["padding"] = padding
            return layer

        def activation_layer(args):
            if len(args) != 1 or args[0] not in ("relu", "sigmoid", "tanh", "softmax"):
                raise R("nn.activation_layer ожидает relu, sigmoid, tanh или softmax")
            layer = DictValue()
            layer["kind"] = "activation"
            layer["name"] = args[0]
            return layer

        def pool2d_layer(args):
            if len(args) not in (1, 2):
                raise R("nn.max_pool2d_layer ожидает kernel_size и stride?")
            kernel = args[0]
            stride = args[1] if len(args) == 2 else kernel
            _positive_int("max_pool2d_layer", kernel)
            _positive_int("max_pool2d_layer", stride)
            layer = DictValue()
            layer["kind"] = "max_pool2d"
            layer["kernel"] = kernel
            layer["stride"] = stride
            return layer

        def flatten_layer(args):
            if args:
                raise R("nn.flatten_layer не ожидает аргументов")
            layer = DictValue()
            layer["kind"] = "flatten"
            return layer

        def sequential(args):
            if len(args) != 1 or not isinstance(args[0], list):
                raise R("nn.sequential ожидает список слоёв")
            model = DictValue()
            model["kind"] = "sequential"
            model["layers"] = args[0]
            return model

        def forward_layers(args):
            if len(args) != 2:
                raise R("nn.forward_layers ожидает модель и вход")
            model, source = args
            layers = model["layers"] if isinstance(model, DictValue) else model
            if not isinstance(layers, list):
                raise R("nn.forward_layers ожидает список слоёв или sequential-модель")
            output = _grad_operand("forward_layers", source)
            for layer in layers:
                if not isinstance(layer, DictValue) or "kind" not in layer:
                    raise R("nn.forward_layers: каждый элемент должен быть слоем")
                kind = layer["kind"]
                if kind == "linear":
                    weights = layer["W"]
                    if len(output.shape()) == 1:
                        output = weights.matmul(output)
                    elif len(output.shape()) == 2:
                        output = output.matmul(weights.transpose())
                    else:
                        raise R("linear слой ожидает вектор или матрицу [batch, features]")
                    output = output.add(layer["b"])
                elif kind == "conv2d":
                    output = output.conv2d(
                        layer["W"], layer["b"], layer["stride"], layer["padding"]
                    )
                elif kind == "activation":
                    name = layer["name"]
                    output = {
                        "relu": output.relu,
                        "sigmoid": output.sigmoid,
                        "tanh": output.tanh,
                        "softmax": output.softmax,
                    }[name]()
                elif kind == "max_pool2d":
                    output = output.max_pool2d(layer["kernel"], layer["stride"])
                elif kind == "flatten":
                    output = output.flatten()
                else:
                    raise R(f"nn.forward_layers: неизвестный тип слоя {kind!r}")
            return output

        def _autograd_batch(view, indices, name):
            if isinstance(view, MNISTView):
                numpy = get_numpy()
                if numpy is None and len(view) >= 10000:
                    raise R(
                        f"nn.{name}: для полного MNIST набора выберите "
                        "--backend=numpy"
                    )
                if numpy is not None:
                    return view.batch(indices, numpy).tolist()
                return [view.sample(index) for index in indices]
            if isinstance(view, Tensor):
                return [view.get(index) for index in indices]
            if isinstance(view, list):
                return [view[index] for index in indices]
            raise R(f"nn.{name}: X и Y должны быть списками, тензорами или MNIST view")

        def _autograd_loss(logits, target, loss_name):
            if loss_name == "cce":
                return logits.cross_entropy(target)
            if loss_name == "bce":
                return logits.binary_cross_entropy(target)
            if loss_name == "mse":
                return logits.mse_loss(target)
            raise R(f"nn: неизвестная функция потерь {loss_name!r}")

        def _prepare_graph_input(batch, input_shape):
            value = as_grad_tensor(batch)
            if input_shape is not None:
                if not isinstance(input_shape, list) or any(
                    not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
                    for dim in input_shape
                ):
                    raise R("input_shape должен содержать положительные целые размеры")
                value = value.reshape([len(batch)] + input_shape)
            return value

        def train_layers(args):
            if len(args) < 6 or len(args) > 10:
                raise R(
                    "nn.train_layers ожидает model, X, Y, epochs, lr, batch_size, "
                    "loss?, optimizer?, show_progress?, input_shape?"
                )
            model, X, Y, epochs, learning_rate, batch_size = args[:6]
            loss_name = args[6] if len(args) >= 7 else "cce"
            optimizer = args[7] if len(args) >= 8 else "adam"
            show_progress = args[8] if len(args) >= 9 else True
            input_shape = args[9] if len(args) >= 10 else None
            if (
                not isinstance(model, DictValue)
                or "kind" not in model
                or model["kind"] != "sequential"
            ):
                raise R("nn.train_layers ожидает модель nn.sequential")
            if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
                raise R("nn.train_layers: epochs должен быть положительным целым")
            if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
                raise R("nn.train_layers: batch_size должен быть положительным целым")
            ensure_number("train_layers", learning_rate)
            if learning_rate <= 0:
                raise R("nn.train_layers: lr должен быть > 0")
            if loss_name not in ("cce", "bce", "mse"):
                raise R("nn.train_layers: loss должен быть cce, bce или mse")
            if optimizer not in ("adam", "sgd"):
                raise R("nn.train_layers: optimizer должен быть adam или sgd")
            if not isinstance(show_progress, bool):
                raise R("nn.train_layers: show_progress должен быть bool")
            try:
                sample_count = len(X)
                if len(Y) != sample_count or sample_count == 0:
                    raise R("nn.train_layers: X и Y должны иметь одинаковую длину > 0")
            except TypeError as error:
                raise R("nn.train_layers: X и Y должны иметь длину") from error
            params = _collect_parameters(model)
            if not params:
                raise R("nn.train_layers: в модели нет обучаемых параметров")
            adam_states = {id(param): {} for param in params}
            history = []
            order = list(range(sample_count))
            batches_per_epoch = _pymath.ceil(sample_count / batch_size)
            total_steps = epochs * batches_per_epoch
            for epoch in range(epochs):
                _pyrandom.shuffle(order)
                epoch_loss = 0.0
                for batch_index, start in enumerate(
                    range(0, sample_count, batch_size)
                ):
                    indices = order[start:start + batch_size]
                    x_batch = _autograd_batch(X, indices, "train_layers")
                    y_batch = _autograd_batch(Y, indices, "train_layers")
                    x_value = _prepare_graph_input(x_batch, input_shape)
                    y_value = as_grad_tensor(y_batch)
                    logits = forward_layers([model, x_value])
                    loss_value = _autograd_loss(logits, y_value, loss_name)
                    loss_value.backward()
                    for param in params:
                        if optimizer == "adam":
                            param.apply_adam(
                                float(learning_rate), adam_states[id(param)]
                            )
                        else:
                            param.apply_sgd(float(learning_rate))
                        param.zero_grad()
                    epoch_loss += loss_value.item() * len(indices)
                    if show_progress:
                        completed_steps = epoch * batches_per_epoch + batch_index + 1
                        width = 20
                        filled = int(width * completed_steps / total_steps)
                        bar = "█" * filled + "░" * (width - filled)
                        running_loss = epoch_loss / (start + len(indices))
                        print(
                            f"\r[{bar}] "
                            f"{int(100 * completed_steps / total_steps)}% "
                            f"loss: {running_loss:.6g}",
                            end="",
                            flush=True,
                        )
                mean_loss = epoch_loss / sample_count
                history.append(mean_loss)
            if show_progress:
                print()
            return history

        def evaluate_layers(args):
            if len(args) < 3 or len(args) > 5:
                raise R("nn.evaluate_layers ожидает model, X, Y, loss?, input_shape?")
            model, X, Y = args[:3]
            loss_name = args[3] if len(args) >= 4 else "cce"
            input_shape = args[4] if len(args) >= 5 else None
            try:
                sample_count = len(X)
                if len(Y) != sample_count or sample_count == 0:
                    raise R("nn.evaluate_layers: X и Y должны иметь одинаковую длину > 0")
            except TypeError as error:
                raise R("nn.evaluate_layers: X и Y должны иметь длину") from error
            total_loss = 0.0
            correct = 0
            for start in range(0, sample_count, 256):
                indices = list(range(start, min(start + 256, sample_count)))
                x_batch = _autograd_batch(X, indices, "evaluate_layers")
                y_batch = _autograd_batch(Y, indices, "evaluate_layers")
                logits = forward_layers([
                    model, _prepare_graph_input(x_batch, input_shape)
                ])
                target = as_grad_tensor(y_batch)
                total_loss += _autograd_loss(logits, target, loss_name).item() * len(indices)
                output = logits.data
                if len(logits.shape()) == 2:
                    if logits.shape()[1] == 1:
                        for row, label in zip(output, y_batch):
                            target_value = label[0] if isinstance(label, list) else label
                            if (row[0] >= 0.0) == (target_value >= 0.5):
                                correct += 1
                    else:
                        for row, label in zip(output, y_batch):
                            if row.index(max(row)) == label.index(max(label)):
                                correct += 1
            return [total_loss / sample_count, correct / sample_count]

        def predict_layers(args):
            if len(args) < 2 or len(args) > 4:
                raise R("nn.predict_layers ожидает model, X, limit?, input_shape?")
            model, X = args[:2]
            limit = args[2] if len(args) >= 3 else None
            input_shape = args[3] if len(args) >= 4 else None
            if limit is not None and (
                not isinstance(limit, int) or isinstance(limit, bool) or limit < 0
            ):
                raise R("nn.predict_layers: limit должен быть целым >= 0")
            count = len(X)
            if limit is not None:
                count = min(count, limit)
            predictions = []
            for start in range(0, count, 256):
                indices = list(range(start, min(start + 256, count)))
                x_batch = _autograd_batch(X, indices, "predict_layers")
                logits = forward_layers([
                    model, _prepare_graph_input(x_batch, input_shape)
                ])
                if len(logits.shape()) == 2 and logits.shape()[1] == 1:
                    probabilities = logits.sigmoid().data
                else:
                    probabilities = logits.softmax().data
                predictions.extend(probabilities)
            return predictions

        def layer_data(name, layer):
            if not isinstance(layer, DictValue) or "W" not in layer or "b" not in layer:
                raise R(f"nn.{name} ожидает слой")
            W = layer["W"]
            b = layer["b"]
            W_list = W.to_list() if isinstance(W, Tensor) else W
            b_list = b.to_list() if isinstance(b, Tensor) else b
            if not isinstance(W_list, list) or not W_list:
                raise R(f"nn.{name}: W должен быть непустой матрицей")
            for row in W_list:
                ensure_vector(name, row)
            width = len(W_list[0])
            if width == 0 or any(len(row) != width for row in W_list):
                raise R(f"nn.{name}: строки W должны иметь одинаковую ненулевую длину")
            ensure_vector(name, b_list)
            if len(b_list) != len(W_list):
                raise R(f"nn.{name}: длина b должна совпадать с числом строк W")
            return W_list, b_list

        def _stable_sigmoid(x):
            x = float(x)
            if x >= 0.0:
                return 1.0 / (1.0 + _pymath.exp(-x))
            e = _pymath.exp(x)
            return e / (1.0 + e)

        def _softmax_vec(z):
            if not z:
                raise R("nn.softmax от пустого вектора")
            mx = max(float(x) for x in z)
            exps = [_pymath.exp(float(x) - mx) for x in z]
            total = sum(exps)
            return [e / total for e in exps]

        def _act_forward(act, z):
            if act == "sigmoid":
                return [_stable_sigmoid(v) for v in z]
            if act == "relu":
                return [max(0.0, float(x)) for x in z]
            if act == "leaky_relu":
                return [float(x) if float(x) > 0.0 else LEAKY_SLOPE * float(x) for x in z]
            raise R(f"nn: неизвестная активация {act!r}")

        def _act_grad(act, z, grad_out):
            if act == "sigmoid":
                out = []
                for x, g in zip(z, grad_out):
                    s = _stable_sigmoid(x)
                    out.append(g * s * (1.0 - s))
                return out
            if act == "relu":
                return [g * (1.0 if float(x) > 0.0 else 0.0) for x, g in zip(z, grad_out)]
            if act == "leaky_relu":
                return [g * (1.0 if float(x) > 0.0 else LEAKY_SLOPE) for x, g in zip(z, grad_out)]
            raise R(f"nn: неизвестная активация {act!r}")

        def shape(args):
            if len(args) != 1:
                raise R("nn.shape ожидает 1 аргумент")
            t = args[0]
            if isinstance(t, Tensor):
                return t.shape()
            if isinstance(t, list):
                if not t:
                    return [0]
                if isinstance(t[0], list):
                    cols = len(t[0])
                    for row in t:
                        if not isinstance(row, list):
                            raise R("nn.shape: матрица должна быть списком списков")
                        if len(row) != cols:
                            raise R("nn.shape: строки матрицы разной длины")
                    return [len(t), cols]
                return [len(t)]
            raise R("nn.shape ожидает список или тензор")

        def relu(args):
            if len(args) != 1: raise R("nn.relu ожидает 1 аргумент")
            if isinstance(args[0], GradTensor):
                return args[0].relu()
            ensure_number("relu", args[0])
            return max(0.0, float(args[0]))

        def leaky_relu(args):
            if len(args) != 1: raise R("nn.leaky_relu ожидает 1 аргумент")
            ensure_number("leaky_relu", args[0])
            x = float(args[0])
            return x if x > 0.0 else LEAKY_SLOPE * x

        def sigmoid(args):
            if len(args) != 1: raise R("nn.sigmoid ожидает 1 аргумент")
            if isinstance(args[0], GradTensor):
                return args[0].sigmoid()
            ensure_number("sigmoid", args[0])
            return _stable_sigmoid(args[0])

        def tanh(args):
            if len(args) != 1: raise R("nn.tanh ожидает 1 аргумент")
            if isinstance(args[0], GradTensor):
                return args[0].tanh()
            ensure_number("tanh", args[0])
            return _pymath.tanh(args[0])

        def relu_vec(args):
            if len(args) != 1: raise R("nn.relu_vec ожидает 1 аргумент")
            v = args[0]
            ensure_vector("relu_vec", v)
            return [max(0.0, float(x)) for x in v]

        def leaky_relu_vec(args):
            if len(args) != 1: raise R("nn.leaky_relu_vec ожидает 1 аргумент")
            v = args[0]
            ensure_vector("leaky_relu_vec", v)
            return [float(x) if float(x) > 0.0 else LEAKY_SLOPE * float(x) for x in v]

        def sigmoid_vec(args):
            if len(args) != 1: raise R("nn.sigmoid_vec ожидает 1 аргумент")
            v = args[0]
            ensure_vector("sigmoid_vec", v)
            return [_stable_sigmoid(v) for v in v]

        def softmax(args):
            if len(args) != 1: raise R("nn.softmax ожидает 1 аргумент")
            if isinstance(args[0], GradTensor):
                return args[0].softmax()
            v = args[0]
            ensure_vector("softmax", v)
            return _softmax_vec(v)

        def mse(args):
            if len(args) != 2: raise R("nn.mse ожидает 2 аргумента")
            yhat, y = args
            if isinstance(yhat, GradTensor):
                return yhat.mse_loss(y)
            ensure_vector("mse", yhat)
            ensure_vector("mse", y)
            if len(yhat) != len(y): raise R("nn.mse: длины не совпадают")
            if not yhat: raise R("nn.mse от пустых векторов")
            s = 0.0
            for a, b in zip(yhat, y):
                d = float(a) - float(b)
                s += d * d
            return s / len(yhat)

        def bce(args):
            if len(args) != 2: raise R("nn.bce ожидает 2 аргумента")
            yhat, y = args
            if isinstance(yhat, GradTensor):
                return yhat.binary_cross_entropy_probabilities(y)
            ensure_vector("bce", yhat)
            ensure_vector("bce", y)
            if len(yhat) != len(y): raise R("nn.bce: длины не совпадают")
            if not yhat: raise R("nn.bce от пустых векторов")
            eps = 1e-12
            s = 0.0
            for a, b in zip(yhat, y):
                p = float(a)
                t = float(b)
                if p < eps: p = eps
                if p > 1.0 - eps: p = 1.0 - eps
                s += -(t * _pymath.log(p) + (1.0 - t) * _pymath.log(1.0 - p))
            return s / len(yhat)

        def categorical_cross_entropy(args):
            if len(args) != 2: raise R("nn.cce ожидает 2 аргумента")
            yhat, y = args
            if isinstance(yhat, GradTensor):
                return yhat.categorical_cross_entropy(y)
            ensure_vector("cce", yhat)
            ensure_vector("cce", y)
            if len(yhat) != len(y): raise R("nn.cce: длины не совпадают")
            if not yhat: raise R("nn.cce от пустых векторов")
            eps = 1e-12
            s = 0.0
            for a, b in zip(yhat, y):
                p = float(a)
                if p < eps: p = eps
                s += -float(b) * _pymath.log(p)
            return s / len(yhat)

        def _init_layer_weights(in_dim, out_dim, act):
            if act == "sigmoid" or act == "leaky_relu":
                limit = _pymath.sqrt(6.0 / (in_dim + out_dim))
            else:
                limit = _pymath.sqrt(6.0 / in_dim)
            W_data = []
            for _ in range(out_dim):
                row = []
                for _ in range(in_dim):
                    row.append((_pyrandom.random() * 2.0 - 1.0) * limit)
                W_data.append(row)
            W = Tensor(W_data)
            b = Tensor([0.0] * out_dim)
            d = DictValue()
            d["W"] = W
            d["b"] = b
            return d

        def dense(args):
            if len(args) != 2: raise R("nn.dense ожидает 2 аргумента")
            in_dim, out_dim = args
            if not isinstance(in_dim, int) or isinstance(in_dim, bool) or in_dim <= 0:
                raise R("nn.dense: in_dim должен быть положительным целым")
            if not isinstance(out_dim, int) or isinstance(out_dim, bool) or out_dim <= 0:
                raise R("nn.dense: out_dim должен быть положительным целым")
            return _init_layer_weights(in_dim, out_dim, "sigmoid")

        def forward(args):
            if len(args) != 2: raise R("nn.forward ожидает 2 аргумента")
            layer, x = args
            W_list, b_list = layer_data("forward", layer)
            x_list = as_vector("forward", x)
            if len(x_list) != len(W_list[0]):
                raise R(
                    f"nn.forward: размер входа {len(x_list)} не совпадает "
                    f"с ожидаемым {len(W_list[0])}"
                )
            out = []
            for i in range(len(W_list)):
                s = float(b_list[i])
                for j in range(len(x_list)):
                    s += W_list[i][j] * float(x_list[j])
                out.append(s)
            return out

        def forward_dense(args):
            if len(args) != 2: raise R("nn.forward_dense ожидает 2 аргумента")
            layer, x = args
            W_list, b_list = layer_data("forward_dense", layer)
            x_list = as_vector("forward_dense", x)
            if len(x_list) != len(W_list[0]):
                raise R(
                    f"nn.forward_dense: размер входа {len(x_list)} не совпадает "
                    f"с ожидаемым {len(W_list[0])}"
                )
            z = []
            for i in range(len(W_list)):
                s = float(b_list[i])
                for j in range(len(x_list)):
                    s += W_list[i][j] * float(x_list[j])
                z.append(s)
            return [z, list(x_list)]

        def backward_dense(args):
            if len(args) != 3: raise R("nn.backward_dense ожидает 3 аргумента")
            layer, x, grad_z = args
            W_list, _ = layer_data("backward_dense", layer)
            x_list = as_vector("backward_dense", x)
            gz = as_vector("backward_dense", grad_z)
            if len(x_list) != len(W_list[0]):
                raise R(
                    f"nn.backward_dense: размер входа {len(x_list)} не совпадает "
                    f"с ожидаемым {len(W_list[0])}"
                )
            if len(gz) != len(W_list):
                raise R(
                    f"nn.backward_dense: размер grad_z {len(gz)} не совпадает "
                    f"с ожидаемым {len(W_list)}"
                )
            grad_W = []
            for i in range(len(W_list)):
                row = []
                for j in range(len(x_list)):
                    row.append(float(gz[i]) * float(x_list[j]))
                grad_W.append(row)
            grad_b = [float(g) for g in gz]
            grad_x = []
            for j in range(len(x_list)):
                s = 0.0
                for i in range(len(W_list)):
                    s += W_list[i][j] * float(gz[i])
                grad_x.append(s)
            return [grad_W, grad_b, grad_x]

        def sigmoid_grad(args):
            if len(args) != 1: raise R("nn.sigmoid_grad ожидает 1 аргумент")
            ensure_number("sigmoid_grad", args[0])
            s = _stable_sigmoid(args[0])
            return s * (1.0 - s)

        def relu_grad(args):
            if len(args) != 1: raise R("nn.relu_grad ожидает 1 аргумент")
            ensure_number("relu_grad", args[0])
            return 1.0 if float(args[0]) > 0.0 else 0.0

        def mse_grad(args):
            if len(args) != 2: raise R("nn.mse_grad ожидает 2 аргумента")
            yhat, y = args
            ensure_vector("mse_grad", yhat)
            ensure_vector("mse_grad", y)
            if len(yhat) != len(y): raise R("nn.mse_grad: длины не совпадают")
            if not yhat: raise R("nn.mse_grad от пустых векторов")
            n = len(yhat)
            return [2.0 * (float(a) - float(b)) / n for a, b in zip(yhat, y)]

        def bce_grad(args):
            if len(args) != 2: raise R("nn.bce_grad ожидает 2 аргумента")
            yhat, y = args
            ensure_vector("bce_grad", yhat)
            ensure_vector("bce_grad", y)
            if len(yhat) != len(y): raise R("nn.bce_grad: длины не совпадают")
            if not yhat: raise R("nn.bce_grad от пустых векторов")
            n = len(yhat)
            return [(float(a) - float(b)) / n for a, b in zip(yhat, y)]

        def cce_grad(args):
            if len(args) != 2: raise R("nn.cce_grad ожидает 2 аргумента")
            yhat, y = args
            ensure_vector("cce_grad", yhat)
            ensure_vector("cce_grad", y)
            if len(yhat) != len(y): raise R("nn.cce_grad: длины не совпадают")
            if not yhat: raise R("nn.cce_grad от пустых векторов")
            n = len(yhat)
            return [(float(a) - float(b)) / n for a, b in zip(yhat, y)]

        def rand_seed(args):
            if len(args) != 1: raise R("nn.rand_seed ожидает 1 аргумент")
            seed = args[0]
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise R("nn.rand_seed: seed должен быть целым")
            _pyrandom.seed(seed)
            return 0

        def accuracy(args):
            if len(args) != 2: raise R("nn.accuracy ожидает yhat и y")
            yhat, y = args
            ensure_vector("accuracy", yhat)
            ensure_vector("accuracy", y)
            if len(yhat) != len(y): raise R("nn.accuracy: длины не совпадают")
            if not yhat: raise R("nn.accuracy от пустых векторов")
            correct = 0
            for a, b in zip(yhat, y):
                pred = 1.0 if float(a) >= 0.5 else 0.0
                if pred == float(b):
                    correct += 1
            return correct / len(yhat)

        def mlp(args):
            if len(args) < 1 or len(args) > 3:
                raise R("nn.mlp ожидает sizes, опционально act и out_act")
            sizes = args[0]
            act = args[1] if len(args) >= 2 else "leaky_relu"
            out_act = args[2] if len(args) == 3 else "sigmoid"
            if not isinstance(sizes, list) or len(sizes) < 2:
                raise R("nn.mlp: sizes должен быть списком хотя бы из 2 чисел")
            for s in sizes:
                if not isinstance(s, int) or isinstance(s, bool) or s <= 0:
                    raise R("nn.mlp: все размеры должны быть положительными целыми")
            if not isinstance(act, str) or act not in ("sigmoid", "relu", "leaky_relu"):
                raise R(f"nn.mlp: неизвестная активация {act!r}")
            if not isinstance(out_act, str) or out_act not in ("sigmoid", "softmax", "linear"):
                raise R(f"nn.mlp: неизвестная выходная активация {out_act!r}")
            layers = []
            n = len(sizes) - 1
            for k in range(n):
                in_dim = sizes[k]
                out_dim = sizes[k + 1]
                layer_act = act if k < n - 1 else ("sigmoid" if out_act == "sigmoid" else "relu")
                layer = _init_layer_weights(in_dim, out_dim, layer_act)
                layers.append(layer)
            model = DictValue()
            model["layers"] = layers
            model["act"] = act
            model["out_act"] = out_act
            return model

        def forward_mlp(args):
            if len(args) != 2: raise R("nn.forward_mlp ожидает model и x")
            model, x = args
            if not isinstance(model, DictValue) or "layers" not in model:
                raise R("nn.forward_mlp ожидает модель")
            layers = model["layers"]
            act = model["act"]
            out_act = model["out_act"]
            x_list = as_vector("forward_mlp", x)
            cache = [list(x_list)]
            cur = list(x_list)
            n = len(layers)
            for k in range(n):
                layer = layers[k]
                W_list, b_list = layer_data("forward_mlp", layer)
                if len(cur) != len(W_list[0]):
                    raise R(
                        f"nn.forward_mlp: размер входа слоя {k} ({len(cur)}) "
                        f"не совпадает с ожидаемым {len(W_list[0])}"
                    )
                z = []
                for i in range(len(W_list)):
                    s = float(b_list[i])
                    for j in range(len(cur)):
                        s += W_list[i][j] * float(cur[j])
                    z.append(s)
                cache.append(z)
                if k == n - 1:
                    if out_act == "softmax":
                        h = _softmax_vec(z)
                    elif out_act == "linear":
                        h = list(z)
                    else:
                        h = [_stable_sigmoid(v) for v in z]
                else:
                    h = _act_forward(act, z)
                cache.append(h)
                cur = h
            return cache

        def backward_mlp(args):
            if len(args) != 3 and len(args) != 4:
                raise R("nn.backward_mlp ожидает model, cache, grad_out (опционально skip_last_act)")
            model, cache, grad_out = args[0], args[1], args[2]
            skip_last_act = args[3] if len(args) == 4 else False
            if not isinstance(skip_last_act, bool):
                raise R("nn.backward_mlp: skip_last_act должен быть bool")
            if not isinstance(model, DictValue) or "layers" not in model:
                raise R("nn.backward_mlp ожидает модель")
            layers = model["layers"]
            act = model["act"]
            out_act = model["out_act"]
            grad_list = as_vector("backward_mlp", grad_out)
            n = len(layers)
            expected = 1 + 2 * n
            if len(cache) != expected:
                raise R(f"nn.backward_mlp: cache должен быть длины {expected}")
            grads_rev = []
            grad = list(grad_list)
            for k in range(n - 1, -1, -1):
                x_in = cache[2 * k]
                z = cache[2 * k + 1]
                x_in = as_vector("backward_mlp", x_in)
                z = as_vector("backward_mlp", z)
                layer = layers[k]
                W_list, _ = layer_data("backward_mlp", layer)
                if len(x_in) != len(W_list[0]) or len(z) != len(W_list):
                    raise R(f"nn.backward_mlp: некорректная форма cache для слоя {k}")
                if len(grad) != len(z):
                    raise R(
                        f"nn.backward_mlp: размер градиента {len(grad)} "
                        f"не совпадает с размером слоя {len(z)}"
                    )
                if k == n - 1:
                    if skip_last_act or out_act == "linear":
                        grad_z = list(grad)
                    elif out_act == "sigmoid":
                        grad_z = []
                        for zi, gi in zip(z, grad):
                            s = _stable_sigmoid(zi)
                            grad_z.append(gi * s * (1.0 - s))
                    elif out_act == "softmax":
                        probabilities = as_vector(
                            "backward_mlp", cache[2 * k + 2]
                        )
                        dot = sum(g * p for g, p in zip(grad, probabilities))
                        grad_z = [
                            p * (g - dot) for g, p in zip(grad, probabilities)
                        ]
                    else:
                        raise R(f"nn.backward_mlp: неизвестная активация {out_act!r}")
                else:
                    grad_z = _act_grad(act, z, grad)
                grad_W = []
                for i in range(len(W_list)):
                    row = []
                    for j in range(len(x_in)):
                        row.append(float(grad_z[i]) * float(x_in[j]))
                    grad_W.append(row)
                grad_b = [float(g) for g in grad_z]
                grads_rev.append([grad_W, grad_b])
                grad_x = []
                for j in range(len(x_in)):
                    s = 0.0
                    for i in range(len(W_list)):
                        s += W_list[i][j] * float(grad_z[i])
                    grad_x.append(s)
                grad = grad_x
            grads_rev.reverse()
            return grads_rev

        def sgd_step(args):
            if len(args) != 3: raise R("nn.sgd_step ожидает model, grads, lr")
            model, grads, lr = args
            if not isinstance(model, DictValue) or "layers" not in model:
                raise R("nn.sgd_step ожидает модель")
            ensure_number("sgd_step", lr)
            layers = model["layers"]
            if not isinstance(grads, list) or len(grads) != len(layers):
                raise R("nn.sgd_step: grads должен быть списком по слоям")
            for k in range(len(layers)):
                layer = layers[k]
                W_list = layer["W"].to_list() if isinstance(layer["W"], Tensor) else layer["W"]
                b_list = layer["b"].to_list() if isinstance(layer["b"], Tensor) else layer["b"]
                grad_W = grads[k][0]
                grad_b = grads[k][1]
                for i in range(len(W_list)):
                    for j in range(len(W_list[i])):
                        W_list[i][j] = W_list[i][j] - float(lr) * grad_W[i][j]
                    b_list[i] = b_list[i] - float(lr) * grad_b[i]
                layer["W"] = Tensor(W_list)
                layer["b"] = Tensor(b_list)
            return model

        def _zero_like_matrix(W):
            if isinstance(W, Tensor):
                return Tensor.zeros(W.shape())
            return [[0.0 for _ in row] for row in W]

        def _zero_like_vec(v):
            if isinstance(v, Tensor):
                return Tensor.zeros(v.shape())
            return [0.0 for _ in v]

        def adam_init(args):
            if len(args) != 1: raise R("nn.adam_init ожидает model")
            model = args[0]
            if not isinstance(model, DictValue) or "layers" not in model:
                raise R("nn.adam_init ожидает модель")
            layers = model["layers"]
            state_layers = []
            for layer in layers:
                W = layer["W"]
                b = layer["b"]
                s = DictValue()
                s["m_W"] = _zero_like_matrix(W)
                s["v_W"] = _zero_like_matrix(W)
                s["m_b"] = _zero_like_vec(b)
                s["v_b"] = _zero_like_vec(b)
                state_layers.append(s)
            state = DictValue()
            state["layers"] = state_layers
            state["t"] = 0
            return state

        def adam_step(args):
            if len(args) != 4 and len(args) != 7:
                raise R("nn.adam_step ожидает model, grads, state, lr (опционально beta1, beta2, eps)")
            model, grads, state, lr = args[0], args[1], args[2], args[3]
            beta1 = args[4] if len(args) > 4 else 0.9
            beta2 = args[5] if len(args) > 5 else 0.999
            eps = args[6] if len(args) > 6 else 1e-8
            if not isinstance(model, DictValue) or "layers" not in model:
                raise R("nn.adam_step ожидает модель")
            if not isinstance(state, DictValue) or "layers" not in state:
                raise R("nn.adam_step ожидает состояние")
            ensure_number("adam_step", lr)
            layers = model["layers"]
            state_layers = state["layers"]
            if len(grads) != len(layers):
                raise R("nn.adam_step: grads не совпадает по числу слоёв")
            state["t"] = state["t"] + 1
            t = state["t"]
            bc1 = 1.0 - beta1 ** t
            bc2 = 1.0 - beta2 ** t
            for k in range(len(layers)):
                layer = layers[k]
                W_list = layer["W"].to_list() if isinstance(layer["W"], Tensor) else layer["W"]
                b_list = layer["b"].to_list() if isinstance(layer["b"], Tensor) else layer["b"]
                grad_W = grads[k][0]
                grad_b = grads[k][1]
                s = state_layers[k]
                m_W = s["m_W"].to_list() if isinstance(s["m_W"], Tensor) else s["m_W"]
                v_W = s["v_W"].to_list() if isinstance(s["v_W"], Tensor) else s["v_W"]
                m_b = s["m_b"].to_list() if isinstance(s["m_b"], Tensor) else s["m_b"]
                v_b = s["v_b"].to_list() if isinstance(s["v_b"], Tensor) else s["v_b"]
                for i in range(len(W_list)):
                    for j in range(len(W_list[i])):
                        g = float(grad_W[i][j])
                        m_W[i][j] = beta1 * m_W[i][j] + (1.0 - beta1) * g
                        v_W[i][j] = beta2 * v_W[i][j] + (1.0 - beta2) * (g * g)
                        mhat = m_W[i][j] / bc1
                        vhat = v_W[i][j] / bc2
                        W_list[i][j] = W_list[i][j] - float(lr) * mhat / (_pymath.sqrt(vhat) + eps)
                    g = float(grad_b[i])
                    m_b[i] = beta1 * m_b[i] + (1.0 - beta1) * g
                    v_b[i] = beta2 * v_b[i] + (1.0 - beta2) * (g * g)
                    mhat = m_b[i] / bc1
                    vhat = v_b[i] / bc2
                    b_list[i] = b_list[i] - float(lr) * mhat / (_pymath.sqrt(vhat) + eps)
                layer["W"] = Tensor(W_list)
                layer["b"] = Tensor(b_list)
                s["m_W"] = Tensor(m_W)
                s["v_W"] = Tensor(v_W)
                s["m_b"] = Tensor(m_b)
                s["v_b"] = Tensor(v_b)
            return model

        def momentum_init(args):
            if len(args) != 1: raise R("nn.momentum_init ожидает model")
            model = args[0]
            if not isinstance(model, DictValue) or "layers" not in model:
                raise R("nn.momentum_init ожидает модель")
            layers = model["layers"]
            state_layers = []
            for layer in layers:
                W = layer["W"]
                b = layer["b"]
                s = DictValue()
                s["v_W"] = _zero_like_matrix(W)
                s["v_b"] = _zero_like_vec(b)
                state_layers.append(s)
            state = DictValue()
            state["layers"] = state_layers
            return state

        def momentum_step(args):
            if len(args) != 5: raise R("nn.momentum_step ожидает model, grads, state, lr, momentum")
            model, grads, state, lr, momentum = args
            if not isinstance(model, DictValue) or "layers" not in model:
                raise R("nn.momentum_step ожидает модель")
            if not isinstance(state, DictValue) or "layers" not in state:
                raise R("nn.momentum_step ожидает состояние")
            ensure_number("momentum_step", lr)
            ensure_number("momentum_step", momentum)
            layers = model["layers"]
            state_layers = state["layers"]
            if len(grads) != len(layers):
                raise R("nn.momentum_step: grads не совпадает по числу слоёв")
            for k in range(len(layers)):
                layer = layers[k]
                W_list = layer["W"].to_list() if isinstance(layer["W"], Tensor) else layer["W"]
                b_list = layer["b"].to_list() if isinstance(layer["b"], Tensor) else layer["b"]
                grad_W = grads[k][0]
                grad_b = grads[k][1]
                s = state_layers[k]
                v_W = s["v_W"].to_list() if isinstance(s["v_W"], Tensor) else s["v_W"]
                v_b = s["v_b"].to_list() if isinstance(s["v_b"], Tensor) else s["v_b"]
                for i in range(len(W_list)):
                    for j in range(len(W_list[i])):
                        v_W[i][j] = float(momentum) * v_W[i][j] - float(lr) * float(grad_W[i][j])
                        W_list[i][j] = W_list[i][j] + v_W[i][j]
                    v_b[i] = float(momentum) * v_b[i] - float(lr) * float(grad_b[i])
                    b_list[i] = b_list[i] + v_b[i]
                layer["W"] = Tensor(W_list)
                layer["b"] = Tensor(b_list)
                s["v_W"] = Tensor(v_W)
                s["v_b"] = Tensor(v_b)
            return model

        def summary(args):
            if len(args) != 1: raise R("nn.summary ожидает model")
            model = args[0]
            if not isinstance(model, DictValue) or "layers" not in model:
                raise R("nn.summary ожидает модель")
            layers = model["layers"]
            act = model["act"]
            out_act = model["out_act"]
            total = 0
            print("Model summary:")
            print("  act:", act, "out_act:", out_act)
            for idx in range(len(layers)):
                layer = layers[idx]
                W = layer["W"]
                W_shape = W.shape() if isinstance(W, Tensor) else [len(W), len(W[0])]
                b = layer["b"]
                b_len = len(b) if not isinstance(b, Tensor) else b.shape()[0]
                params = W_shape[0] * W_shape[1] + b_len
                total = total + params
                print(f"  layer {idx} : in = {W_shape[1]} out = {W_shape[0]} params = {params}")
            print("  total params:", total)
            return 0

        def _model_to_plain(model):
            layers = model["layers"]
            act = model["act"]
            out_act = model["out_act"]
            plain_layers = []
            for layer in layers:
                W = layer["W"]
                b = layer["b"]
                W_list = W.to_list() if isinstance(W, Tensor) else W
                b_list = b.to_list() if isinstance(b, Tensor) else b
                plain_layers.append({
                    "W": [[float(x) for x in row] for row in W_list],
                    "b": [float(x) for x in b_list],
                })
            return {
                "version": 1,
                "act": act,
                "out_act": out_act,
                "layers": plain_layers,
            }

        def _plain_to_model(data):
            if not isinstance(data, dict) or "layers" not in data or "act" not in data:
                raise R("load: некорректный формат модели")
            version = data.get("version", 0)
            if version > 1:
                raise R(f"load: версия формата {version} не поддерживается")
            out_act = data.get("out_act", "sigmoid")
            if out_act not in ("sigmoid", "softmax", "linear"):
                raise R(f"load: неизвестный out_act {out_act!r}")
            layers = []
            for pl in data["layers"]:
                if "W" not in pl or "b" not in pl:
                    raise R("load: в слое нет W или b")
                layer = DictValue()
                layer["W"] = Tensor(pl["W"])
                layer["b"] = Tensor(pl["b"])
                layers.append(layer)
            model = DictValue()
            model["layers"] = layers
            model["act"] = data["act"]
            model["out_act"] = out_act
            return model

        def _resolve_path(path):
            if os.path.isabs(path):
                return path
            return os.path.join(values.URAY_BASE_DIR, path)

        def save(args):
            if len(args) != 2: raise R("nn.save ожидает model и path")
            model, path = args
            if not isinstance(model, DictValue) or "layers" not in model:
                raise R("nn.save ожидает модель")
            if not isinstance(path, str):
                raise R("nn.save: path должен быть строкой")
            data = _model_to_plain(model)
            full_path = _resolve_path(path)
            try:
                with open(full_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except OSError as e:
                raise R(f"nn.save: не удалось записать: {e}")
            return full_path

        def load(args):
            if len(args) != 1: raise R("nn.load ожидает path")
            path = args[0]
            if not isinstance(path, str):
                raise R("nn.load: path должен быть строкой")
            full_path = _resolve_path(path)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except OSError as e:
                raise R(f"nn.load: не удалось прочитать: {e}")
            except ValueError as e:
                raise R(f"nn.load: некорректный JSON: {e}")
            return _plain_to_model(data)

        def save_layers(args):
            if len(args) != 2:
                raise R("nn.save_layers ожидает sequential-модель и path")
            model, path = args
            if (
                not isinstance(model, DictValue)
                or "kind" not in model
                or model["kind"] != "sequential"
            ):
                raise R("nn.save_layers ожидает модель nn.sequential")
            if not isinstance(path, str):
                raise R("nn.save_layers: path должен быть строкой")
            plain_layers = []
            for layer in model["layers"]:
                if not isinstance(layer, DictValue) or "kind" not in layer:
                    raise R("nn.save_layers: некорректный слой")
                kind = layer["kind"]
                plain = {"kind": kind}
                if kind in ("linear", "conv2d"):
                    plain["W"] = layer["W"].to_list()
                    plain["b"] = layer["b"].to_list()
                if kind == "conv2d":
                    plain["stride"] = layer["stride"]
                    plain["padding"] = layer["padding"]
                elif kind == "activation":
                    plain["name"] = layer["name"]
                elif kind == "max_pool2d":
                    plain["kernel"] = layer["kernel"]
                    plain["stride"] = layer["stride"]
                elif kind not in ("linear", "conv2d", "flatten"):
                    raise R(f"nn.save_layers: неизвестный тип слоя {kind!r}")
                plain_layers.append(plain)
            data = {"format": "uray-sequential", "version": 1, "layers": plain_layers}
            full_path = _resolve_path(path)
            try:
                with open(full_path, "w", encoding="utf-8") as output:
                    json.dump(data, output, ensure_ascii=False, indent=2)
            except OSError as error:
                raise R(f"nn.save_layers: не удалось записать: {error}")
            return full_path

        def load_layers(args):
            if len(args) != 1 or not isinstance(args[0], str):
                raise R("nn.load_layers ожидает path")
            full_path = _resolve_path(args[0])
            try:
                with open(full_path, "r", encoding="utf-8") as source:
                    data = json.load(source)
            except OSError as error:
                raise R(f"nn.load_layers: не удалось прочитать: {error}")
            except ValueError as error:
                raise R(f"nn.load_layers: некорректный JSON: {error}")
            if (
                not isinstance(data, dict)
                or data.get("format") != "uray-sequential"
                or data.get("version") != 1
                or not isinstance(data.get("layers"), list)
            ):
                raise R("nn.load_layers: неподдерживаемый формат модели")
            layers = []
            for plain in data["layers"]:
                if not isinstance(plain, dict) or "kind" not in plain:
                    raise R("nn.load_layers: некорректный слой")
                kind = plain["kind"]
                layer = DictValue()
                layer["kind"] = kind
                if kind in ("linear", "conv2d"):
                    if "W" not in plain or "b" not in plain:
                        raise R("nn.load_layers: в слое отсутствуют веса или bias")
                    layer["W"] = as_grad_tensor(plain["W"], requires_grad=True)
                    layer["b"] = as_grad_tensor(plain["b"], requires_grad=True)
                if kind == "conv2d":
                    layer["stride"] = plain.get("stride", 1)
                    layer["padding"] = plain.get("padding", 0)
                elif kind == "activation":
                    if plain.get("name") not in ("relu", "sigmoid", "tanh", "softmax"):
                        raise R("nn.load_layers: неизвестная активация")
                    layer["name"] = plain["name"]
                elif kind == "max_pool2d":
                    layer["kernel"] = plain.get("kernel")
                    layer["stride"] = plain.get("stride")
                elif kind not in ("linear", "conv2d", "flatten"):
                    raise R(f"nn.load_layers: неизвестный тип слоя {kind!r}")
                layers.append(layer)
            model = DictValue()
            model["kind"] = "sequential"
            model["layers"] = layers
            return model

        def export_pytorch(args):
            if len(args) != 2:
                raise R("nn.export_pytorch ожидает model и path")
            model, path = args
            if not isinstance(model, DictValue) or "layers" not in model:
                raise R("nn.export_pytorch ожидает модель")
            if not isinstance(path, str):
                raise R("nn.export_pytorch: path должен быть строкой")
            try:
                import torch
            except ImportError as e:
                raise R(
                    "nn.export_pytorch требует PyTorch. Установите его отдельно: "
                    "python3 -m pip install torch"
                ) from e

            plain = _model_to_plain(model)
            plain_layers = plain["layers"]
            if not plain_layers:
                raise R("nn.export_pytorch: модель должна содержать хотя бы один слой")
            sizes = [len(plain_layers[0]["W"][0])]
            state_dict = {}
            for index, layer in enumerate(plain_layers):
                weights = layer["W"]
                if not weights or any(len(row) != len(weights[0]) for row in weights):
                    raise R(f"nn.export_pytorch: некорректная матрица весов слоя {index}")
                if len(layer["b"]) != len(weights):
                    raise R(f"nn.export_pytorch: некорректный bias слоя {index}")
                if len(weights[0]) != sizes[-1]:
                    raise R(
                        f"nn.export_pytorch: размер входа слоя {index} "
                        "не совпадает с предыдущим слоем"
                    )
                sizes.append(len(weights))
                state_dict[f"layers.{index}.weight"] = torch.tensor(
                    weights, dtype=torch.float32
                )
                state_dict[f"layers.{index}.bias"] = torch.tensor(
                    layer["b"], dtype=torch.float32
                )
            checkpoint = {
                "format": "uray.pytorch",
                "version": 1,
                "sizes": sizes,
                "act": plain["act"],
                "out_act": plain["out_act"],
                "state_dict": state_dict,
            }
            full_path = _resolve_path(path)
            try:
                torch.save(checkpoint, full_path)
            except (OSError, RuntimeError, ValueError) as e:
                raise R(f"nn.export_pytorch: не удалось записать checkpoint: {e}")
            return full_path

        def _compute_loss(loss_name, yhat, y):
            if loss_name == "bce": return bce([yhat, y])
            if loss_name == "mse": return mse([yhat, y])
            if loss_name == "cce": return categorical_cross_entropy([yhat, y])
            raise R(f"nn.train: неизвестная функция потерь {loss_name!r}")

        def _compute_grad(loss_name, yhat, y):
            if loss_name == "bce": return bce_grad([yhat, y])
            if loss_name == "mse": return mse_grad([yhat, y])
            if loss_name == "cce": return cce_grad([yhat, y])
            raise R(f"nn.train: неизвестная функция потерь {loss_name!r}")

        def load_mnist(args):
            if len(args) != 1 or not isinstance(args[0], str):
                raise R("nn.load_mnist ожидает путь к каталогу MNIST")
            try:
                splits = load_mnist_directory(_resolve_path(args[0]))
            except (OSError, ValueError) as error:
                raise R(f"nn.load_mnist: {error}")

            result = DictValue()
            for name, storage in splits.items():
                split = DictValue()
                split["X"] = MNISTView(storage, "images")
                split["Y"] = MNISTView(storage, "labels")
                split["count"] = storage.count
                split["rows"] = storage.rows
                split["columns"] = storage.columns
                result[name] = split
            return result

        def _dataset_length(data, operation):
            if isinstance(data, (list, MNISTView)):
                return len(data)
            raise R(f"{operation} ожидает список или набор MNIST")

        def _dataset_sample(data, index, operation):
            if isinstance(data, MNISTView):
                try:
                    return data.sample(index)
                except (IndexError, TypeError, ValueError) as error:
                    raise R(f"{operation}: {error}")
            return data[index]

        def _validate_dataset_pair(X, Y, operation):
            if isinstance(X, MNISTView) or isinstance(Y, MNISTView):
                if not isinstance(X, MNISTView) or not isinstance(Y, MNISTView):
                    raise R(f"{operation}: X и Y должны быть одним типом данных")
                if X.kind != "images" or Y.kind != "labels":
                    raise R(f"{operation}: MNIST ожидает X=изображения, Y=метки")
                if X.storage is not Y.storage:
                    raise R(f"{operation}: X и Y должны относиться к одному split")
            x_count = _dataset_length(X, operation)
            y_count = _dataset_length(Y, operation)
            if x_count != y_count:
                raise R(f"{operation}: X и Y разной длины")
            if x_count == 0:
                raise R(f"{operation}: набор данных не должен быть пустым")
            return x_count

        def _validate_mnist_model(model, X, Y, operation):
            if not isinstance(model, DictValue) or "layers" not in model:
                raise R(f"{operation} ожидает модель")
            if not model["layers"]:
                raise R(f"{operation}: модель должна содержать слои")
            if isinstance(X, MNISTView):
                first_W, _ = layer_data(operation, model["layers"][0])
                expected = X.storage.rows * X.storage.columns
                if len(first_W[0]) != expected:
                    raise R(
                        f"{operation}: модель ожидает {len(first_W[0])} входов, "
                        f"MNIST даёт {expected}"
                    )
            if isinstance(Y, MNISTView):
                last_W, _ = layer_data(operation, model["layers"][-1])
                if len(last_W) != 10:
                    raise R(
                        f"{operation}: выходной слой должен содержать 10 "
                        f"классов, получено {len(last_W)}"
                    )

        def _numpy_batch(data, indices, numpy, operation):
            try:
                if isinstance(data, MNISTView):
                    batch = data.batch(indices, numpy)
                else:
                    batch = numpy.asarray(
                        [data[index] for index in indices], dtype=numpy.float64
                    )
            except (TypeError, ValueError, IndexError) as error:
                raise R(f"{operation}: некорректный пакет данных: {error}")
            if batch.ndim != 2:
                raise R(f"{operation}: пакет должен быть двумерной матрицей")
            if not numpy.all(numpy.isfinite(batch)):
                raise R(f"{operation}: данные содержат NaN или бесконечность")
            return batch.astype(numpy.float64, copy=False)

        def _numpy_parameters(model, numpy, operation):
            weights = []
            biases = []
            for layer in model["layers"]:
                W, b = layer_data(operation, layer)
                weights.append(numpy.asarray(W, dtype=numpy.float64))
                biases.append(numpy.asarray(b, dtype=numpy.float64))
            return weights, biases

        def _numpy_forward(model, x, weights, biases, numpy):
            activations = [x]
            pre_activations = []
            current = x
            last = len(weights) - 1
            for index in range(len(weights)):
                z = current @ weights[index].T + biases[index]
                pre_activations.append(z)
                if index == last:
                    out_act = model["out_act"]
                    if out_act == "softmax":
                        shifted = z - numpy.max(z, axis=1, keepdims=True)
                        exp_z = numpy.exp(shifted)
                        current = exp_z / numpy.sum(
                            exp_z, axis=1, keepdims=True
                        )
                    elif out_act == "sigmoid":
                        current = 1.0 / (
                            1.0 + numpy.exp(-numpy.clip(z, -709.0, 709.0))
                        )
                    else:
                        current = z
                else:
                    act = model["act"]
                    if act == "relu":
                        current = numpy.maximum(z, 0.0)
                    elif act == "leaky_relu":
                        current = numpy.where(z > 0.0, z, 0.1 * z)
                    else:
                        current = 1.0 / (
                            1.0 + numpy.exp(-numpy.clip(z, -709.0, 709.0))
                        )
                activations.append(current)
            return activations, pre_activations

        def _numpy_predict(model, X, numpy, batch_size=512):
            count = _dataset_length(X, "nn.predict")
            weights, biases = _numpy_parameters(model, numpy, "nn.predict")
            result = []
            for start in range(0, count, batch_size):
                indices = list(range(start, min(start + batch_size, count)))
                xb = _numpy_batch(X, indices, numpy, "nn.predict")
                if xb.shape[1] != weights[0].shape[1]:
                    raise R(
                        f"nn.predict: модель ожидает {weights[0].shape[1]} "
                        f"входов, получено {xb.shape[1]}"
                    )
                activations, _ = _numpy_forward(
                    model, xb, weights, biases, numpy
                )
                result.extend(activations[-1].tolist())
            return result

        def sample(args):
            if len(args) != 2:
                raise R("nn.sample ожидает набор MNIST и индекс")
            data, index = args
            if not isinstance(data, MNISTView):
                raise R("nn.sample ожидает набор из nn.load_mnist")
            try:
                return data.sample(index)
            except (IndexError, TypeError, ValueError) as error:
                raise R(f"nn.sample: {error}")

        def _numpy_loss_rows(loss_name, predictions, targets, numpy):
            eps = 1e-12
            clipped = numpy.clip(predictions, eps, 1.0 - eps)
            if loss_name == "cce":
                return -numpy.mean(
                    targets * numpy.log(numpy.clip(predictions, eps, 1.0)),
                    axis=1,
                )
            if loss_name == "bce":
                return -numpy.mean(
                    targets * numpy.log(clipped)
                    + (1.0 - targets) * numpy.log(1.0 - clipped),
                    axis=1,
                )
            return numpy.mean((predictions - targets) ** 2, axis=1)

        def _numpy_evaluate(model, X, Y, loss_name, numpy, batch_size=512):
            count = _validate_dataset_pair(X, Y, "nn.evaluate")
            _validate_mnist_model(model, X, Y, "nn.evaluate")
            weights, biases = _numpy_parameters(model, numpy, "nn.evaluate")
            total_loss = 0.0
            correct = 0
            for start in range(0, count, batch_size):
                indices = list(range(start, min(start + batch_size, count)))
                xb = _numpy_batch(X, indices, numpy, "nn.evaluate")
                yb = _numpy_batch(Y, indices, numpy, "nn.evaluate")
                if xb.shape[1] != weights[0].shape[1]:
                    raise R("nn.evaluate: ширина X не совпадает с входом модели")
                if yb.shape[1] != weights[-1].shape[0]:
                    raise R("nn.evaluate: ширина Y не совпадает с выходом модели")
                activations, _ = _numpy_forward(
                    model, xb, weights, biases, numpy
                )
                predictions = activations[-1]
                total_loss += float(numpy.sum(
                    _numpy_loss_rows(loss_name, predictions, yb, numpy)
                ))
                if predictions.shape[1] == 1:
                    predicted = predictions[:, 0] >= 0.5
                    actual = yb[:, 0] >= 0.5
                else:
                    predicted = numpy.argmax(predictions, axis=1)
                    actual = numpy.argmax(yb, axis=1)
                correct += int(numpy.sum(predicted == actual))
            return [total_loss / count, correct / count]

        def _numpy_train(
            model, X, Y, epochs, lr, batch_size, loss_name, opt_name,
            show_progress, numpy, count
        ):
            weights, biases = _numpy_parameters(model, numpy, "nn.train")
            if not weights:
                raise R("nn.train: модель должна содержать слои")
            if loss_name == "cce" and model["out_act"] != "softmax":
                raise R("nn.train: cce требует out_act=\"softmax\"")
            if loss_name == "bce" and model["out_act"] != "sigmoid":
                raise R("nn.train: bce требует out_act=\"sigmoid\"")

            if opt_name == "adam":
                first_w = [numpy.zeros_like(value) for value in weights]
                second_w = [numpy.zeros_like(value) for value in weights]
                first_b = [numpy.zeros_like(value) for value in biases]
                second_b = [numpy.zeros_like(value) for value in biases]
                step = 0
            elif opt_name == "momentum":
                velocity_w = [numpy.zeros_like(value) for value in weights]
                velocity_b = [numpy.zeros_like(value) for value in biases]

            indices = list(range(count))
            history = []
            for epoch in range(epochs):
                _pyrandom.shuffle(indices)
                epoch_loss = 0.0
                for start in range(0, count, batch_size):
                    batch_indices = indices[start:start + batch_size]
                    xb = _numpy_batch(X, batch_indices, numpy, "nn.train")
                    yb = _numpy_batch(Y, batch_indices, numpy, "nn.train")
                    if xb.shape[1] != weights[0].shape[1]:
                        raise R(
                            f"nn.train: модель ожидает {weights[0].shape[1]} "
                            f"входов, получено {xb.shape[1]}"
                        )
                    if yb.shape[1] != weights[-1].shape[0]:
                        raise R(
                            f"nn.train: модель ожидает {weights[-1].shape[0]} "
                            f"выходов, получено {yb.shape[1]}"
                        )

                    activations, pre_activations = _numpy_forward(
                        model, xb, weights, biases, numpy
                    )
                    predictions = activations[-1]
                    row_losses = _numpy_loss_rows(
                        loss_name, predictions, yb, numpy
                    )
                    epoch_loss += float(numpy.sum(row_losses))
                    sample_count = len(batch_indices)
                    output_width = predictions.shape[1]

                    if loss_name == "cce":
                        grad = (predictions - yb) / (
                            sample_count * output_width
                        )
                    elif loss_name == "bce":
                        grad = (predictions - yb) / (
                            sample_count * output_width
                        )
                    else:
                        grad_a = 2.0 * (predictions - yb) / (
                            sample_count * output_width
                        )
                        out_act = model["out_act"]
                        if out_act == "sigmoid":
                            grad = grad_a * predictions * (1.0 - predictions)
                        elif out_act == "softmax":
                            projection = numpy.sum(
                                grad_a * predictions, axis=1, keepdims=True
                            )
                            grad = predictions * (grad_a - projection)
                        else:
                            grad = grad_a

                    grad_weights = [None] * len(weights)
                    grad_biases = [None] * len(biases)
                    for layer_index in range(len(weights) - 1, -1, -1):
                        grad_weights[layer_index] = (
                            grad.T @ activations[layer_index]
                        )
                        grad_biases[layer_index] = numpy.sum(grad, axis=0)
                        if layer_index == 0:
                            continue

                        grad_a = grad @ weights[layer_index]
                        z = pre_activations[layer_index - 1]
                        if model["act"] == "relu":
                            grad = grad_a * (z > 0.0)
                        elif model["act"] == "leaky_relu":
                            grad = grad_a * numpy.where(z > 0.0, 1.0, 0.1)
                        else:
                            hidden = activations[layer_index]
                            grad = grad_a * hidden * (1.0 - hidden)

                    if opt_name == "adam":
                        step += 1
                        beta1 = 0.9
                        beta2 = 0.999
                        for layer_index in range(len(weights)):
                            first_w[layer_index] = (
                                beta1 * first_w[layer_index]
                                + (1.0 - beta1) * grad_weights[layer_index]
                            )
                            second_w[layer_index] = (
                                beta2 * second_w[layer_index]
                                + (1.0 - beta2)
                                * grad_weights[layer_index] ** 2
                            )
                            first_b[layer_index] = (
                                beta1 * first_b[layer_index]
                                + (1.0 - beta1) * grad_biases[layer_index]
                            )
                            second_b[layer_index] = (
                                beta2 * second_b[layer_index]
                                + (1.0 - beta2)
                                * grad_biases[layer_index] ** 2
                            )
                            corrected_w = first_w[layer_index] / (
                                1.0 - beta1 ** step
                            )
                            corrected_vw = second_w[layer_index] / (
                                1.0 - beta2 ** step
                            )
                            corrected_b = first_b[layer_index] / (
                                1.0 - beta1 ** step
                            )
                            corrected_vb = second_b[layer_index] / (
                                1.0 - beta2 ** step
                            )
                            weights[layer_index] -= lr * corrected_w / (
                                numpy.sqrt(corrected_vw) + 1e-8
                            )
                            biases[layer_index] -= lr * corrected_b / (
                                numpy.sqrt(corrected_vb) + 1e-8
                            )
                    elif opt_name == "momentum":
                        for layer_index in range(len(weights)):
                            velocity_w[layer_index] = (
                                0.9 * velocity_w[layer_index]
                                - lr * grad_weights[layer_index]
                            )
                            velocity_b[layer_index] = (
                                0.9 * velocity_b[layer_index]
                                - lr * grad_biases[layer_index]
                            )
                            weights[layer_index] += velocity_w[layer_index]
                            biases[layer_index] += velocity_b[layer_index]
                    else:
                        for layer_index in range(len(weights)):
                            weights[layer_index] -= lr * grad_weights[layer_index]
                            biases[layer_index] -= lr * grad_biases[layer_index]

                mean_loss = epoch_loss / count
                history.append(mean_loss)
                if show_progress:
                    width = 20
                    current = epoch + 1
                    filled = int(width * current / epochs)
                    bar = "█" * filled + "░" * (width - filled)
                    pct = int(100 * current / epochs)
                    print(
                        f"\r[{bar}] {pct}% loss: {mean_loss:.6g}",
                        end="",
                        flush=True,
                    )
                    if current == epochs:
                        print()

            for index, layer in enumerate(model["layers"]):
                layer["W"] = Tensor(weights[index].tolist())
                layer["b"] = Tensor(biases[index].tolist())
            return history

        def predict(args):
            if len(args) not in (2, 3):
                raise R("nn.predict ожидает model, X и необязательный limit")
            model, X = args[0], args[1]
            count = _dataset_length(X, "nn.predict")
            limit = args[2] if len(args) == 3 else count
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
                raise R("nn.predict: limit должен быть неотрицательным целым")
            count = min(count, limit)
            if isinstance(X, MNISTView) and X.kind != "images":
                raise R("nn.predict: ожидаются изображения, а не метки")
            if count == 0:
                return []
            _validate_mnist_model(model, X, None, "nn.predict")
            numpy = get_numpy()
            if numpy is None and isinstance(X, MNISTView) and count >= 10000:
                raise R(
                    "nn.predict: полный MNIST требует NumPy; "
                    "запустите uray с --backend=numpy"
                )
            if numpy is not None:
                if count == _dataset_length(X, "nn.predict"):
                    return _numpy_predict(model, X, numpy)
                weights, biases = _numpy_parameters(model, numpy, "nn.predict")
                indices = list(range(count))
                xb = _numpy_batch(X, indices, numpy, "nn.predict")
                if xb.shape[1] != weights[0].shape[1]:
                    raise R(
                        f"nn.predict: модель ожидает {weights[0].shape[1]} "
                        f"входов, получено {xb.shape[1]}"
                    )
                activations, _ = _numpy_forward(
                    model, xb, weights, biases, numpy
                )
                return activations[-1].tolist()
            res = []
            for index in range(count):
                x = _dataset_sample(X, index, "nn.predict")
                cache = forward_mlp([model, x])
                res.append(cache[-1])
            return res

        def evaluate(args):
            if len(args) != 4: raise R("nn.evaluate ожидает model, X, Y, loss")
            model, X, Y, loss_name = args
            count = _validate_dataset_pair(X, Y, "nn.evaluate")
            if loss_name not in ("bce", "mse", "cce"):
                raise R(f"nn.evaluate: неизвестная функция потерь {loss_name!r}")
            _validate_mnist_model(model, X, Y, "nn.evaluate")
            numpy = get_numpy()
            if numpy is None and isinstance(X, MNISTView) and count >= 10000:
                raise R(
                    "nn.evaluate: полный MNIST требует NumPy; "
                    "запустите uray с --backend=numpy"
                )
            if numpy is not None:
                return _numpy_evaluate(model, X, Y, loss_name, numpy)
            total_loss = 0.0
            correct = 0
            for i in range(count):
                x = _dataset_sample(X, i, "nn.evaluate")
                y = _dataset_sample(Y, i, "nn.evaluate")
                yhat = forward_mlp([model, x])[-1]
                total_loss += _compute_loss(loss_name, yhat, y)
                if len(yhat) == 1:
                    pred = 1.0 if yhat[0] >= 0.5 else 0.0
                    target = 1.0 if y[0] >= 0.5 else 0.0
                    if pred == target:
                        correct += 1
                else:
                    pred = 0
                    for k in range(1, len(yhat)):
                        if yhat[k] > yhat[pred]: pred = k
                    true_best = 0
                    for k in range(1, len(y)):
                        if y[k] > y[true_best]: true_best = k
                    if pred == true_best: correct += 1
            return [total_loss / len(X), correct / len(X)]

        def train(args):
            if len(args) < 5 or len(args) > 9:
                raise R("nn.train ожидает model, X, Y, epochs, lr (опционально batch_size, loss, optimizer, show_progress)")
            model, X, Y, epochs, lr = args[0], args[1], args[2], args[3], args[4]
            batch_size = args[5] if len(args) > 5 else 32
            loss_name = args[6] if len(args) > 6 else "bce"
            opt_name = args[7] if len(args) > 7 else "adam"
            show_progress = args[8] if len(args) > 8 else False
            if not isinstance(X, (list, MNISTView)) \
                    or not isinstance(Y, (list, MNISTView)):
                raise R("nn.train ожидает списки X и Y или набор MNIST")
            count = _validate_dataset_pair(X, Y, "nn.train")
            if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0: raise R("nn.train: epochs > 0")
            if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0: raise R("nn.train: batch_size > 0")
            if not isinstance(show_progress, bool):
                raise R("nn.train: show_progress должен быть bool")
            if not isinstance(model, DictValue) or "layers" not in model:
                raise R("nn.train ожидает модель")
            if loss_name not in ("bce", "mse", "cce"):
                raise R(f"nn.train: неизвестная функция потерь {loss_name!r}")
            if loss_name == "bce" and model["out_act"] != "sigmoid":
                raise R("nn.train: bce требует out_act=\"sigmoid\"")
            if loss_name == "cce" and model["out_act"] != "softmax":
                raise R("nn.train: cce требует out_act=\"softmax\"")
            ensure_number("train", lr)
            _validate_mnist_model(model, X, Y, "nn.train")
            if opt_name not in ("sgd", "momentum", "adam"):
                raise R(f"nn.train: неизвестный оптимизатор {opt_name!r}")
            numpy = get_numpy()
            if numpy is None and isinstance(X, MNISTView) and count >= 10000:
                raise R(
                    "nn.train: полный MNIST требует NumPy; "
                    "запустите uray с --backend=numpy"
                )
            if numpy is not None:
                return _numpy_train(
                    model, X, Y, epochs, lr, batch_size, loss_name,
                    opt_name, show_progress, numpy, count
                )
            state = None
            if opt_name == "adam": state = adam_init([model])
            elif opt_name == "momentum": state = momentum_init([model])
            history = []
            n = count
            indices = list(range(n))
            for epoch in range(epochs):
                _pyrandom.shuffle(indices)
                epoch_loss = 0.0
                for i in range(0, n, batch_size):
                    batch_indices = indices[i:i+batch_size]
                    count = len(batch_indices)
                    accum_grads = None
                    batch_loss = 0.0
                    for idx in batch_indices:
                        x = _dataset_sample(X, idx, "nn.train")
                        y = _dataset_sample(Y, idx, "nn.train")
                        cache = forward_mlp([model, x])
                        yhat = cache[-1]
                        batch_loss += _compute_loss(loss_name, yhat, y)
                        grad_out = _compute_grad(loss_name, yhat, y)
                        # BCE+sigmoid and CCE+softmax have a fused dL/dz.
                        # Other loss/activation pairs need the activation's Jacobian.
                        skip_last_act = (
                            (loss_name == "bce" and model["out_act"] == "sigmoid")
                            or (loss_name == "cce" and model["out_act"] == "softmax")
                        )
                        grads = backward_mlp([model, cache, grad_out, skip_last_act])
                        if accum_grads is None:
                            accum_grads = grads
                        else:
                            for k in range(len(grads)):
                                for a in range(len(grads[k][0])):
                                    for b in range(len(grads[k][0][a])):
                                        accum_grads[k][0][a][b] += grads[k][0][a][b]
                                for b in range(len(grads[k][1])):
                                    accum_grads[k][1][b] += grads[k][1][b]
                    for k in range(len(accum_grads)):
                        for a in range(len(accum_grads[k][0])):
                            for b in range(len(accum_grads[k][0][a])):
                                accum_grads[k][0][a][b] /= count
                        for b in range(len(accum_grads[k][1])):
                            accum_grads[k][1][b] /= count
                    if opt_name == "adam": adam_step([model, accum_grads, state, lr])
                    elif opt_name == "momentum": momentum_step([model, accum_grads, state, lr, 0.9])
                    else: sgd_step([model, accum_grads, lr])
                    epoch_loss += batch_loss
                mean_loss = epoch_loss / n
                history.append(mean_loss)
                if show_progress:
                    width = 20
                    current = epoch + 1
                    filled = int(width * current / epochs)
                    bar = "█" * filled + "░" * (width - filled)
                    pct = int(100 * current / epochs)
                    print(
                        f"\r[{bar}] {pct}% loss: {mean_loss:.6g}",
                        end="",
                        flush=True,
                    )
                    if current == epochs:
                        print()
            return history

        return Module("nn", {
            "shape": Builtin("shape", shape),
            "tensor": Builtin("tensor", grad_tensor),
            "parameter": Builtin("parameter", parameter),
            "add": Builtin("add", grad_add),
            "sub": Builtin("sub", grad_sub),
            "multiply": Builtin("multiply", grad_mul),
            "divide": Builtin("divide", grad_div),
            "matmul": Builtin("matmul", grad_matmul),
            "reshape": Builtin("reshape", grad_reshape),
            "reduce_sum": Builtin("reduce_sum", grad_sum),
            "mean": Builtin("mean", grad_mean),
            "cross_entropy": Builtin("cross_entropy", grad_cross_entropy),
            "binary_cross_entropy": Builtin("binary_cross_entropy", grad_binary_cross_entropy),
            "backward": Builtin("backward", grad_backward),
            "grad": Builtin("grad", grad_value),
            "item": Builtin("item", grad_item),
            "parameters": Builtin("parameters", grad_parameters),
            "zero_grad": Builtin("zero_grad", grad_zero),
            "sgd_update": Builtin("sgd_update", grad_sgd),
            "conv2d": Builtin("conv2d", grad_conv2d),
            "max_pool2d": Builtin("max_pool2d", grad_max_pool2d),
            "linear_layer": Builtin("linear_layer", linear_layer),
            "conv2d_layer": Builtin("conv2d_layer", conv2d_layer),
            "activation_layer": Builtin("activation_layer", activation_layer),
            "max_pool2d_layer": Builtin("max_pool2d_layer", pool2d_layer),
            "flatten_layer": Builtin("flatten_layer", flatten_layer),
            "sequential": Builtin("sequential", sequential),
            "forward_layers": Builtin("forward_layers", forward_layers),
            "train_layers": Builtin("train_layers", train_layers),
            "evaluate_layers": Builtin("evaluate_layers", evaluate_layers),
            "predict_layers": Builtin("predict_layers", predict_layers),
            "save_layers": Builtin("save_layers", save_layers),
            "load_layers": Builtin("load_layers", load_layers),
            "relu": Builtin("relu", relu),
            "leaky_relu": Builtin("leaky_relu", leaky_relu),
            "sigmoid": Builtin("sigmoid", sigmoid),
            "tanh": Builtin("tanh", tanh),
            "relu_vec": Builtin("relu_vec", relu_vec),
            "leaky_relu_vec": Builtin("leaky_relu_vec", leaky_relu_vec),
            "sigmoid_vec": Builtin("sigmoid_vec", sigmoid_vec),
            "softmax": Builtin("softmax", softmax),
            "mse": Builtin("mse", mse),
            "bce": Builtin("bce", bce),
            "categorical_cross_entropy": Builtin("categorical_cross_entropy", categorical_cross_entropy),
            "dense": Builtin("dense", dense),
            "forward": Builtin("forward", forward),
            "forward_dense": Builtin("forward_dense", forward_dense),
            "backward_dense": Builtin("backward_dense", backward_dense),
            "sigmoid_grad": Builtin("sigmoid_grad", sigmoid_grad),
            "relu_grad": Builtin("relu_grad", relu_grad),
            "mse_grad": Builtin("mse_grad", mse_grad),
            "bce_grad": Builtin("bce_grad", bce_grad),
            "cce_grad": Builtin("cce_grad", cce_grad),
            "rand_seed": Builtin("rand_seed", rand_seed),
            "accuracy": Builtin("accuracy", accuracy),
            "mlp": Builtin("mlp", mlp),
            "forward_mlp": Builtin("forward_mlp", forward_mlp),
            "backward_mlp": Builtin("backward_mlp", backward_mlp),
            "sgd_step": Builtin("sgd_step", sgd_step),
            "adam_init": Builtin("adam_init", adam_init),
            "adam_step": Builtin("adam_step", adam_step),
            "momentum_init": Builtin("momentum_init", momentum_init),
            "momentum_step": Builtin("momentum_step", momentum_step),
            "summary": Builtin("summary", summary),
            "save": Builtin("save", save),
            "load": Builtin("load", load),
            "load_mnist": Builtin("load_mnist", load_mnist),
            "export_pytorch": Builtin("export_pytorch", export_pytorch),
            "predict": Builtin("predict", predict),
            "sample": Builtin("sample", sample),
            "evaluate": Builtin("evaluate", evaluate),
            "train": Builtin("train", train),
        })
