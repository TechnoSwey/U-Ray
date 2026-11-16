"""Built-in native CPU backend for U-Ray (standard library only)."""

import array
import ctypes
import hashlib
from itertools import chain
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from errors import RuntimeErrorUray


_ENGINE = None
_SIZE_T = ctypes.c_size_t
_VOID_P = ctypes.c_void_p
_DOUBLE_P = ctypes.POINTER(ctypes.c_double)
_SIZE_T_P = ctypes.POINTER(_SIZE_T)
_INT_P = ctypes.POINTER(ctypes.c_int)


def _find_compiler():
    requested = os.environ.get("CC")
    candidates = [requested] if requested else []
    candidates.extend(("cc", "clang", "gcc"))
    for candidate in candidates:
        if candidate and shutil.which(candidate):
            return candidate
    return None


def _compile_library():
    if sys.platform == "win32":
        raise RuntimeErrorUray(
            "Нативный backend пока не собирается в Windows-среде. "
            "Используйте --backend=python или Linux/Termux с C-компилятором."
        )

    source = Path(__file__).with_name("native_backend.c")
    try:
        source_bytes = source.read_bytes()
    except OSError as error:
        raise RuntimeErrorUray(
            f"Не удалось прочитать исходник нативного ядра: {error}"
        ) from error
    digest = hashlib.sha256(source_bytes).hexdigest()[:20]

    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    cache_dir = Path(tempfile.gettempdir()) / "uray-native" / digest
    library = cache_dir / f"liburay_native{suffix}"
    if library.is_file():
        return library

    compiler = _find_compiler()
    if compiler is None:
        raise RuntimeErrorUray(
            "Для --backend=native нужен C-компилятор (cc, clang или gcc). "
            "NumPy и сторонние ML-библиотеки не требуются."
        )
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RuntimeErrorUray(
            f"Не удалось создать кэш нативного ядра: {error}"
        ) from error

    temporary = cache_dir / f"liburay_native-{os.getpid()}{suffix}"
    if sys.platform == "darwin":
        command = [
            compiler, "-O3", "-std=c11", "-fPIC", "-dynamiclib",
            str(source), "-lm", "-o", str(temporary),
        ]
    else:
        command = [
            compiler, "-O3", "-std=c11", "-fPIC", "-shared",
            str(source), "-lm", "-o", str(temporary),
        ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeErrorUray(
            f"Не удалось собрать нативное ядро U-Ray: {error}"
        ) from error
    if result.returncode != 0:
        diagnostic = (result.stderr or result.stdout or "").strip()
        raise RuntimeErrorUray(
            "Не удалось собрать нативное ядро U-Ray"
            + (f": {diagnostic[-1200:]}" if diagnostic else "")
        )
    try:
        os.replace(temporary, library)
    except OSError as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        if not library.is_file():
            raise RuntimeErrorUray(
                f"Не удалось сохранить библиотеку U-Ray: {error}"
            ) from error
    return library


def _array_pointer(values, c_type):
    if not values:
        return _VOID_P()
    view = (c_type * len(values)).from_buffer(values)
    return ctypes.cast(view, _VOID_P), view


def _as_matrix(data, name):
    if hasattr(data, "to_list"):
        data = data.to_list()
    if not isinstance(data, (list, tuple)) or not data:
        raise RuntimeErrorUray(f"{name}: ожидается непустая матрица")
    rows = []
    width = None
    for row in data:
        if hasattr(row, "to_list"):
            row = row.to_list()
        if not isinstance(row, (list, tuple)):
            row = [row]
        if width is None:
            width = len(row)
            if width == 0:
                raise RuntimeErrorUray(f"{name}: строки матрицы пусты")
        elif len(row) != width:
            raise RuntimeErrorUray(f"{name}: строки имеют разную длину")
        converted = []
        for value in row:
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise RuntimeErrorUray(
                    f"{name}: элементы должны быть числами"
                ) from error
            if not math.isfinite(number):
                raise RuntimeErrorUray(
                    f"{name}: элементы должны быть конечными числами"
                )
            converted.append(number)
        rows.append(converted)
    return rows


def _flat_array(data, name):
    result = array.array("d")

    def append_values(value):
        if isinstance(value, (list, tuple)):
            for item in value:
                append_values(item)
            return
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise RuntimeErrorUray(f"{name}: элементы должны быть числами") from error
        if not math.isfinite(number):
            raise RuntimeErrorUray(f"{name}: элементы должны быть конечными")
        result.append(number)

    append_values(data)
    return result


def _nested_from_flat(values, shape):
    if len(shape) == 4:
        flat_values = values.tolist()
        offset = 0
        batches, channels, height, width = shape
        result = []
        for _ in range(batches):
            batch = []
            for _ in range(channels):
                channel = []
                for _ in range(height):
                    channel.append(flat_values[offset:offset + width])
                    offset += width
                batch.append(channel)
            result.append(batch)
        return result

    def build(offset, dimensions):
        if not dimensions:
            return values[offset]
        if len(dimensions) == 1:
            return list(values[offset:offset + dimensions[0]])
        width = math.prod(dimensions[1:]) if len(dimensions) > 1 else 1
        return [
            build(offset + index * width, dimensions[1:])
            for index in range(dimensions[0])
        ]

    return build(0, shape)


def _flat_4d_array(data):
    result = array.array("d")
    for sample in data:
        for channel in sample:
            for row in channel:
                result.extend(row)
    return result


def _dataset_buffers(x_data, y_data, sample_count):
    """Return C pointers for either lazy MNIST bytes or generic numeric rows."""
    from mnist import MNISTView

    if isinstance(x_data, MNISTView):
        if x_data.kind != "images":
            raise RuntimeErrorUray("Нативный MLP backend ожидает изображения")
        if y_data is not None and (
            not isinstance(y_data, MNISTView)
            or y_data.kind != "labels"
            or y_data.storage is not x_data.storage
        ):
            raise RuntimeErrorUray(
                "Нативный MLP backend ожидает соответствующие MNIST-метки"
            )
        images = x_data.storage.images
        labels = y_data.storage.labels if y_data is not None else None
        image_ref = ctypes.c_char_p(images)
        label_ref = ctypes.c_char_p(labels) if labels is not None else None
        image_pointer = ctypes.cast(image_ref, _VOID_P)
        label_pointer = (
            ctypes.cast(label_ref, _VOID_P) if label_ref is not None else _VOID_P()
        )
        return {
            "mnist_mode": 1,
            "x_pointer": _VOID_P(),
            "y_pointer": _VOID_P(),
            "image_pointer": image_pointer,
            "label_pointer": label_pointer,
            "references": (images, labels, image_ref, label_ref),
            "input_dim": x_data.storage.rows * x_data.storage.columns,
            "output_dim": 10 if y_data is not None else None,
        }

    if not isinstance(x_data, list) or (y_data is not None and not isinstance(y_data, list)):
        raise RuntimeErrorUray(
            "Нативный MLP backend ожидает списки или MNIST dataset"
        )
    if len(x_data) < sample_count or (
        y_data is not None and len(y_data) < sample_count
    ):
        raise RuntimeErrorUray("Набор данных короче указанного размера")

    x_rows = _as_matrix(x_data[:sample_count], "X")
    input_dim = len(x_rows[0])
    if any(len(row) != input_dim for row in x_rows):
        raise RuntimeErrorUray("X: строки матрицы имеют разную длину")
    x_values = array.array("d", (value for row in x_rows for value in row))
    x_pointer, x_reference = _array_pointer(x_values, ctypes.c_double)

    y_values = None
    y_pointer = _VOID_P()
    output_dim = None
    y_reference = None
    if y_data is not None:
        y_rows = _as_matrix(y_data[:sample_count], "Y")
        output_dim = len(y_rows[0])
        if any(len(row) != output_dim for row in y_rows):
            raise RuntimeErrorUray("Y: строки матрицы имеют разную длину")
        y_values = array.array("d", (value for row in y_rows for value in row))
        y_pointer, y_reference = _array_pointer(y_values, ctypes.c_double)

    return {
        "mnist_mode": 0,
        "x_pointer": x_pointer,
        "y_pointer": y_pointer,
        "image_pointer": _VOID_P(),
        "label_pointer": _VOID_P(),
        "references": (x_values, x_reference, y_values, y_reference),
        "input_dim": input_dim,
        "output_dim": output_dim,
    }


def _model_arrays(model):
    from tensor import Tensor

    layers = model["layers"]
    if not isinstance(layers, list) or not layers:
        raise RuntimeErrorUray("Нативный backend ожидает MLP-модель")
    dims = []
    weights = array.array("d")
    biases = array.array("d")
    for index, layer in enumerate(layers):
        weight_data = layer["W"].to_list() if isinstance(layer["W"], Tensor) else layer["W"]
        bias_data = layer["b"].to_list() if isinstance(layer["b"], Tensor) else layer["b"]
        matrix = _as_matrix(weight_data, f"слой {index}, W")
        bias = [float(value) for value in bias_data]
        if len(bias) != len(matrix):
            raise RuntimeErrorUray(
                f"слой {index}: число смещений не совпадает с выходами"
            )
        if index == 0:
            dims.append(len(matrix[0]))
        elif len(matrix[0]) != dims[-1]:
            raise RuntimeErrorUray(
                f"слой {index}: размер входа не совпадает с предыдущим слоем"
            )
        dims.append(len(matrix))
        weights.extend(value for row in matrix for value in row)
        biases.extend(bias)

    if dims[0] <= 0 or any(dim <= 0 for dim in dims):
        raise RuntimeErrorUray("Размеры MLP должны быть положительными")
    if "act" not in model or "out_act" not in model:
        raise RuntimeErrorUray("Модель не содержит параметры активации")
    return dims, weights, biases


def _restore_weights(model, dims, weights, biases):
    from tensor import Tensor

    weight_offset = 0
    bias_offset = 0
    for index, layer in enumerate(model["layers"]):
        input_dim = dims[index]
        output_dim = dims[index + 1]
        matrix = []
        for _ in range(output_dim):
            matrix.append(list(weights[weight_offset:weight_offset + input_dim]))
            weight_offset += input_dim
        vector = list(biases[bias_offset:bias_offset + output_dim])
        bias_offset += output_dim
        layer["W"] = Tensor(matrix)
        layer["b"] = Tensor(vector)


class NativeMLPTrainer:
    """Keep model and optimizer state in flat native-compatible buffers."""

    def __init__(self, engine, model, x_data, y_data, sample_count, optimizer):
        self.engine = engine
        self.model = model
        self.sample_count = sample_count
        self.dims, self.weights, self.biases = _model_arrays(model)
        self.data = _dataset_buffers(x_data, y_data, sample_count)
        if self.dims[0] != self.data["input_dim"]:
            raise RuntimeErrorUray(
                f"Модель ожидает {self.dims[0]} входов, "
                f"получено {self.data['input_dim']}"
            )
        if self.data["output_dim"] is not None and (
            self.dims[-1] != self.data["output_dim"]
        ):
            raise RuntimeErrorUray(
                f"Модель ожидает {self.dims[-1]} выходов, "
                f"получено {self.data['output_dim']}"
            )
        self.optimizer = {"sgd": 0, "momentum": 1, "adam": 2}[optimizer]
        self.state_mw = array.array("d", [0.0]) * len(self.weights)
        self.state_vw = array.array("d", [0.0]) * len(self.weights)
        self.state_mb = array.array("d", [0.0]) * len(self.biases)
        self.state_vb = array.array("d", [0.0]) * len(self.biases)
        self.step = ctypes.c_int(0)
        self._dims = (ctypes.c_int * len(self.dims))(*self.dims)
        self._weight_pointer, self._weight_ref = _array_pointer(
            self.weights, ctypes.c_double
        )
        self._bias_pointer, self._bias_ref = _array_pointer(
            self.biases, ctypes.c_double
        )
        self._mw_pointer, self._mw_ref = _array_pointer(
            self.state_mw, ctypes.c_double
        )
        self._vw_pointer, self._vw_ref = _array_pointer(
            self.state_vw, ctypes.c_double
        )
        self._mb_pointer, self._mb_ref = _array_pointer(
            self.state_mb, ctypes.c_double
        )
        self._vb_pointer, self._vb_ref = _array_pointer(
            self.state_vb, ctypes.c_double
        )
        self.activation = {
            "relu": 0, "leaky_relu": 1, "sigmoid": 2
        }[model["act"]]
        self.output_activation = {
            "linear": 0, "sigmoid": 1, "softmax": 2
        }[model["out_act"]]

    def train_epoch(self, indices, batch_size, learning_rate, loss_name):
        native_indices = (ctypes.c_size_t * len(indices))(*indices)
        loss_value = ctypes.c_double(0.0)
        status = self.engine.library.uray_mlp_train_epoch(
            self.sample_count,
            self.data["x_pointer"],
            self.data["y_pointer"],
            self.data["image_pointer"],
            self.data["label_pointer"],
            self.data["mnist_mode"],
            self._dims,
            len(self.dims) - 1,
            self._weight_pointer,
            self._bias_pointer,
            self._mw_pointer,
            self._vw_pointer,
            self._mb_pointer,
            self._vb_pointer,
            ctypes.byref(self.step),
            self.optimizer,
            self.activation,
            self.output_activation,
            {"mse": 0, "cce": 1, "bce": 2}[loss_name],
            float(learning_rate),
            native_indices,
            batch_size,
            ctypes.byref(loss_value),
        )
        if status:
            raise RuntimeErrorUray(
                f"Нативное обучение завершилось с кодом ошибки {status}"
            )
        return loss_value.value

    def sync_model(self):
        _restore_weights(self.model, self.dims, self.weights, self.biases)


class NativeBackend:
    def __init__(self, library):
        self.library = library
        self.library.uray_matmul.argtypes = [
            _SIZE_T, _SIZE_T, _SIZE_T, _VOID_P, _VOID_P, _VOID_P,
        ]
        self.library.uray_matmul.restype = None
        self.library.uray_all_finite.argtypes = [_SIZE_T, _VOID_P]
        self.library.uray_all_finite.restype = ctypes.c_int
        self.library.uray_mlp_train_epoch.argtypes = [
            _SIZE_T, _VOID_P, _VOID_P, _VOID_P, _VOID_P, ctypes.c_int,
            _INT_P, _SIZE_T, _VOID_P, _VOID_P, _VOID_P, _VOID_P,
            _VOID_P, _VOID_P, _INT_P, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_double, _SIZE_T_P,
            _SIZE_T, _DOUBLE_P,
        ]
        self.library.uray_mlp_train_epoch.restype = ctypes.c_int
        self.library.uray_mlp_evaluate.argtypes = [
            _SIZE_T, _VOID_P, _VOID_P, _VOID_P, _VOID_P, ctypes.c_int,
            _INT_P, _SIZE_T, _VOID_P, _VOID_P, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, _SIZE_T, _VOID_P, _DOUBLE_P,
        ]
        self.library.uray_mlp_evaluate.restype = ctypes.c_int
        self.library.uray_conv2d_forward.argtypes = [
            _SIZE_T, _SIZE_T, _SIZE_T, _SIZE_T,
            _SIZE_T, _SIZE_T, _SIZE_T, _SIZE_T, _SIZE_T,
            _VOID_P, _VOID_P, _VOID_P, _VOID_P, _VOID_P, _VOID_P,
        ]
        self.library.uray_conv2d_forward.restype = ctypes.c_int
        self.library.uray_conv2d_backward.argtypes = [
            _SIZE_T, _SIZE_T, _SIZE_T, _SIZE_T,
            _SIZE_T, _SIZE_T, _SIZE_T, _SIZE_T, _SIZE_T,
            _VOID_P, _VOID_P, _VOID_P, _VOID_P, _VOID_P, _VOID_P,
        ]
        self.library.uray_conv2d_backward.restype = ctypes.c_int
        self.library.uray_max_pool2d_forward.argtypes = [
            _SIZE_T, _SIZE_T, _SIZE_T, _SIZE_T, _SIZE_T, _SIZE_T,
            _VOID_P, _VOID_P, _SIZE_T_P,
        ]
        self.library.uray_max_pool2d_forward.restype = ctypes.c_int
        self.library.uray_max_pool2d_backward.argtypes = [
            _SIZE_T, _SIZE_T, _SIZE_T, _SIZE_T, _SIZE_T, _SIZE_T,
            _VOID_P, _SIZE_T_P, _VOID_P,
        ]
        self.library.uray_max_pool2d_backward.restype = ctypes.c_int
        self.library.uray_relu_forward.argtypes = [
            _SIZE_T, _VOID_P, _VOID_P,
        ]
        self.library.uray_relu_forward.restype = None
        self.library.uray_relu_backward.argtypes = [
            _SIZE_T, _VOID_P, _VOID_P, _VOID_P,
        ]
        self.library.uray_relu_backward.restype = None
        self.library.uray_attention_forward.argtypes = (
            [_SIZE_T] * 4 + [ctypes.c_int] + [_VOID_P] * 15
        )
        self.library.uray_attention_forward.restype = ctypes.c_int
        self.library.uray_attention_backward.argtypes = (
            [_SIZE_T] * 4 + [ctypes.c_int] + [_VOID_P] * 20
        )
        self.library.uray_attention_backward.restype = ctypes.c_int

    def matmul(self, left, right, validated=False):
        if validated:
            left_rows = left
            right_rows = right
            if (
                not isinstance(left_rows, list) or not left_rows
                or not isinstance(right_rows, list) or not right_rows
                or not isinstance(left_rows[0], list)
                or not isinstance(right_rows[0], list)
            ):
                raise RuntimeErrorUray("matmul: ожидаются непустые матрицы")
        else:
            left_rows = _as_matrix(left, "matmul")
            right_rows = _as_matrix(right, "matmul")
        if len(left_rows[0]) != len(right_rows):
            raise RuntimeErrorUray("Несовместимые размерности для matmul")
        m, k, n = len(left_rows), len(right_rows), len(right_rows[0])
        a = array.array("d", chain.from_iterable(left_rows))
        b = array.array("d", chain.from_iterable(right_rows))
        c = array.array("d", [0.0]) * (m * n)
        ap, ar = _array_pointer(a, ctypes.c_double)
        bp, br = _array_pointer(b, ctypes.c_double)
        cp, cr = _array_pointer(c, ctypes.c_double)
        if validated and (
            not self.library.uray_all_finite(len(a), ap)
            or not self.library.uray_all_finite(len(b), bp)
        ):
            raise RuntimeErrorUray("matmul: элементы должны быть конечными числами")
        self.library.uray_matmul(m, k, n, ap, bp, cp)
        return [list(c[row * n:(row + 1) * n]) for row in range(m)]

    def multihead_attention_forward(
        self, input_data, weights, biases, heads, causal
    ):
        from tensor import Tensor

        original_shape = Tensor._shape_list(input_data)
        squeeze_batch = len(original_shape) == 2
        if squeeze_batch:
            input_data = [input_data]
            batches, sequence, features = 1, *original_shape
        elif len(original_shape) == 3:
            batches, sequence, features = original_shape
        else:
            raise RuntimeErrorUray(
                "Нативный attention ожидает [sequence, features] "
                "или [batch, sequence, features]"
            )
        if (
            batches <= 0 or sequence <= 0 or features <= 0 or heads <= 0
            or features % heads != 0
            or len(weights) != 4 or len(biases) != 4
        ):
            raise RuntimeErrorUray("Неверные размерности нативного attention")

        input_values = _flat_array(input_data, "attention input")
        weight_values = [
            _flat_array(value, "attention weight") for value in weights
        ]
        bias_values = [
            _flat_array(value, "attention bias") for value in biases
        ]
        output_count = batches * sequence * features
        attention_count = batches * sequence * features
        score_count = batches * heads * sequence * sequence
        output_values = array.array("d", [0.0]) * output_count
        queries = array.array("d", [0.0]) * attention_count
        keys = array.array("d", [0.0]) * attention_count
        values = array.array("d", [0.0]) * attention_count
        probabilities = array.array("d", [0.0]) * score_count
        merged = array.array("d", [0.0]) * output_count
        buffers = [
            input_values, *weight_values, *bias_values, output_values,
            queries, keys, values, probabilities, merged,
        ]
        pointer_refs = [
            _array_pointer(buffer, ctypes.c_double) for buffer in buffers
        ]
        pointers = [pointer[0] for pointer in pointer_refs]
        status = self.library.uray_attention_forward(
            batches, sequence, features, heads, int(causal), *pointers
        )
        if status:
            raise RuntimeErrorUray(
                f"Нативный attention forward завершился с кодом ошибки {status}"
            )
        result_shape = (
            [sequence, features] if squeeze_batch
            else [batches, sequence, features]
        )
        context = {
            "shape": (batches, sequence, features),
            "squeeze_batch": squeeze_batch,
            "heads": heads,
            "causal": bool(causal),
            "input": input_values,
            "weights": weight_values,
            "queries": queries,
            "keys": keys,
            "values": values,
            "probabilities": probabilities,
            "merged": merged,
        }
        return _nested_from_flat(output_values, result_shape), context

    def multihead_attention_backward(self, gradient, context):
        batches, sequence, features = context["shape"]
        heads = context["heads"]
        input_values = context["input"]
        weight_values = context["weights"]
        gradient_values = _flat_array(gradient, "attention gradient")
        input_count = batches * sequence * features
        matrix_count = features * features
        grad_input = array.array("d", [0.0]) * input_count
        grad_weights = [
            array.array("d", [0.0]) * matrix_count for _ in range(4)
        ]
        grad_biases = [
            array.array("d", [0.0]) * features for _ in range(4)
        ]
        buffers = [
            input_values, context["queries"], context["keys"],
            context["values"], context["probabilities"], context["merged"],
            *weight_values, gradient_values, grad_input,
            *grad_weights, *grad_biases,
        ]
        pointer_refs = [
            _array_pointer(buffer, ctypes.c_double) for buffer in buffers
        ]
        pointers = [pointer[0] for pointer in pointer_refs]
        status = self.library.uray_attention_backward(
            batches, sequence, features, heads,
            int(context["causal"]), *pointers
        )
        if status:
            raise RuntimeErrorUray(
                f"Нативный attention backward завершился с кодом ошибки {status}"
            )
        input_shape = (
            [sequence, features] if context["squeeze_batch"]
            else [batches, sequence, features]
        )
        return (
            _nested_from_flat(grad_input, input_shape),
            *[
                _nested_from_flat(value, [features, features])
                for value in grad_weights
            ],
            *[
                _nested_from_flat(value, [features])
                for value in grad_biases
            ],
        )

    def create_mlp_trainer(self, model, x_data, y_data, count, optimizer):
        return NativeMLPTrainer(self, model, x_data, y_data, count, optimizer)

    def conv2d_forward(self, x_data, weight_data, bias_data, stride, padding):
        from tensor import Tensor

        x_shape = Tensor._shape_list(x_data)
        weight_shape = Tensor._shape_list(weight_data)
        if len(x_shape) != 4 or len(weight_shape) != 4:
            raise RuntimeErrorUray("Нативная conv2d ожидает NCHW-вход и OIHW-веса")
        batches, in_channels, height, width = x_shape
        out_channels, weight_channels, kernel_h, kernel_w = weight_shape
        if in_channels != weight_channels:
            raise RuntimeErrorUray("conv2d: число каналов не совпадает")
        out_h = (height + 2 * padding - kernel_h) // stride + 1
        out_w = (width + 2 * padding - kernel_w) // stride + 1
        if out_h <= 0 or out_w <= 0:
            raise RuntimeErrorUray("conv2d: ядро больше входа")
        input_values = _flat_4d_array(x_data)
        weight_values = _flat_4d_array(weight_data)
        bias_values = _flat_array(bias_data, "conv2d bias") if bias_data is not None else None
        im2col = array.array(
            "d", [0.0]
        ) * (batches * out_h * out_w * in_channels * kernel_h * kernel_w)
        weights_col = array.array(
            "d", [0.0]
        ) * (in_channels * kernel_h * kernel_w * out_channels)
        output_values = array.array(
            "d", [0.0]
        ) * (batches * out_channels * out_h * out_w)
        input_pointer, input_ref = _array_pointer(input_values, ctypes.c_double)
        weight_pointer, weight_ref = _array_pointer(weight_values, ctypes.c_double)
        bias_pointer, bias_ref = (
            _array_pointer(bias_values, ctypes.c_double)
            if bias_values is not None else (_VOID_P(), None)
        )
        output_pointer, output_ref = _array_pointer(output_values, ctypes.c_double)
        im2col_pointer, im2col_ref = _array_pointer(im2col, ctypes.c_double)
        weights_col_pointer, weights_col_ref = _array_pointer(
            weights_col, ctypes.c_double
        )
        status = self.library.uray_conv2d_forward(
            batches, in_channels, height, width,
            out_channels, kernel_h, kernel_w, stride, padding,
            input_pointer, weight_pointer, bias_pointer, output_pointer,
            im2col_pointer, weights_col_pointer,
        )
        if status:
            raise RuntimeErrorUray(
                f"Нативная conv2d завершилась с кодом ошибки {status}"
            )
        context = {
            "shape": (batches, in_channels, height, width),
            "weight_shape": (out_channels, in_channels, kernel_h, kernel_w),
            "stride": stride,
            "padding": padding,
            "im2col": im2col,
            "weights_col": weights_col,
        }
        return (
            _nested_from_flat(output_values, [batches, out_channels, out_h, out_w]),
            context,
        )

    def conv2d_relu_max_pool2d_forward(
        self, x_data, weight_data, bias_data, stride, padding,
        pool_kernel, pool_stride,
    ):
        from tensor import Tensor

        x_shape = Tensor._shape_list(x_data)
        weight_shape = Tensor._shape_list(weight_data)
        if len(x_shape) != 4 or len(weight_shape) != 4:
            raise RuntimeErrorUray(
                "Слияние conv-ReLU-pool ожидает NCHW-вход и OIHW-веса"
            )
        batches, in_channels, height, width = x_shape
        out_channels, weight_channels, kernel_h, kernel_w = weight_shape
        if in_channels != weight_channels:
            raise RuntimeErrorUray("conv2d: число каналов не совпадает")
        conv_h = (height + 2 * padding - kernel_h) // stride + 1
        conv_w = (width + 2 * padding - kernel_w) // stride + 1
        if conv_h <= 0 or conv_w <= 0:
            raise RuntimeErrorUray("conv2d: ядро больше входа")
        if pool_kernel <= 0 or pool_stride <= 0:
            raise RuntimeErrorUray(
                "max_pool2d: kernel и stride должны быть >0"
            )
        pool_h = (conv_h - pool_kernel) // pool_stride + 1
        pool_w = (conv_w - pool_kernel) // pool_stride + 1
        if pool_h <= 0 or pool_w <= 0:
            raise RuntimeErrorUray("max_pool2d: окно больше входа")

        input_values = _flat_4d_array(x_data)
        weight_values = _flat_4d_array(weight_data)
        bias_values = (
            _flat_array(bias_data, "conv2d bias")
            if bias_data is not None else None
        )
        reduction = in_channels * kernel_h * kernel_w
        rows = batches * conv_h * conv_w
        conv_count = batches * out_channels * conv_h * conv_w
        pool_count = batches * out_channels * pool_h * pool_w
        im2col = array.array("d", [0.0]) * (rows * reduction)
        weights_col = array.array("d", [0.0]) * (reduction * out_channels)
        conv_values = array.array("d", [0.0]) * conv_count
        pooled_values = array.array("d", [0.0]) * pool_count
        pool_indices = (ctypes.c_size_t * pool_count)()
        input_pointer, input_ref = _array_pointer(input_values, ctypes.c_double)
        weight_pointer, weight_ref = _array_pointer(
            weight_values, ctypes.c_double
        )
        bias_pointer, bias_ref = (
            _array_pointer(bias_values, ctypes.c_double)
            if bias_values is not None else (_VOID_P(), None)
        )
        conv_pointer, conv_ref = _array_pointer(conv_values, ctypes.c_double)
        im2col_pointer, im2col_ref = _array_pointer(im2col, ctypes.c_double)
        weights_col_pointer, weights_col_ref = _array_pointer(
            weights_col, ctypes.c_double
        )
        status = self.library.uray_conv2d_forward(
            batches, in_channels, height, width,
            out_channels, kernel_h, kernel_w, stride, padding,
            input_pointer, weight_pointer, bias_pointer, conv_pointer,
            im2col_pointer, weights_col_pointer,
        )
        if status:
            raise RuntimeErrorUray(
                f"Нативная conv2d завершилась с кодом ошибки {status}"
            )
        self.library.uray_relu_forward(
            conv_count, conv_pointer, conv_pointer
        )
        pool_pointer, pool_ref = _array_pointer(
            pooled_values, ctypes.c_double
        )
        status = self.library.uray_max_pool2d_forward(
            batches, out_channels, conv_h, conv_w, pool_kernel, pool_stride,
            conv_pointer, pool_pointer, pool_indices,
        )
        if status:
            raise RuntimeErrorUray(
                f"Нативный max_pool2d завершился с кодом ошибки {status}"
            )

        context = {
            "shape": (batches, in_channels, height, width),
            "weight_shape": (out_channels, in_channels, kernel_h, kernel_w),
            "stride": stride,
            "padding": padding,
            "im2col": im2col,
            "weights_col": weights_col,
            "activation": conv_values,
            "conv_shape": (batches, out_channels, conv_h, conv_w),
            "pool_kernel": pool_kernel,
            "pool_stride": pool_stride,
            "pool_indices": pool_indices,
        }
        return (
            _nested_from_flat(
                pooled_values, [batches, out_channels, pool_h, pool_w]
            ),
            context,
        )

    def conv2d_relu_max_pool2d_backward(
        self, context, grad_output, include_bias
    ):
        batches, out_channels, conv_h, conv_w = context["conv_shape"]
        conv_count = batches * out_channels * conv_h * conv_w
        grad_pool = _flat_4d_array(grad_output)
        grad_activation = array.array("d", [0.0]) * conv_count
        grad_conv = array.array("d", [0.0]) * conv_count
        grad_pool_pointer, grad_pool_ref = _array_pointer(
            grad_pool, ctypes.c_double
        )
        grad_activation_pointer, grad_activation_ref = _array_pointer(
            grad_activation, ctypes.c_double
        )
        activation_pointer, activation_ref = _array_pointer(
            context["activation"], ctypes.c_double
        )
        status = self.library.uray_max_pool2d_backward(
            batches, out_channels, conv_h, conv_w,
            context["pool_kernel"], context["pool_stride"],
            grad_pool_pointer, context["pool_indices"],
            grad_activation_pointer,
        )
        if status:
            raise RuntimeErrorUray(
                f"Обратный проход max_pool2d завершился с кодом ошибки {status}"
            )
        grad_conv_pointer, grad_conv_ref = _array_pointer(
            grad_conv, ctypes.c_double
        )
        self.library.uray_relu_backward(
            conv_count, activation_pointer, grad_activation_pointer,
            grad_conv_pointer,
        )
        return self._conv2d_backward_flat(
            context, grad_conv, include_bias
        )

    def conv2d_backward(self, context, grad_output, include_bias):
        return self._conv2d_backward_flat(
            context, _flat_4d_array(grad_output), include_bias
        )

    def _conv2d_backward_flat(self, context, grad_values, include_bias):
        batches, in_channels, height, width = context["shape"]
        out_channels, _, kernel_h, kernel_w = context["weight_shape"]
        stride = context["stride"]
        padding = context["padding"]
        out_h = (height + 2 * padding - kernel_h) // stride + 1
        out_w = (width + 2 * padding - kernel_w) // stride + 1
        grad_input = array.array("d", [0.0]) * (
            batches * in_channels * height * width
        )
        grad_weights = array.array("d", [0.0]) * (
            out_channels * in_channels * kernel_h * kernel_w
        )
        grad_bias = array.array("d", [0.0]) * out_channels
        grad_pointer, grad_ref = _array_pointer(grad_values, ctypes.c_double)
        im2col_pointer, im2col_ref = _array_pointer(
            context["im2col"], ctypes.c_double
        )
        weights_col_pointer, weights_col_ref = _array_pointer(
            context["weights_col"], ctypes.c_double
        )
        input_pointer, input_ref = _array_pointer(grad_input, ctypes.c_double)
        weights_pointer, weights_ref = _array_pointer(grad_weights, ctypes.c_double)
        bias_pointer, bias_ref = _array_pointer(grad_bias, ctypes.c_double)
        status = self.library.uray_conv2d_backward(
            batches, in_channels, height, width,
            out_channels, kernel_h, kernel_w, stride, padding,
            grad_pointer, im2col_pointer, weights_col_pointer,
            input_pointer, weights_pointer, bias_pointer,
        )
        if status:
            raise RuntimeErrorUray(
                f"Обратный проход conv2d завершился с кодом ошибки {status}"
            )
        input_gradient = _nested_from_flat(
            grad_input, [batches, in_channels, height, width]
        )
        weight_gradient = _nested_from_flat(
            grad_weights, [out_channels, in_channels, kernel_h, kernel_w]
        )
        result = (input_gradient, weight_gradient)
        if include_bias:
            result += (list(grad_bias),)
        return result

    def max_pool2d_forward(self, x_data, kernel, stride):
        from tensor import Tensor

        shape = Tensor._shape_list(x_data)
        if len(shape) != 4:
            raise RuntimeErrorUray("Нативный max_pool2d ожидает NCHW-вход")
        batches, channels, height, width = shape
        out_h = (height - kernel) // stride + 1
        out_w = (width - kernel) // stride + 1
        if out_h <= 0 or out_w <= 0:
            raise RuntimeErrorUray("max_pool2d: окно больше входа")
        input_values = _flat_4d_array(x_data)
        output_values = array.array(
            "d", [0.0]
        ) * (batches * channels * out_h * out_w)
        indices = (ctypes.c_size_t * len(output_values))()
        input_pointer, input_ref = _array_pointer(input_values, ctypes.c_double)
        output_pointer, output_ref = _array_pointer(output_values, ctypes.c_double)
        status = self.library.uray_max_pool2d_forward(
            batches, channels, height, width, kernel, stride,
            input_pointer, output_pointer, indices,
        )
        if status:
            raise RuntimeErrorUray(
                f"Нативный max_pool2d завершился с кодом ошибки {status}"
            )
        context = {
            "shape": shape,
            "kernel": kernel,
            "stride": stride,
            "indices": indices,
        }
        return (
            _nested_from_flat(
                output_values, [batches, channels, out_h, out_w]
            ),
            context,
        )

    def max_pool2d_backward(self, context, grad_output):
        batches, channels, height, width = context["shape"]
        kernel = context["kernel"]
        stride = context["stride"]
        grad_values = _flat_4d_array(grad_output)
        grad_input = array.array(
            "d", [0.0]
        ) * (batches * channels * height * width)
        grad_pointer, grad_ref = _array_pointer(grad_values, ctypes.c_double)
        input_pointer, input_ref = _array_pointer(grad_input, ctypes.c_double)
        status = self.library.uray_max_pool2d_backward(
            batches, channels, height, width, kernel, stride,
            grad_pointer, context["indices"], input_pointer,
        )
        if status:
            raise RuntimeErrorUray(
                f"Обратный проход max_pool2d завершился с кодом ошибки {status}"
            )
        return _nested_from_flat(
            grad_input, [batches, channels, height, width]
        )

    def relu_forward(self, data):
        from tensor import Tensor

        shape = Tensor._shape_list(data)
        values = (
            _flat_4d_array(data)
            if len(shape) == 4 else _flat_array(data, "ReLU input")
        )
        output = array.array("d", [0.0]) * len(values)
        input_pointer, input_ref = _array_pointer(values, ctypes.c_double)
        output_pointer, output_ref = _array_pointer(output, ctypes.c_double)
        self.library.uray_relu_forward(len(values), input_pointer, output_pointer)
        context = {"shape": shape, "input": values}
        return _nested_from_flat(output, shape), context

    def relu_backward(self, context, grad_output):
        shape = context["shape"]
        values = (
            _flat_4d_array(grad_output)
            if len(shape) == 4 else _flat_array(grad_output, "ReLU gradient")
        )
        if len(values) != len(context["input"]):
            raise RuntimeErrorUray("ReLU: градиент имеет неверный размер")
        output = array.array("d", [0.0]) * len(values)
        input_pointer, input_ref = _array_pointer(
            context["input"], ctypes.c_double
        )
        grad_pointer, grad_ref = _array_pointer(values, ctypes.c_double)
        output_pointer, output_ref = _array_pointer(output, ctypes.c_double)
        self.library.uray_relu_backward(
            len(values), input_pointer, grad_pointer, output_pointer
        )
        return _nested_from_flat(output, shape)

    def _predict_or_evaluate(
        self, model, x_data, y_data, loss_name, batch_size, return_predictions
    ):
        count = len(x_data)
        dims, weights, biases = _model_arrays(model)
        data = _dataset_buffers(x_data, y_data, count)
        if dims[0] != data["input_dim"]:
            raise RuntimeErrorUray(
                f"Модель ожидает {dims[0]} входов, получено {data['input_dim']}"
            )
        _dims = (ctypes.c_int * len(dims))(*dims)
        wp, wr = _array_pointer(weights, ctypes.c_double)
        bp, br = _array_pointer(biases, ctypes.c_double)
        output_dim = dims[-1]
        predictions = (
            array.array("d", [0.0]) * (count * output_dim)
            if return_predictions else None
        )
        pp, pr = (
            _array_pointer(predictions, ctypes.c_double)
            if predictions is not None else (_VOID_P(), None)
        )
        metrics = (ctypes.c_double * 2)() if y_data is not None else None
        mp = ctypes.cast(metrics, _DOUBLE_P) if metrics is not None else _DOUBLE_P()
        status = self.library.uray_mlp_evaluate(
            count,
            data["x_pointer"],
            data["y_pointer"],
            data["image_pointer"],
            data["label_pointer"],
            data["mnist_mode"],
            _dims,
            len(dims) - 1,
            wp,
            bp,
            {"relu": 0, "leaky_relu": 1, "sigmoid": 2}[model["act"]],
            {"linear": 0, "sigmoid": 1, "softmax": 2}[model["out_act"]],
            {"mse": 0, "cce": 1, "bce": 2}[loss_name],
            batch_size,
            pp,
            mp,
        )
        if status:
            raise RuntimeErrorUray(
                f"Нативная оценка завершилась с кодом ошибки {status}"
            )
        if return_predictions:
            return [
                list(predictions[start:start + output_dim])
                for start in range(0, len(predictions), output_dim)
            ]
        return [metrics[0], metrics[1]]

    def predict_mlp(self, model, x_data, limit=None, batch_size=512):
        count = len(x_data) if limit is None else min(len(x_data), limit)
        if count == 0:
            return []
        # The C evaluator receives a prefix of the dataset.
        result = self._predict_or_evaluate(
            model, x_data[:count] if isinstance(x_data, list) else x_data,
            None, "mse", batch_size, True
        )
        return result[:count]

    def evaluate_mlp(self, model, x_data, y_data, loss_name, batch_size=512):
        return self._predict_or_evaluate(
            model, x_data, y_data, loss_name, batch_size, False
        )


def load_native():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    library_path = _compile_library()
    try:
        library = ctypes.CDLL(str(library_path))
    except OSError as error:
        raise RuntimeErrorUray(
            f"Не удалось загрузить нативное ядро U-Ray: {error}"
        ) from error
    _ENGINE = NativeBackend(library)
    return _ENGINE
