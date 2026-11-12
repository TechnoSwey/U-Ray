# modules.py

import json
import os
import math as _pymath
import random as _pyrandom

import values
from errors import RuntimeErrorUray as R
from values import DictValue, Builtin, Module, Function
from tensor import Tensor


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
                except ValueError as e:
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
            ensure_number("relu", args[0])
            return max(0.0, float(args[0]))

        def leaky_relu(args):
            if len(args) != 1: raise R("nn.leaky_relu ожидает 1 аргумент")
            ensure_number("leaky_relu", args[0])
            x = float(args[0])
            return x if x > 0.0 else LEAKY_SLOPE * x

        def sigmoid(args):
            if len(args) != 1: raise R("nn.sigmoid ожидает 1 аргумент")
            ensure_number("sigmoid", args[0])
            return _stable_sigmoid(args[0])

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
            v = args[0]
            ensure_vector("softmax", v)
            return _softmax_vec(v)

        def mse(args):
            if len(args) != 2: raise R("nn.mse ожидает 2 аргумента")
            yhat, y = args
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
            if not isinstance(layer, DictValue) or "W" not in layer or "b" not in layer:
                raise R("nn.forward ожидает слой")
            W = layer["W"]
            b = layer["b"]
            x_list = x.to_list() if isinstance(x, Tensor) else list(x)
            W_list = W.to_list()
            b_list = b.to_list()
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
            if not isinstance(layer, DictValue) or "W" not in layer or "b" not in layer:
                raise R("nn.forward_dense ожидает слой")
            W = layer["W"]
            b = layer["b"]
            x_list = x.to_list() if isinstance(x, Tensor) else list(x)
            W_list = W.to_list()
            b_list = b.to_list()
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
            if not isinstance(layer, DictValue) or "W" not in layer or "b" not in layer:
                raise R("nn.backward_dense ожидает слой")
            W = layer["W"]
            x_list = x.to_list() if isinstance(x, Tensor) else list(x)
            W_list = W.to_list()
            gz = grad_z.to_list() if isinstance(grad_z, Tensor) else list(grad_z)
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
            x_list = x.to_list() if isinstance(x, Tensor) else list(x)
            cache = [list(x_list)]
            cur = list(x_list)
            n = len(layers)
            for k in range(n):
                layer = layers[k]
                W_list = layer["W"].to_list() if isinstance(layer["W"], Tensor) else layer["W"]
                b_list = layer["b"].to_list() if isinstance(layer["b"], Tensor) else layer["b"]
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
            if not isinstance(model, DictValue) or "layers" not in model:
                raise R("nn.backward_mlp ожидает модель")
            layers = model["layers"]
            act = model["act"]
            out_act = model["out_act"]
            grad_list = grad_out.to_list() if isinstance(grad_out, Tensor) else list(grad_out)
            n = len(layers)
            expected = 1 + 2 * n
            if len(cache) != expected:
                raise R(f"nn.backward_mlp: cache должен быть длины {expected}")
            grads_rev = []
            grad = list(grad_list)
            for k in range(n - 1, -1, -1):
                x_in = cache[2 * k]
                z = cache[2 * k + 1]
                if k == n - 1:
                    if skip_last_act or out_act == "softmax" or out_act == "linear":
                        grad_z = list(grad)
                    else:
                        grad_z = []
                        for zi, gi in zip(z, grad):
                            s = _stable_sigmoid(zi)
                            grad_z.append(gi * s * (1.0 - s))
                else:
                    grad_z = _act_grad(act, z, grad)
                layer = layers[k]
                W_list = layer["W"].to_list() if isinstance(layer["W"], Tensor) else layer["W"]
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

        def predict(args):
            if len(args) != 2: raise R("nn.predict ожидает model и X")
            model, X = args
            if not isinstance(X, list): raise R("nn.predict ожидает список точек")
            res = []
            for x in X:
                cache = forward_mlp([model, x])
                res.append(cache[-1])
            return res

        def evaluate(args):
            if len(args) != 4: raise R("nn.evaluate ожидает model, X, Y, loss")
            model, X, Y, loss_name = args
            if not isinstance(X, list) or not isinstance(Y, list): raise R("nn.evaluate ожидает списки")
            if len(X) != len(Y): raise R("nn.evaluate: X и Y разной длины")
            total_loss = 0.0
            correct = 0
            for i in range(len(X)):
                yhat = forward_mlp([model, X[i]])[-1]
                y = Y[i]
                total_loss += _compute_loss(loss_name, yhat, y)
                pred = 0
                for k in range(1, len(yhat)):
                    if yhat[k] > yhat[pred]: pred = k
                true_best = 0
                for k in range(1, len(y)):
                    if y[k] > y[true_best]: true_best = k
                if pred == true_best: correct += 1
            return [total_loss / len(X), correct / len(X)]

        def train(args):
            if len(args) < 5 or len(args) > 8:
                raise R("nn.train ожидает model, X, Y, epochs, lr (опционально batch_size, loss, optimizer)")
            model, X, Y, epochs, lr = args[0], args[1], args[2], args[3], args[4]
            batch_size = args[5] if len(args) > 5 else 32
            loss_name = args[6] if len(args) > 6 else "bce"
            opt_name = args[7] if len(args) > 7 else "adam"
            if not isinstance(X, list) or not isinstance(Y, list):
                raise R("nn.train ожидает списки X и Y")
            if len(X) != len(Y): raise R("nn.train: X и Y разной длины")
            if not isinstance(epochs, int) or epochs <= 0: raise R("nn.train: epochs > 0")
            if not isinstance(batch_size, int) or batch_size <= 0: raise R("nn.train: batch_size > 0")
            state = None
            if opt_name == "adam": state = adam_init([model])
            elif opt_name == "momentum": state = momentum_init([model])
            elif opt_name != "sgd": raise R(f"nn.train: неизвестный оптимизатор {opt_name!r}")
            history = []
            n = len(X)
            indices = list(range(n))
            for epoch in range(epochs):
                _pyrandom.shuffle(indices)
                epoch_loss = 0.0
                num_batches = 0
                for i in range(0, n, batch_size):
                    batch_indices = indices[i:i+batch_size]
                    count = len(batch_indices)
                    accum_grads = None
                    batch_loss = 0.0
                    for idx in batch_indices:
                        cache = forward_mlp([model, X[idx]])
                        yhat = cache[-1]
                        y = Y[idx]
                        batch_loss += _compute_loss(loss_name, yhat, y)
                        grad_out = _compute_grad(loss_name, yhat, y)
                        grads = backward_mlp([model, cache, grad_out, True])
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
                    num_batches += 1
                history.append(epoch_loss / num_batches)
            return history

        return Module("nn", {
            "shape": Builtin("shape", shape),
            "relu": Builtin("relu", relu),
            "leaky_relu": Builtin("leaky_relu", leaky_relu),
            "sigmoid": Builtin("sigmoid", sigmoid),
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
            "predict": Builtin("predict", predict),
            "evaluate": Builtin("evaluate", evaluate),
            "train": Builtin("train", train),
        })
