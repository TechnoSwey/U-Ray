import copy
import math
import tempfile
import unittest
from pathlib import Path

from autograd import GradTensor
from modules import ModuleRegistry
from tensor import set_backend


def _flatten_nested(value):
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_flatten_nested(item))
        return result
    return [value]


def _perturb(value, path, amount):
    result = copy.deepcopy(value)
    target = result
    for index in path[:-1]:
        target = target[index]
    target[path[-1]] += amount
    return result


def _finite_difference(fn, value, path, epsilon=1e-5):
    plus = fn(_perturb(value, path, epsilon))
    minus = fn(_perturb(value, path, -epsilon))
    return (plus - minus) / (2.0 * epsilon)


class AutogradTests(unittest.TestCase):
    def setUp(self):
        set_backend("python")
        self.nn = ModuleRegistry().get("nn")

    def tearDown(self):
        set_backend("python")

    def test_broadcast_graph_accumulates_gradients(self):
        x = GradTensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        bias = GradTensor([0.5, -1.0], requires_grad=True)
        loss = x.add(bias).mul(x).mean()
        loss.backward()
        self.assertAlmostEqual(loss.item(), 6.5)
        self.assertEqual(x.grad, [[0.625, 0.75], [1.625, 1.75]])
        self.assertEqual(bias.grad, [1.0, 1.5])

    def test_tensor_arithmetic_operators_track_gradients(self):
        x = GradTensor([2.0, -1.0], requires_grad=True)
        loss = x.pow(2).sub(x.scalar_mul(3.0)).div(2.0).mean()
        loss.backward()
        self.assertAlmostEqual(loss.item(), 0.5)
        self.assertEqual(x.grad, [0.25, -1.25])

    def test_backward_releases_computation_graph(self):
        x = GradTensor([1.0, 2.0], requires_grad=True)
        loss = x.mul(x).mean()
        loss.backward()
        self.assertEqual(x.grad, [1.0, 2.0])
        self.assertEqual(loss._parents, ())
        self.assertIsNone(loss._backward_fn)

    def test_matmul_gradient_matches_finite_difference(self):
        left = [[0.2, -0.4], [0.7, 0.3]]
        right = [[-0.1, 0.8], [0.5, -0.2]]

        def objective(value):
            return GradTensor(value).matmul(GradTensor(right)).mean().item()

        x = GradTensor(left, requires_grad=True)
        loss = x.matmul(GradTensor(right)).mean()
        loss.backward()
        self.assertAlmostEqual(
            x.grad[0][1], _finite_difference(objective, left, [0, 1]), places=6
        )

    def test_cross_entropy_logits_gradient(self):
        logits = GradTensor([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]], True)
        target = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        loss = logits.cross_entropy(target)
        loss.backward()
        self.assertTrue(math.isfinite(loss.item()))
        self.assertAlmostEqual(sum(logits.grad[0]), 0.0, places=12)
        self.assertAlmostEqual(sum(logits.grad[1]), 0.0, places=12)
        self.assertLess(logits.grad[0][0], 0.0)
        self.assertLess(logits.grad[1][2], 0.0)

    def test_probability_losses_preserve_existing_semantics(self):
        probabilities = GradTensor([0.8, 0.2], True)
        loss = probabilities.categorical_cross_entropy([1.0, 0.0])
        self.assertAlmostEqual(loss.item(), -math.log(0.8) / 2.0)
        loss.backward()
        self.assertAlmostEqual(probabilities.grad[0], -1.0 / 1.6)
        self.assertEqual(probabilities.grad[1], 0.0)

        binary = GradTensor([0.8], True)
        binary_loss = binary.binary_cross_entropy_probabilities([1.0])
        binary_loss.backward()
        self.assertAlmostEqual(binary_loss.item(), -math.log(0.8))
        self.assertAlmostEqual(binary.grad[0], -1.0 / 0.8)

    def test_conv2d_gradients_match_finite_difference(self):
        image = [[[[0.2, -0.1, 0.5], [0.4, 0.7, -0.3], [0.9, 0.1, 0.6]]]]
        weights = [[[[0.3, -0.2], [0.1, 0.5]]]]

        def objective(value):
            return GradTensor(image).conv2d(GradTensor(value)).mean().item()

        x = GradTensor(image, requires_grad=True)
        w = GradTensor(weights, requires_grad=True)
        loss = x.conv2d(w).mean()
        loss.backward()
        self.assertAlmostEqual(
            w.grad[0][0][1][0],
            _finite_difference(objective, weights, [0, 0, 1, 0]),
            places=6,
        )
        self.assertEqual(x.grad[0][0][0][0], weights[0][0][0][0] / 4.0)

    def test_max_pool_routes_gradient_to_maximum(self):
        x = GradTensor([[[[1.0, 2.0], [3.0, 4.0]]]], True)
        y = x.max_pool2d(2)
        y.sum_all().backward()
        self.assertEqual(y.data, [[[[4.0]]]])
        self.assertEqual(x.grad, [[[[0.0, 0.0], [0.0, 1.0]]]])

    def test_sequential_layers_train_with_generic_autograd(self):
        self.nn.members["rand_seed"].fn([7])
        first = self.nn.members["linear_layer"].fn([2, 4])
        activation = self.nn.members["activation_layer"].fn(["tanh"])
        last = self.nn.members["linear_layer"].fn([4, 1])
        model = self.nn.members["sequential"].fn([[first, activation, last]])
        params = self.nn.members["parameters"].fn([model])
        x = self.nn.members["tensor"].fn([[1.0, -1.0]])
        target = self.nn.members["tensor"].fn([[1.0]])
        prediction = self.nn.members["forward_layers"].fn([model, x])
        before = prediction.mse_loss(target).item()

        self.nn.members["backward"].fn([prediction.mse_loss(target)])
        self.nn.members["sgd_update"].fn([params, 0.05])
        self.nn.members["zero_grad"].fn([params])

        after_prediction = self.nn.members["forward_layers"].fn([model, x])
        after = after_prediction.mse_loss(target).item()
        self.assertTrue(math.isfinite(before))
        self.assertTrue(math.isfinite(after))
        self.assertNotEqual(before, after)

    def test_cnn_layers_compose_and_backpropagate(self):
        self.nn.members["rand_seed"].fn([11])
        model = self.nn.members["sequential"].fn([[
            self.nn.members["conv2d_layer"].fn([1, 2, 3, 1, 1]),
            self.nn.members["activation_layer"].fn(["relu"]),
            self.nn.members["max_pool2d_layer"].fn([2, 2]),
            self.nn.members["flatten_layer"].fn([]),
            self.nn.members["linear_layer"].fn([8, 3]),
        ]])
        image = [[[[float(row * 4 + col) for col in range(4)] for row in range(4)]]]
        logits = self.nn.members["forward_layers"].fn(
            [model, GradTensor(image)]
        )
        self.assertEqual(logits.shape(), [1, 3])
        loss = logits.cross_entropy([[0.0, 1.0, 0.0]])
        loss.backward()
        params = self.nn.members["parameters"].fn([model])
        self.assertEqual(len(params), 4)
        self.assertTrue(all(param.grad is not None for param in params))

    def test_sequential_model_save_load_preserves_output(self):
        self.nn.members["rand_seed"].fn([13])
        model = self.nn.members["sequential"].fn([[
            self.nn.members["conv2d_layer"].fn([1, 2, 3, 1, 1]),
            self.nn.members["activation_layer"].fn(["relu"]),
            self.nn.members["max_pool2d_layer"].fn([2, 2]),
            self.nn.members["flatten_layer"].fn([]),
            self.nn.members["linear_layer"].fn([8, 3]),
        ]])
        image = GradTensor([[[[float(i + j) for j in range(4)] for i in range(4)]]])
        before = self.nn.members["forward_layers"].fn([model, image]).data
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "cnn.json")
            self.nn.members["save_layers"].fn([model, path])
            loaded = self.nn.members["load_layers"].fn([path])
        after = self.nn.members["forward_layers"].fn([loaded, image]).data
        self.assertEqual(before, after)
        self.assertEqual(len(self.nn.members["parameters"].fn([loaded])), 4)

    def test_batch_norm_gradients_and_running_statistics(self):
        values = [[0.2, 1.0], [1.4, -0.5], [2.2, 3.0]]
        probe = [[0.3, -0.2], [1.1, 0.7], [-0.4, 0.8]]

        def objective(data, gamma):
            layer = self.nn.members["batch_norm_layer"].fn([2, 0.2, 1e-5])
            layer["gamma"]._data = gamma
            output = GradTensor(data).batch_norm(
                layer["gamma"], layer["beta"], layer["running_mean"],
                layer["running_var"], True, layer["momentum"],
                layer["epsilon"], layer["axis"],
            )
            return output.mul(probe).sum_all().item()

        layer = self.nn.members["batch_norm_layer"].fn([2, 0.2, 1e-5])
        x = GradTensor(values, requires_grad=True)
        output = x.batch_norm(
            layer["gamma"], layer["beta"], layer["running_mean"],
            layer["running_var"], True, layer["momentum"],
            layer["epsilon"], layer["axis"],
        )
        output.mul(probe).sum_all().backward()
        self.assertAlmostEqual(
            x.grad[1][0],
            _finite_difference(
                lambda data: objective(data, [1.0, 1.0]), values, [1, 0]
            ),
            places=5,
        )
        gamma = [1.0, 1.0]
        numeric_gamma = _finite_difference(
            lambda value: objective(values, value), gamma, [1]
        )
        self.assertAlmostEqual(layer["gamma"].grad[1], numeric_gamma, places=5)
        self.assertAlmostEqual(layer["running_mean"][0], 0.2 * (3.8 / 3.0))
        self.assertAlmostEqual(layer["running_mean"][1], 0.2 * (3.5 / 3.0))
        self.assertNotEqual(layer["running_var"], [1.0, 1.0])
        self.assertEqual(layer["beta"].grad, [
            sum(row[0] for row in probe),
            sum(row[1] for row in probe),
        ])

    def test_batch_norm_numpy_matches_python_for_channel_axis(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is optional")
        data = [
            [[[1.0, 2.0]], [[3.0, 4.0]]],
            [[[5.0, 6.0]], [[7.0, 8.0]]],
        ]
        results = []
        for backend in ("python", "numpy"):
            set_backend(backend)
            layer = self.nn.members["batch_norm_layer"].fn([2])
            x = GradTensor(data, requires_grad=True)
            output = x.batch_norm(
                layer["gamma"], layer["beta"], layer["running_mean"],
                layer["running_var"], True, layer["momentum"],
                layer["epsilon"], 1,
            )
            output.mean().backward()
            results.append((output.data, x.grad, layer["gamma"].grad))
        for left, right in zip(
            _flatten_nested(results[0]), _flatten_nested(results[1])
        ):
            self.assertAlmostEqual(left, right, places=12)

    def test_dropout_and_residual_layer_modes_and_serialization(self):
        self.nn.members["rand_seed"].fn([31])
        model = self.nn.members["sequential"].fn([[
            self.nn.members["batch_norm_layer"].fn([2]),
            self.nn.members["dropout_layer"].fn([0.5]),
            self.nn.members["residual_layer"].fn([[
                self.nn.members["linear_layer"].fn([2, 2]),
                self.nn.members["activation_layer"].fn(["tanh"]),
                self.nn.members["linear_layer"].fn([2, 2]),
            ]]),
            self.nn.members["linear_layer"].fn([2, 2]),
        ]])
        x = GradTensor([[1.0, 0.0], [0.0, 1.0]])
        before_stats = list(model["layers"][0]["running_mean"])
        training_output = self.nn.members["forward_layers"].fn([model, x, True])
        self.assertEqual(training_output.shape(), [2, 2])
        self.assertNotEqual(model["layers"][0]["running_mean"], before_stats)
        self.nn.members["rand_seed"].fn([31])
        dropout_model = self.nn.members["sequential"].fn([[
            self.nn.members["dropout_layer"].fn([0.5])
        ]])
        dropped = self.nn.members["forward_layers"].fn([
            dropout_model, x, True,
        ])
        flat_dropped = _flatten_nested(dropped.data)
        self.assertIn(0.0, flat_dropped)
        self.assertIn(2.0, flat_dropped)

        output_before = self.nn.members["forward_layers"].fn([model, x]).data
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "regularized.json")
            self.nn.members["save_layers"].fn([model, path])
            loaded = self.nn.members["load_layers"].fn([path])
        output_after = self.nn.members["forward_layers"].fn([loaded, x]).data
        self.assertEqual(output_after, output_before)
        self.assertEqual(len(self.nn.members["parameters"].fn([model])), 8)

    def test_training_updates_batch_norm_and_uses_regularization(self):
        self.nn.members["rand_seed"].fn([17])
        model = self.nn.members["sequential"].fn([[
            self.nn.members["batch_norm_layer"].fn([2]),
            self.nn.members["dropout_layer"].fn([0.1]),
            self.nn.members["residual_layer"].fn([[
                self.nn.members["linear_layer"].fn([2, 2]),
                self.nn.members["activation_layer"].fn(["relu"]),
                self.nn.members["linear_layer"].fn([2, 2]),
            ]]),
        ]])
        x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, -1.0]]
        y = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        history = self.nn.members["train_layers"].fn([
            model, x, y, 2, 0.01, 2, "cce", "adam", False,
        ])
        self.assertEqual(len(history), 2)
        self.assertTrue(all(math.isfinite(value) for value in history))
        self.assertNotEqual(model["layers"][0]["running_mean"], [0.0, 0.0])
        running_mean = list(model["layers"][0]["running_mean"])
        metrics = self.nn.members["evaluate_layers"].fn([
            model, x, y, "cce",
        ])
        self.assertEqual(len(metrics), 2)
        self.assertEqual(model["layers"][0]["running_mean"], running_mean)

    def test_numpy_conv2d_matches_python_when_available(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is optional")
        image = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]
        weight = [[[[0.5, -0.25], [0.75, 0.1]]]]
        outputs = []
        gradients = []
        for backend in ("python", "numpy"):
            set_backend(backend)
            x = GradTensor(image, True)
            w = GradTensor(weight, True)
            y = x.conv2d(w, [0.2], stride=1, padding=1)
            outputs.append(y.data)
            y.mean().backward()
            gradients.append((x.grad, w.grad))
        for left, right in zip(
            [value for value in _flatten_nested(outputs[0])],
            [value for value in _flatten_nested(outputs[1])],
        ):
            self.assertAlmostEqual(left, right, places=12)
        for left_group, right_group in zip(gradients[0], gradients[1]):
            for left, right in zip(
                _flatten_nested(left_group), _flatten_nested(right_group)
            ):
                self.assertAlmostEqual(left, right, places=12)

    def test_numpy_matmul_graph_matches_python_when_available(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is optional")
        left = [[0.2, -0.4, 0.7], [0.5, 0.1, -0.3]]
        weight = [[0.1, 0.2, 0.3], [-0.5, 0.4, 0.8]]
        results = []
        for backend in ("python", "numpy"):
            set_backend(backend)
            x = GradTensor(left, True)
            w = GradTensor(weight, True)
            loss = x.matmul(w.transpose()).mean()
            loss.backward()
            results.append((loss.item(), x.grad, w.grad))
        self.assertAlmostEqual(results[0][0], results[1][0], places=12)
        for left_group, right_group in zip(results[0][1:], results[1][1:]):
            for left_value, right_value in zip(
                _flatten_nested(left_group), _flatten_nested(right_group)
            ):
                self.assertAlmostEqual(left_value, right_value, places=12)


if __name__ == "__main__":
    unittest.main()
