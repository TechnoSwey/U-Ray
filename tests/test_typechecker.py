import unittest
import io
import tempfile
from contextlib import redirect_stderr, redirect_stdout

from errors import StaticTypeError
from main import main
from parser import parse
from typechecker import StaticChecker, check_program


class StaticCheckerTests(unittest.TestCase):
    def test_connected_dense_layers_are_checked_before_execution(self):
        source = """
import nn
first = nn.dense(2, 4)
second = nn.dense(5, 1)
x = [0.0, 1.0]
hidden = nn.forward(first, x)
output = nn.forward(second, hidden)
"""
        with self.assertRaisesRegex(
            StaticTypeError, "ожидает вход размера 5, получен размер 4"
        ):
            check_program(parse(source))

    def test_valid_dense_chain_passes(self):
        source = """
import nn
first = nn.dense(2, 4)
second = nn.dense(4, 1)
x = [0.0, 1.0]
hidden = nn.forward(first, x)
output = nn.forward(second, hidden)
"""
        check_program(parse(source))

    def test_mlp_training_input_shape_is_checked(self):
        source = """
import nn
model = nn.mlp([2, 4, 1])
X = [[0.0, 1.0, 2.0]]
Y = [[1.0]]
nn.train(model, X, Y, 1, 0.01)
"""
        with self.assertRaisesRegex(
            StaticTypeError, "ожидает X шириной 2, получена ширина 3"
        ):
            check_program(parse(source))

    def test_mlp_training_target_width_is_checked(self):
        source = """
import nn
model = nn.mlp([2, 4, 1])
X = [[0.0, 1.0]]
Y = [[0.0, 1.0]]
nn.train(model, X, Y, 1, 0.01)
"""
        with self.assertRaisesRegex(
            StaticTypeError, "ожидает Y шириной 1, получена ширина 2"
        ):
            check_program(parse(source))

    def test_dynamic_dimensions_are_left_to_runtime(self):
        source = """
import nn
func run(layer, values) {
    return nn.forward(layer, values)
}
"""
        check_program(parse(source))

    def test_mutated_lists_are_not_treated_as_having_stale_shapes(self):
        source = """
import nn
model = nn.mlp([2, 1])
X = [[0.0, 1.0]]
X[0] = [0.0, 1.0, 2.0]
Y = [[1.0]]
nn.train(model, X, Y, 1, 0.01)
"""
        # The changed row is dynamic from the checker's point of view;
        # the interpreter's runtime validation remains authoritative.
        check_program(parse(source))

    def test_non_integer_literal_layer_sizes_are_rejected_statically(self):
        with self.assertRaisesRegex(
            StaticTypeError, "размеры слоёв должны быть положительными целыми"
        ):
            check_program(parse('import nn\nmodel = nn.mlp([2, 1.5])'))

    def test_cli_checks_program_before_executing_any_statement(self):
        source = """
import nn
print("THIS MUST NOT RUN")
first = nn.dense(2, 4)
second = nn.dense(5, 1)
hidden = nn.forward(first, [0.0, 1.0])
output = nn.forward(second, hidden)
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ur", encoding="utf-8"
        ) as script:
            script.write(source)
            script.flush()
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = main([script.name])
        self.assertEqual(status, 1)
        self.assertNotIn("THIS MUST NOT RUN", stdout.getvalue())
        self.assertIn("Статическая проверка", stderr.getvalue())
        self.assertIn("строка", stderr.getvalue())

    def test_check_command_validates_without_running_source(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ur", encoding="utf-8"
        ) as script:
            script.write('print("DO NOT RUN")\n')
            script.flush()
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = main(["check", script.name])
        self.assertEqual(status, 0)
        self.assertNotIn("DO NOT RUN", stdout.getvalue())
        self.assertIn("статическая проверка пройдена", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

    def test_repl_checker_can_carry_known_symbols_forward(self):
        checker = StaticChecker()
        first = parse("import nn\nmodel = nn.mlp([2, 3, 1])\n")
        checker.symbols = checker.check(first)
        second = parse("X = [[1, 2, 3]]\nY = [[0]]\nnn.train(model, X, Y, 1, 0.1)\n")
        with self.assertRaisesRegex(StaticTypeError, "ожидает X шириной 2"):
            checker.check(second)


if __name__ == "__main__":
    unittest.main()
