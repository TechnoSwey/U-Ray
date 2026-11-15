import unittest

from interpreter import Interpreter
from parser import parse
from repl_completion import completion_candidates
from values import format_value


class ReplCompletionTests(unittest.TestCase):
    def setUp(self):
        self.interpreter = Interpreter()

    def test_completes_keywords_builtins_and_modules(self):
        self.assertIn("print", completion_candidates("pri", self.interpreter))
        self.assertIn("nn", completion_candidates("nn", self.interpreter))
        self.assertIn("nn.mlp", completion_candidates("nn.m", self.interpreter))

    def test_completed_builtin_has_a_readable_repl_representation(self):
        self.interpreter.run(parse("import nn"))
        value = self.interpreter.eval_expr(parse("nn.mlp").statements[0].expr)
        self.assertEqual(format_value(value), "<builtin mlp>")

    def test_completes_variables_struct_fields_and_methods(self):
        self.interpreter.run(parse("""
import nn
type Point {
    x
    y
    func get_x() {
        return self.x
    }
}
point = Point(3, 4)
"""))
        self.assertIn("point", completion_candidates("poi", self.interpreter))
        members = completion_candidates("point.", self.interpreter)
        self.assertIn("point.x", members)
        self.assertIn("point.y", members)
        self.assertIn("point.get_x", members)

    def test_completes_dictionary_keys_without_evaluating_source(self):
        self.interpreter.run(parse('settings = {"backend": "python"}'))
        self.assertEqual(
            completion_candidates("settings.b", self.interpreter),
            ["settings.backend"],
        )


if __name__ == "__main__":
    unittest.main()
