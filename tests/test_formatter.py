import unittest
import io
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from formatter import format_source
from main import main
from parser import parse


class FormatterTests(unittest.TestCase):
    def test_normalizes_spacing_indentation_and_keeps_comments(self):
        source = (
            'func  add(a,b){ // header\n'
            'return a+b  // expression\n'
            '}\n'
            'text="// not a comment"\n'
        )
        expected = (
            'func add(a, b) {  // header\n'
            '    return a + b  // expression\n'
            '}\n'
            'text = "// not a comment"\n'
        )
        formatted = format_source(source)
        self.assertEqual(formatted, expected)
        parse(formatted)

    def test_expands_single_line_blocks_without_reformatting_dictionary_literals(self):
        source = (
            'func  identity(x){return x}\n'
            'value={"url":"https://host/path"}\n'
        )
        expected = (
            'func identity(x) {\n'
            '    return x\n'
            '}\n'
            'value = { "url": "https://host/path" }\n'
        )
        formatted = format_source(source)
        self.assertEqual(formatted, expected)
        parse(formatted)

    def test_expands_type_match_and_case_blocks(self):
        source = (
            "type Point{x y}\n"
            "match Point(1,2){Point(x,y){print(x)}}\n"
        )
        expected = (
            "type Point {\n"
            "    x y\n"
            "}\n"
            "match Point(1, 2) {\n"
            "    Point(x, y) {\n"
            "        print(x)\n"
            "    }\n"
            "}\n"
        )
        formatted = format_source(source)
        self.assertEqual(formatted, expected)
        parse(formatted)

    def test_formatter_is_idempotent_for_repository_programs(self):
        root = Path(__file__).resolve().parents[1]
        paths = sorted((root / "examples").glob("*.ur"))
        paths += sorted((root / "tests").glob("*.ur"))
        self.assertGreaterEqual(len(paths), 50)
        for path in paths:
            with self.subTest(path=path.name):
                original = path.read_text(encoding="utf-8")
                once = format_source(original)
                self.assertEqual(format_source(once), once)
                parse(once)

    def test_fmt_cli_defaults_to_stdout_and_has_safe_write_and_check_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.ur"
            original = "value=1\nprint(value)\n"
            expected = "value = 1\nprint(value)\n"
            path.write_text(original, encoding="utf-8")

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["fmt", str(path)]), 0)
            self.assertEqual(output.getvalue(), expected)
            self.assertEqual(path.read_text(encoding="utf-8"), original)

            errors = io.StringIO()
            with redirect_stderr(errors):
                self.assertEqual(main(["fmt", "--check", str(path)]), 1)
            self.assertIn("требуется форматирование", errors.getvalue())

            self.assertEqual(main(["fmt", "--write", str(path)]), 0)
            self.assertEqual(path.read_text(encoding="utf-8"), expected)
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["fmt", "--check", str(path)]), 0)


if __name__ == "__main__":
    unittest.main()
