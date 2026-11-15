import re
import unittest
from pathlib import Path

from lexer import KEYWORDS


ROOT = Path(__file__).resolve().parents[1]


class TermuxSyntaxTests(unittest.TestCase):
    def test_nano_syntax_targets_ur_and_highlights_core_tokens(self):
        config = (ROOT / "termux/nano/uray.nanorc").read_text(encoding="utf-8")
        self.assertIn('syntax "uray" "\\.ur$"', config)
        self.assertIn('color lightblack "//.*$"', config)
        self.assertIn('start="\\"" end="\\""', config)
        self.assertIn("[0-9]+", config)
        for keyword in KEYWORDS:
            self.assertIn(keyword, config)

    def test_vim_detects_ur_files_and_defines_language_scopes(self):
        detector = (
            ROOT / "termux/vim/ftdetect/uray.vim"
        ).read_text(encoding="utf-8")
        syntax = (ROOT / "termux/vim/syntax/uray.vim").read_text(
            encoding="utf-8"
        )
        self.assertRegex(detector, r"\*\.ur\s+setfiletype uray")
        for group in (
            "urayKeyword", "urayNumber", "urayString", "urayComment",
            "urayFunction", "urayModule", "urayOperator",
        ):
            self.assertIn(group, syntax)
        for keyword in KEYWORDS:
            self.assertRegex(syntax, rf"\b{re.escape(keyword)}\b")

    def test_micro_syntax_declares_uray_rules(self):
        syntax = (ROOT / "termux/micro/uray.yaml").read_text(encoding="utf-8")
        self.assertIn("filetype: uray", syntax)
        self.assertIn('filename: "\\\\.ur$"', syntax)
        for group in (
            "statement", "type.keyword", "constant.bool", "constant.number",
            "constant.string", "comment", "symbol.operator",
        ):
            self.assertIn(f"- {group}:", syntax)
        for keyword in KEYWORDS:
            self.assertIn(keyword, syntax)

    def test_termux_installation_instructions_cover_supported_editors(self):
        instructions = (ROOT / "termux/README.md").read_text(encoding="utf-8")
        self.assertIn("pkg install nano vim neovim micro", instructions)
        self.assertIn("nano/uray.nanorc", instructions)
        self.assertIn("vim/ftdetect/uray.vim", instructions)
        self.assertIn("micro/uray.yaml", instructions)
        self.assertIn("Neovim", instructions)


if __name__ == "__main__":
    unittest.main()
