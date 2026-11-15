import json
import unittest
from pathlib import Path


class VSCodeExtensionTests(unittest.TestCase):
    def test_extension_manifests_are_valid_and_register_ur_files(self):
        root = Path(__file__).resolve().parents[1] / "vscode" / "uray-language"
        package = json.loads((root / "package.json").read_text(encoding="utf-8"))
        grammar = json.loads(
            (root / "syntaxes" / "uray.tmLanguage.json").read_text(
                encoding="utf-8"
            )
        )
        language = json.loads(
            (root / "language-configuration.json").read_text(encoding="utf-8")
        )

        self.assertEqual(package["contributes"]["languages"][0]["id"], "uray")
        self.assertIn(".ur", package["contributes"]["languages"][0]["extensions"])
        self.assertEqual(grammar["scopeName"], "source.uray")
        self.assertTrue(grammar["patterns"])
        self.assertIn("comments", grammar["repository"])
        self.assertEqual(language["comments"]["lineComment"], "//")


if __name__ == "__main__":
    unittest.main()
