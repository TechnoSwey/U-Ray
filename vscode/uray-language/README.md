# U-Ray language support for VS Code

This local extension adds `.ur` language detection, TextMate syntax
highlighting, `//` comments, bracket matching, and automatic closing pairs.
It is not published to the Visual Studio Marketplace.

## Install locally

From the U-Ray repository root on Linux or macOS:

```bash
mkdir -p ~/.vscode/extensions/technoswey.uray-language-0.1.0
cp -R vscode/uray-language/. \
  ~/.vscode/extensions/technoswey.uray-language-0.1.0/
```

Restart VS Code or run **Developer: Reload Window**. Files ending in `.ur`
will use U-Ray highlighting automatically. On Windows, copy the extension
folder into `%USERPROFILE%\.vscode\extensions\technoswey.uray-language-0.1.0`.

The extension contains no JavaScript runtime and sends no code anywhere.
