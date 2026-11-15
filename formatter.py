"""A conservative, comment-preserving formatter for U-Ray source files."""

from ast_nodes import Block, Match, TypeDecl
from parser import parse

_OPERATORS = {
    "=", "+", "-", "*", "/", "%", "**",
    "==", "!=", "<", ">", "<=", ">=",
}
_OPEN = {"(", "[", "{"}
_CLOSE = {")", "]", "}"}
_PREFIX_WORDS = {"return", "throw"}
_CONTROL_PARENS = {"if", "while", "for", "match"}


def _split_comment(line):
    in_string = False
    escaped = False
    for index, char in enumerate(line):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "/" and index + 1 < len(line) and line[index + 1] == "/":
            return line[:index], line[index:]
    return line, ""


def _scan_line(code):
    tokens = []
    index = 0
    while index < len(code):
        char = code[index]
        if char.isspace():
            index += 1
            continue
        start = index
        if char == '"':
            index += 1
            escaped = False
            while index < len(code):
                current = code[index]
                index += 1
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    break
            tokens.append(code[start:index])
            continue
        if char.isalpha() or char == "_":
            index += 1
            while index < len(code) and (
                code[index].isalnum() or code[index] == "_"
            ):
                index += 1
            tokens.append(code[start:index])
            continue
        if char.isdigit():
            index += 1
            while index < len(code) and code[index].isdigit():
                index += 1
            if index + 1 < len(code) and code[index] == "." \
                    and code[index + 1].isdigit():
                index += 1
                while index < len(code) and code[index].isdigit():
                    index += 1
            tokens.append(code[start:index])
            continue
        pair = code[index:index + 2]
        if pair in {"==", "!=", "<=", ">=", "**"}:
            tokens.append(pair)
            index += 2
        else:
            tokens.append(char)
            index += 1
    return tokens


def _is_word(token):
    return bool(token) and (
        token[0].isalpha() or token[0] == "_" or token[0].isdigit()
        or token[0] == '"'
    )


def _unary_sign(tokens, index):
    if tokens[index] not in {"+", "-"}:
        return False
    if index == 0:
        return True
    previous = tokens[index - 1]
    return previous in _OPEN | _OPERATORS | {",", ":", ";"} \
        or previous in _PREFIX_WORDS


def _inside_square_brackets(tokens, stop):
    depth = 0
    for token in tokens[:stop]:
        if token == "[":
            depth += 1
        elif token == "]" and depth:
            depth -= 1
    return depth > 0


def _format_tokens(tokens):
    output = ""
    for index, token in enumerate(tokens):
        previous = tokens[index - 1] if index else None
        unary = _unary_sign(tokens, index)
        space = False

        if previous is not None:
            if token in {".", ",", ";", ")", "]"}:
                space = False
            elif token == ":":
                space = False
            elif previous in {".", "(", "["}:
                space = False
            elif previous == "{":
                space = token != "}"
            elif previous in {",", ";"}:
                space = True
            elif previous == ":":
                space = not _inside_square_brackets(tokens, index - 1)
            elif token == "(":
                space = previous in _CONTROL_PARENS | _OPERATORS | {",", ";", ":"}
            elif token == "[":
                space = not (
                    _is_word(previous) or previous in {")", "]", "}"}
                )
            elif token == "{":
                space = True
            elif token == "}":
                space = previous != "{"
            elif token in _OPERATORS or previous in _OPERATORS:
                if unary:
                    space = previous not in _OPEN | {".", ","}
                else:
                    space = True
            elif _is_word(previous) and _is_word(token):
                space = True
            elif previous == "}" and _is_word(token):
                space = True
            elif _is_word(previous) and token == "{":
                space = True
        if space:
            output += " "
        output += token
    return output


def _block_positions(program):
    opens = set()
    closes = set()
    seen = set()

    def visit(value):
        if id(value) in seen:
            return
        if isinstance(value, (Block, Match, TypeDecl)):
            seen.add(id(value))
            if value.start is not None and value.end is not None:
                opens.add(value.start)
                closes.add(value.end)
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif hasattr(value, "__dict__"):
            seen.add(id(value))
            for item in vars(value).values():
                visit(item)

    visit(program)
    return opens, closes


def _expand_inline_blocks(source, opens, closes):
    expanded = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        code, _ = _split_comment(line)
        insertions = set()
        for number, column in opens:
            if number == line_number and code[column:].strip():
                insertions.add(column)
        for number, column in closes:
            index = column - 1
            if number == line_number and code[:index].strip():
                insertions.add(index)
        for index in sorted(insertions, reverse=True):
            line = line[:index] + "\n" + line[index:]
        expanded.extend(line.split("\n"))
    return "\n".join(expanded)


def format_source(source):
    """Validate and format U-Ray code without discarding comments."""
    program = parse(source)
    opens, closes = _block_positions(program)
    source = _expand_inline_blocks(source, opens, closes)
    lines = source.splitlines()
    if not lines:
        return ""

    formatted = []
    brace_depth = 0
    continuation_depth = 0
    for line in lines:
        code, comment = _split_comment(line)
        tokens = _scan_line(code)
        if not tokens:
            if comment:
                formatted.append(
                    "    " * (brace_depth + (1 if continuation_depth else 0))
                    + comment.strip()
                )
            else:
                formatted.append("")
            continue

        leading_braces = 0
        for token in tokens:
            if token != "}":
                break
            leading_braces += 1
        leading_delims = 0
        for token in tokens:
            if token not in {")", "]"}:
                break
            leading_delims += 1
        indent = max(0, brace_depth - leading_braces)
        indent += max(0, continuation_depth - leading_delims)
        rendered = _format_tokens(tokens)
        if comment:
            rendered += "  " + comment.strip()
        formatted.append("    " * indent + rendered.rstrip())

        for token in tokens:
            if token == "{":
                brace_depth += 1
            elif token == "}":
                brace_depth = max(0, brace_depth - 1)
            elif token in {"(", "["}:
                continuation_depth += 1
            elif token in {")", "]"}:
                continuation_depth = max(0, continuation_depth - 1)

    while formatted and not formatted[-1]:
        formatted.pop()
    return "\n".join(formatted) + ("\n" if formatted else "")
