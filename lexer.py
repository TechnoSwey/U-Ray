# lexer.py

from errors import LexerError

NUMBER = "NUMBER"
STRING = "STRING"
IDENT  = "IDENT"
ASSIGN = "ASSIGN"
PLUS   = "PLUS"
MINUS  = "MINUS"
STAR   = "STAR"
SLASH  = "SLASH"
PERCENT = "PERCENT"
POW    = "POW"
LPAREN = "LPAREN"
RPAREN = "RPAREN"
LBRACKET = "LBRACKET"
RBRACKET = "RBRACKET"
LBRACE = "LBRACE"
RBRACE = "RBRACE"
COLON  = "COLON"
SEMICOLON = "SEMICOLON"
PRINT  = "PRINT"
EOF    = "EOF"

EQ     = "EQ"
NEQ    = "NEQ"
LT     = "LT"
GT     = "GT"
LE     = "LE"
GE     = "GE"
IF     = "IF"
ELSE   = "ELSE"
WHILE  = "WHILE"
FOR    = "FOR"
IN     = "IN"
BREAK  = "BREAK"
CONTINUE = "CONTINUE"
FUNC   = "FUNC"
RETURN = "RETURN"
IMPORT = "IMPORT"
DEL    = "DEL"
TRY    = "TRY"
CATCH  = "CATCH"
THROW  = "THROW"
MATCH  = "MATCH"
UNDERSCORE = "UNDERSCORE"
TYPE   = "TYPE"

COMMA  = "COMMA"
DOT    = "DOT"
TRUE   = "TRUE"
FALSE  = "FALSE"

KEYWORDS = {
    "print": PRINT,
    "if": IF,
    "else": ELSE,
    "while": WHILE,
    "for": FOR,
    "in": IN,
    "break": BREAK,
    "continue": CONTINUE,
    "func": FUNC,
    "return": RETURN,
    "import": IMPORT,
    "del": DEL,
    "try": TRY,
    "catch": CATCH,
    "throw": THROW,
    "match": MATCH,
    "true": TRUE,
    "false": FALSE,
    "type": TYPE,
}


class Token:
    def __init__(self, type_, value=None, line=1, col=1):
        self.type = type_
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, {self.line}:{self.col})"


class Lexer:
    def __init__(self, src):
        self.src = src
        self.pos = 0
        self.line = 1
        self.col = 1

    def peek(self, offset=0):
        i = self.pos + offset
        return self.src[i] if i < len(self.src) else ""

    def advance(self):
        ch = self.src[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def tokenize(self):
        tokens = []
        while self.pos < len(self.src):
            ch = self.peek()

            if ch in " \t\r\n":
                self.advance()
                continue

            if ch == "/" and self.peek(1) == "/":
                while self.pos < len(self.src) and self.peek() != "\n":
                    self.advance()
                continue

            if ch.isdigit():
                tokens.append(self.read_number())
                continue

            if ch == '"':
                tokens.append(self.read_string())
                continue

            if ch.isalpha() or ch == "_":
                tokens.append(self.read_ident())
                continue

            line, col = self.line, self.col

            two = self.src[self.pos:self.pos + 2]
            if two == "==":
                self.advance(); self.advance()
                tokens.append(Token(EQ, "==", line, col)); continue
            if two == "!=":
                self.advance(); self.advance()
                tokens.append(Token(NEQ, "!=", line, col)); continue
            if two == "<=":
                self.advance(); self.advance()
                tokens.append(Token(LE, "<=", line, col)); continue
            if two == ">=":
                self.advance(); self.advance()
                tokens.append(Token(GE, ">=", line, col)); continue
            if two == "**":
                self.advance(); self.advance()
                tokens.append(Token(POW, "**", line, col)); continue

            if ch == "=":
                self.advance(); tokens.append(Token(ASSIGN, "=", line, col)); continue
            if ch == "+":
                self.advance(); tokens.append(Token(PLUS, "+", line, col)); continue
            if ch == "-":
                self.advance(); tokens.append(Token(MINUS, "-", line, col)); continue
            if ch == "*":
                self.advance(); tokens.append(Token(STAR, "*", line, col)); continue
            if ch == "/":
                self.advance(); tokens.append(Token(SLASH, "/", line, col)); continue
            if ch == "%":
                self.advance(); tokens.append(Token(PERCENT, "%", line, col)); continue
            if ch == "<":
                self.advance(); tokens.append(Token(LT, "<", line, col)); continue
            if ch == ">":
                self.advance(); tokens.append(Token(GT, ">", line, col)); continue
            if ch == "(":
                self.advance(); tokens.append(Token(LPAREN, "(", line, col)); continue
            if ch == ")":
                self.advance(); tokens.append(Token(RPAREN, ")", line, col)); continue
            if ch == "[":
                self.advance(); tokens.append(Token(LBRACKET, "[", line, col)); continue
            if ch == "]":
                self.advance(); tokens.append(Token(RBRACKET, "]", line, col)); continue
            if ch == "{":
                self.advance(); tokens.append(Token(LBRACE, "{", line, col)); continue
            if ch == "}":
                self.advance(); tokens.append(Token(RBRACE, "}", line, col)); continue
            if ch == ":":
                self.advance(); tokens.append(Token(COLON, ":", line, col)); continue
            if ch == ";":
                self.advance(); tokens.append(Token(SEMICOLON, ";", line, col)); continue
            if ch == ",":
                self.advance(); tokens.append(Token(COMMA, ",", line, col)); continue
            if ch == ".":
                self.advance(); tokens.append(Token(DOT, ".", line, col)); continue

            raise LexerError(f"Неизвестный символ {ch!r} в {self.line}:{self.col}")

        tokens.append(Token(EOF, None, self.line, self.col))
        return tokens

    def read_number(self):
        line, col = self.line, self.col
        start = self.pos
        while self.peek().isdigit():
            self.advance()
        is_float = False
        if self.peek() == "." and self.peek(1).isdigit():
            is_float = True
            self.advance()
            while self.peek().isdigit():
                self.advance()
        text = self.src[start:self.pos]
        if is_float:
            return Token(NUMBER, float(text), line, col)
        return Token(NUMBER, int(text), line, col)

    def read_ident(self):
        line, col = self.line, self.col
        start = self.pos
        while self.peek().isalnum() or self.peek() == "_":
            self.advance()
        text = self.src[start:self.pos]
        if text == "_":
            return Token(UNDERSCORE, "_", line, col)
        if text in KEYWORDS:
            return Token(KEYWORDS[text], text, line, col)
        return Token(IDENT, text, line, col)

    def read_string(self):
        line, col = self.line, self.col
        self.advance()
        buf = []
        while True:
            if self.pos >= len(self.src):
                raise LexerError(f"Незакрытая строка, начатая в {line}:{col}")
            ch = self.peek()
            if ch == '"':
                self.advance()
                break
            if ch == "\\":
                self.advance()
                if self.pos >= len(self.src):
                    raise LexerError(f"Незакрытая строка, начатая в {line}:{col}")
                esc = self.peek()
                if esc == "n": buf.append("\n")
                elif esc == "t": buf.append("\t")
                elif esc == '"': buf.append('"')
                elif esc == "\\": buf.append("\\")
                else:
                    raise LexerError(f"Неизвестная escape-последовательность \\{esc} в {self.line}:{self.col}")
                self.advance()
                continue
            if ch == "\n":
                raise LexerError(f"Перевод строки внутри строки в {self.line}:{self.col}")
            buf.append(ch)
            self.advance()
        return Token(STRING, "".join(buf), line, col)


def tokenize(src):
    return Lexer(src).tokenize()
