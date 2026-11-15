# parser.py

from errors import ParseError
from lexer import (
    tokenize, NUMBER, STRING, IDENT, ASSIGN, PLUS, MINUS,
    STAR, SLASH, PERCENT, POW, LPAREN, RPAREN, LBRACKET, RBRACKET,
    LBRACE, RBRACE, COLON, SEMICOLON, PRINT, EOF,
    EQ, NEQ, LT, GT, LE, GE, IF, ELSE, WHILE, FOR, IN, BREAK, CONTINUE,
    FUNC, RETURN, IMPORT, DEL, COMMA, DOT, TRUE, FALSE,
    TRY, CATCH, THROW, MATCH, UNDERSCORE, TYPE
)
from ast_nodes import (
    Program, Import, Assign, Del, Print, Number, Str, Var, BinOp,
    If, While, For, Break, Continue, FuncDecl, MethodDecl, Return, Block, Bool, Call,
    ListLit, DictLit, Index, Attr, UnaryOp, TupleLit,
    Try, Throw, Match, TypeDecl, Slice, Lambda, Ternary,
    StructPattern, BindPattern, WildcardPattern, ExprStmt
)


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.loop_depth = 0
        self.no_ternary = 0

    def peek(self):
        return self.tokens[self.pos]

    def advance(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def check(self, type_):
        return self.peek().type == type_

    def expect(self, type_):
        tok = self.peek()
        if tok.type != type_:
            raise ParseError(
                f"Ожидался {type_}, но получен {tok.type} "
                f"({tok.value!r}) в {tok.line}:{tok.col}"
            )
        return self.advance()

    def parse_program(self):
        statements = []
        while not self.check(EOF):
            statements.append(self.parse_statement())
        return Program(statements)

    def parse_statement(self):
        if self.check(PRINT):
            return self.parse_print()
        if self.check(IF):
            return self.parse_if()
        if self.check(WHILE):
            return self.parse_while()
        if self.check(FOR):
            return self.parse_for()
        if self.check(BREAK):
            return self.parse_break()
        if self.check(CONTINUE):
            return self.parse_continue()
        if self.check(FUNC):
            return self.parse_func()
        if self.check(RETURN):
            return self.parse_return()
        if self.check(IMPORT):
            return self.parse_import()
        if self.check(DEL):
            return self.parse_del()
        if self.check(TRY):
            return self.parse_try()
        if self.check(THROW):
            return self.parse_throw()
        if self.check(MATCH):
            return self.parse_match()
        if self.check(TYPE):
            return self.parse_type()
        if self.check(IDENT):
            left = self.parse_expr()
            if self.check(ASSIGN):
                self.advance()
                value = self.parse_expr()
                return Assign(left, value)
            return ExprStmt(left)
        if self.check(NUMBER) or self.check(STRING) or self.check(TRUE) \
                or self.check(FALSE) or self.check(LBRACKET) \
                or self.check(LBRACE) or self.check(LPAREN) \
                or self.check(FUNC) or self.check(MINUS) or self.check(PLUS):
            return ExprStmt(self.parse_expr())
        tok = self.peek()
        raise ParseError(
            f"Неожиданный токен {tok.type} ({tok.value!r}) в {tok.line}:{tok.col}"
        )

    def parse_del(self):
        self.expect(DEL)
        target = self.parse_factor()
        if not isinstance(target, Index):
            raise ParseError("del работает только с d[key] или a[index]")
        return Del(target)

    def parse_try(self):
        self.expect(TRY)
        try_block = self.parse_block()
        self.expect(CATCH)
        catch_var = self.expect(IDENT).value
        catch_block = self.parse_block()
        return Try(try_block, catch_var, catch_block)

    def parse_throw(self):
        self.expect(THROW)
        value = self.parse_expr()
        return Throw(value)

    def parse_match(self):
        self.expect(MATCH)
        expr = self.parse_expr()
        open_token = self.expect(LBRACE)
        cases = []
        default_block = None
        while not self.check(RBRACE):
            if self.check(UNDERSCORE):
                self.advance()
                default_block = self.parse_block()
            else:
                pattern = self.parse_match_pattern()
                block = self.parse_block()
                cases.append((pattern, block))
        close_token = self.expect(RBRACE)
        node = Match(
            expr,
            cases,
            start=(open_token.line, open_token.col),
            end=(close_token.line, close_token.col),
        )
        node.default_block = default_block
        return node

    def parse_match_pattern(self):
        """Parse constructor-style struct patterns, falling back to value patterns."""
        if self.check(IDENT) and self.pos + 1 < len(self.tokens) \
                and self.tokens[self.pos + 1].type in (LPAREN, LBRACE):
            type_name = self.advance().value
            if self.check(LPAREN):
                self.advance()
                fields = []
                if not self.check(RPAREN):
                    fields.append(self.parse_match_pattern_item())
                    while self.check(COMMA):
                        self.advance()
                        if self.check(RPAREN):
                            break
                        fields.append(self.parse_match_pattern_item())
                self.expect(RPAREN)
                return StructPattern(type_name, fields)

            self.expect(LBRACE)
            fields = []
            seen_fields = set()
            while not self.check(RBRACE):
                field_name = self.expect(IDENT).value
                if field_name in seen_fields:
                    raise ParseError(f"Поле {field_name!r} повторяется в шаблоне")
                seen_fields.add(field_name)
                if self.check(COLON):
                    self.advance()
                    field_pattern = self.parse_match_pattern_item()
                else:
                    field_pattern = BindPattern(field_name)
                fields.append((field_name, field_pattern))
                if self.check(COMMA):
                    self.advance()
                    continue
                break
            self.expect(RBRACE)
            return StructPattern(type_name, fields, named=True)
        return self.parse_expr()

    def parse_match_pattern_item(self):
        if self.check(UNDERSCORE):
            self.advance()
            return WildcardPattern()
        if self.check(IDENT) and self.pos + 1 < len(self.tokens) \
                and self.tokens[self.pos + 1].type in (LPAREN, LBRACE):
            return self.parse_match_pattern()
        if self.check(IDENT):
            return BindPattern(self.advance().value)
        return self.parse_expr()

    def parse_type(self):
        self.expect(TYPE)
        name = self.expect(IDENT).value
        open_token = self.expect(LBRACE)
        fields = []
        methods = []
        while not self.check(RBRACE):
            if self.check(FUNC):
                self.advance()
                mname = self.expect(IDENT).value
                self.expect(LPAREN)
                params = []
                if not self.check(RPAREN):
                    params.append(self.expect(IDENT).value)
                    while self.check(COMMA):
                        self.advance()
                        params.append(self.expect(IDENT).value)
                self.expect(RPAREN)
                saved_depth = self.loop_depth
                self.loop_depth = 0
                try:
                    body = self.parse_block()
                finally:
                    self.loop_depth = saved_depth
                methods.append(MethodDecl(mname, params, body))
            else:
                fields.append(self.expect(IDENT).value)
        close_token = self.expect(RBRACE)
        return TypeDecl(
            name,
            fields,
            methods,
            start=(open_token.line, open_token.col),
            end=(close_token.line, close_token.col),
        )

    def parse_import(self):
        self.expect(IMPORT)
        if self.check(STRING):
            path = self.expect(STRING).value
            return Import(None, path)
        name = self.expect(IDENT).value
        return Import(name, None)

    def parse_print(self):
        self.expect(PRINT)
        self.expect(LPAREN)
        values = []
        if not self.check(RPAREN):
            values.append(self.parse_expr())
            while self.check(COMMA):
                self.advance()
                values.append(self.parse_expr())
        self.expect(RPAREN)
        return Print(values)

    def parse_if(self):
        self.expect(IF)
        self.no_ternary += 1
        try:
            cond = self.parse_expr()
        finally:
            self.no_ternary -= 1
        then_block = self.parse_block()
        else_block = None
        if self.check(ELSE):
            self.advance()
            else_block = self.parse_block()
        return If(cond, then_block, else_block)

    def parse_while(self):
        self.expect(WHILE)
        self.no_ternary += 1
        try:
            cond = self.parse_expr()
        finally:
            self.no_ternary -= 1
        self.loop_depth += 1
        try:
            body = self.parse_block()
        finally:
            self.loop_depth -= 1
        return While(cond, body)

    def parse_for(self):
        self.expect(FOR)
        # C-style for (for i = 0; i < 5; i = i + 1)
        if self.check(IDENT) and self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == ASSIGN:
            var = self.expect(IDENT).value
            self.expect(ASSIGN)
            start_val = self.parse_expr()
            start = Assign(Var(var), start_val)
            self.expect(SEMICOLON)
            
            self.no_ternary += 1
            try:
                cond = self.parse_expr()
            finally:
                self.no_ternary -= 1
            self.expect(SEMICOLON)
            
            step_target = self.expect(IDENT).value
            self.expect(ASSIGN)
            step_val = self.parse_expr()
            step = Assign(Var(step_target), step_val)
            
            self.loop_depth += 1
            try:
                body = self.parse_block()
            finally:
                self.loop_depth -= 1
            return For(var, cond, body=body, start=start, step=step)

        # Обычный for in
        first_var = self.expect(IDENT).value
        unpack_vars = None
        if self.check(COMMA):
            unpack_vars = [first_var]
            while self.check(COMMA):
                self.advance()
                unpack_vars.append(self.expect(IDENT).value)
        self.expect(IN)
        iterable = self.parse_expr()
        self.loop_depth += 1
        try:
            body = self.parse_block()
        finally:
            self.loop_depth -= 1
        if unpack_vars is None:
            return For(first_var, iterable, body)
        return For(unpack_vars, iterable, body)

    def parse_break(self):
        tok = self.expect(BREAK)
        if self.loop_depth == 0:
            raise ParseError(f"'break' вне цикла в {tok.line}:{tok.col}")
        return Break()

    def parse_continue(self):
        tok = self.expect(CONTINUE)
        if self.loop_depth == 0:
            raise ParseError(f"'continue' вне цикла в {tok.line}:{tok.col}")
        return Continue()

    def parse_func(self):
        self.expect(FUNC)
        name = self.expect(IDENT).value
        self.expect(LPAREN)
        params = []
        if not self.check(RPAREN):
            params.append(self.expect(IDENT).value)
            while self.check(COMMA):
                self.advance()
                params.append(self.expect(IDENT).value)
        self.expect(RPAREN)
        saved_depth = self.loop_depth
        self.loop_depth = 0
        try:
            body = self.parse_block()
        finally:
            self.loop_depth = saved_depth
        return FuncDecl(name, params, body)

    def parse_lambda(self):
        self.expect(FUNC)
        self.expect(LPAREN)
        params = []
        if not self.check(RPAREN):
            params.append(self.expect(IDENT).value)
            while self.check(COMMA):
                self.advance()
                params.append(self.expect(IDENT).value)
        self.expect(RPAREN)
        body = self.parse_block()
        return Lambda(params, body)

    def parse_return(self):
        self.expect(RETURN)
        if self.check(RBRACE) or self.check(EOF):
            return Return(None)
        value = self.parse_expr()
        return Return(value)

    def parse_block(self):
        open_token = self.expect(LBRACE)
        statements = []
        while not self.check(RBRACE):
            statements.append(self.parse_statement())
        close_token = self.expect(RBRACE)
        return Block(
            statements,
            start=(open_token.line, open_token.col),
            end=(close_token.line, close_token.col),
        )

    def parse_expr(self):
        left = self.parse_additive()
        while self.check(EQ) or self.check(NEQ) or self.check(LT) \
                or self.check(GT) or self.check(LE) or self.check(GE):
            op = self.advance().value
            right = self.parse_additive()
            left = BinOp(op, left, right)

        if self.check(SEMICOLON):
            return left

        # Тернарный оператор (только если на той же строке)
        if self.check(IF) and self.no_ternary == 0:
            prev_tok = self.tokens[self.pos - 1]
            if_tok = self.peek()
            if if_tok.line == prev_tok.line:
                self.advance()
                cond = self.parse_expr()
                self.expect(ELSE)
                false_val = self.parse_expr()
                return Ternary(cond, left, false_val)
        
        return left

    def parse_additive(self):
        left = self.parse_term()
        while self.check(PLUS) or self.check(MINUS):
            op = self.advance().value
            right = self.parse_term()
            left = BinOp(op, left, right)
        return left

    def parse_term(self):
        left = self.parse_unary()
        while self.check(STAR) or self.check(SLASH) or self.check(PERCENT):
            op = self.advance().value
            right = self.parse_unary()
            left = BinOp(op, left, right)
        return left

    def parse_unary(self):
        if self.check(MINUS) or self.check(PLUS):
            op = self.advance().value
            operand = self.parse_unary()
            return UnaryOp(op, operand)
        return self.parse_power()

    def parse_power(self):
        left = self.parse_factor()
        if self.check(POW):
            self.advance()
            right = self.parse_unary()
            return BinOp("**", left, right)
        return left

    def parse_factor(self):
        node = self.parse_primary()
        while True:
            if self.check(DOT):
                self.advance()
                name = self.expect(IDENT).value
                node = Attr(node, name)
                continue
            if self.check(LBRACKET):
                self.advance()
                if self.check(COLON):
                    self.advance()
                    end = None
                    if not self.check(RBRACKET):
                        end = self.parse_expr()
                    self.expect(RBRACKET)
                    node = Index(node, Slice(None, end))
                    continue
                
                start = self.parse_expr()
                if self.check(COLON):
                    self.advance()
                    end = None
                    if not self.check(RBRACKET):
                        end = self.parse_expr()
                    self.expect(RBRACKET)
                    node = Index(node, Slice(start, end))
                    continue
                
                if self.check(COMMA):
                    items = [start]
                    while self.check(COMMA):
                        self.advance()
                        items.append(self.parse_expr())
                    self.expect(RBRACKET)
                    node = Index(node, TupleLit(items))
                    continue
                
                self.expect(RBRACKET)
                node = Index(node, start)
                continue
            if self.check(LPAREN):
                open_token = self.advance()
                args = []
                if not self.check(RPAREN):
                    args.append(self.parse_expr())
                    while self.check(COMMA):
                        self.advance()
                        args.append(self.parse_expr())
                self.expect(RPAREN)
                node = Call(node, args, line=open_token.line, col=open_token.col)
                continue
            break
        return node

    def parse_primary(self):
        tok = self.peek()
        if tok.type == TRUE:
            self.advance()
            return Bool(True)
        if tok.type == FALSE:
            self.advance()
            return Bool(False)
        if tok.type == NUMBER:
            self.advance()
            return Number(tok.value)
        if tok.type == STRING:
            self.advance()
            return Str(tok.value)
        if tok.type == LBRACKET:
            return self.parse_list_literal()
        if tok.type == LBRACE:
            return self.parse_dict_literal()
        if tok.type == LPAREN:
            return self.parse_tuple_or_group()
        if tok.type == FUNC:
            return self.parse_lambda()
        if tok.type == IDENT:
            self.advance()
            return Var(tok.value)
        raise ParseError(
            f"Ожидалось выражение, но получен {tok.type} "
            f"({tok.value!r}) в {tok.line}:{tok.col}"
        )

    def parse_list_literal(self):
        self.expect(LBRACKET)
        items = []
        if not self.check(RBRACKET):
            items.append(self.parse_expr())
            while self.check(COMMA):
                self.advance()
                items.append(self.parse_expr())
        self.expect(RBRACKET)
        return ListLit(items)

    def parse_dict_literal(self):
        self.expect(LBRACE)
        pairs = []
        if not self.check(RBRACE):
            while True:
                key = self.parse_expr()
                self.expect(COLON)
                value = self.parse_expr()
                pairs.append((key, value))
                if self.check(COMMA):
                    self.advance()
                    continue
                break
        self.expect(RBRACE)
        return DictLit(pairs)

    def parse_tuple_or_group(self):
        self.expect(LPAREN)
        if self.check(RPAREN):
            self.advance()
            return TupleLit([])
        first = self.parse_expr()
        if not self.check(COMMA):
            self.expect(RPAREN)
            return first
        items = [first]
        while self.check(COMMA):
            self.advance()
            if self.check(RPAREN):
                break
            items.append(self.parse_expr())
        self.expect(RPAREN)
        return TupleLit(items)


def parse(src):
    tokens = tokenize(src)
    return Parser(tokens).parse_program()
