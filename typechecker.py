"""Conservative, compile-time checks for statically known U-Ray shapes."""

from dataclasses import dataclass

from ast_nodes import (
    Assign, Attr, Block, Bool, Break, Call, Continue, Del, DictLit, ExprStmt,
    For, FuncDecl, If, Import, Index, Lambda, ListLit, Match, MethodDecl,
    Number, Print, Program, Return, Str, Throw, Try, TupleLit, TypeDecl,
    UnaryOp, Var, While,
)
from errors import StaticTypeError


@dataclass(frozen=True)
class TypeInfo:
    kind: str = "value"
    shape: tuple = None
    input_dim: int = None
    output_dim: int = None


class StaticChecker:
    """Track dimensions only when they can be proven from literal expressions."""

    def __init__(self):
        self.symbols = {}

    def check(self, program):
        symbols = dict(self.symbols)
        self.errors = []
        self._statements(program.statements, symbols)
        if self.errors:
            raise StaticTypeError("\n".join(self.errors))
        return symbols

    @staticmethod
    def _qualified_name(node):
        if isinstance(node, Var):
            return node.name
        if isinstance(node, Attr):
            parent = StaticChecker._qualified_name(node.target)
            return f"{parent}.{node.name}" if parent else None
        return None

    @staticmethod
    def _shape(expr):
        if isinstance(expr, (Number, Str, Bool)):
            return ()
        if isinstance(expr, (ListLit, TupleLit)):
            if not expr.items:
                # Empty list literals are commonly filled later with push().
                return None
            children = [StaticChecker._shape(item) for item in expr.items]
            if any(shape is None for shape in children):
                return None
            if any(shape != children[0] for shape in children[1:]):
                return None
            return (len(expr.items),) + children[0]
        return None

    @staticmethod
    def _literal_integer(expr):
        if isinstance(expr, Number) and isinstance(expr.value, int) \
                and not isinstance(expr.value, bool):
            return expr.value
        if isinstance(expr, UnaryOp) and isinstance(expr.operand, Number) \
                and isinstance(expr.operand.value, int) \
                and not isinstance(expr.operand.value, bool):
            if expr.op == "-":
                return -expr.operand.value
            if expr.op == "+":
                return expr.operand.value
        return None

    @staticmethod
    def _is_literal_dimension(expr):
        return isinstance(expr, (Number, Bool)) or (
            isinstance(expr, UnaryOp)
            and isinstance(expr.operand, Number)
        )

    def _error(self, message, node=None):
        location = ""
        if node is not None and node.line is not None:
            location = f" (строка {node.line}:{node.col})"
        diagnostic = f"Статическая проверка{location}: {message}"
        if diagnostic not in self.errors:
            self.errors.append(diagnostic)

    @staticmethod
    def _root_variable(target):
        while isinstance(target, (Attr, Index)):
            target = target.target
        return target.name if isinstance(target, Var) else None

    def _check_layer(self, layer, vector_shape, operation, node):
        if layer.kind != "layer" or layer.input_dim is None \
                or vector_shape is None:
            return
        if len(vector_shape) != 1:
            self._error(
                f"{operation} ожидает вектор, получена форма "
                f"{list(vector_shape)}",
                node,
            )
        elif vector_shape[0] != layer.input_dim:
            self._error(
                f"{operation}: слой ожидает вход размера {layer.input_dim}, "
                f"получен размер {vector_shape[0]}",
                node,
            )

    def _check_model_input(self, model, vector_shape, operation, node):
        if model.kind != "model" or model.input_dim is None \
                or vector_shape is None:
            return
        if len(vector_shape) != 1:
            self._error(
                f"{operation} ожидает вектор, получена форма "
                f"{list(vector_shape)}",
                node,
            )
        elif vector_shape[0] != model.input_dim:
            self._error(
                f"{operation}: модель ожидает вход размера {model.input_dim}, "
                f"получен размер {vector_shape[0]}",
                node,
            )

    def _check_training_data(self, model, x_info, y_info, operation, node):
        if model.kind != "model":
            return
        x_shape = x_info.shape
        y_shape = y_info.shape
        if x_shape is not None:
            if len(x_shape) != 2:
                self._error(
                    f"{operation}: X должен быть матрицей, "
                    f"получена форма {list(x_shape)}",
                    node,
                )
            elif model.input_dim is not None and x_shape[1] != model.input_dim:
                self._error(
                    f"{operation}: модель ожидает X шириной "
                    f"{model.input_dim}, получена ширина {x_shape[1]}",
                    node,
                )
        if y_shape is not None:
            if len(y_shape) != 2:
                self._error(
                    f"{operation}: Y должен быть матрицей, "
                    f"получена форма {list(y_shape)}",
                    node,
                )
            elif model.output_dim is not None and y_shape[1] != model.output_dim:
                self._error(
                    f"{operation}: модель ожидает Y шириной "
                    f"{model.output_dim}, получена ширина {y_shape[1]}",
                    node,
                )
        if x_shape is not None and y_shape is not None \
                and len(x_shape) >= 1 and len(y_shape) >= 1 \
                and x_shape[0] != y_shape[0]:
            self._error(
                f"{operation}: число примеров X ({x_shape[0]}) "
                f"не совпадает с числом ответов Y ({y_shape[0]})",
                node,
            )

    def _expression(self, expr, symbols):
        if isinstance(expr, Var):
            return symbols.get(expr.name, TypeInfo())

        if isinstance(expr, (Number, Str, Bool)):
            return TypeInfo(shape=())

        if isinstance(expr, (ListLit, TupleLit)):
            for item in expr.items:
                self._expression(item, symbols)
            return TypeInfo(shape=self._shape(expr))

        if isinstance(expr, DictLit):
            for key, value in expr.pairs:
                self._expression(key, symbols)
                self._expression(value, symbols)
            return TypeInfo()

        if isinstance(expr, Attr):
            self._expression(expr.target, symbols)
            return TypeInfo()

        if isinstance(expr, Index):
            self._expression(expr.target, symbols)
            self._expression(expr.index, symbols)
            return TypeInfo()

        if isinstance(expr, UnaryOp):
            self._expression(expr.operand, symbols)
            return TypeInfo()

        if not isinstance(expr, Call):
            for value in vars(expr).values() if hasattr(expr, "__dict__") else ():
                if hasattr(value, "__dict__"):
                    self._expression(value, symbols)
                elif isinstance(value, list):
                    for item in value:
                        if hasattr(item, "__dict__"):
                            self._expression(item, symbols)
            return TypeInfo()

        name = self._qualified_name(expr.callee)
        args = [self._expression(arg, symbols) for arg in expr.args]
        if name in ("push", "pop") and expr.args:
            mutated = self._root_variable(expr.args[0])
            if mutated:
                symbols.pop(mutated, None)

        if name == "nn.dense":
            if len(expr.args) == 2:
                if all(self._is_literal_dimension(arg) for arg in expr.args):
                    input_dim, output_dim = (
                        self._literal_integer(arg) for arg in expr.args
                    )
                    if input_dim is None or output_dim is None \
                            or input_dim <= 0 or output_dim <= 0:
                        self._error(
                            "nn.dense: размеры слоя должны быть "
                            "положительными целыми числами",
                            expr,
                        )
                    else:
                        return TypeInfo(
                            kind="layer",
                            input_dim=input_dim,
                            output_dim=output_dim,
                        )
            return TypeInfo(kind="layer")

        if name == "nn.mlp" and expr.args:
            dims_expr = expr.args[0]
            if isinstance(dims_expr, ListLit):
                if all(self._is_literal_dimension(item) for item in dims_expr.items):
                    dims = [
                        self._literal_integer(item) for item in dims_expr.items
                    ]
                    if len(dims) < 2:
                        self._error("nn.mlp: нужны размеры входа и выхода", expr)
                    elif any(dim is None or dim <= 0 for dim in dims):
                        self._error(
                            "nn.mlp: размеры слоёв должны быть "
                            "положительными целыми числами",
                            expr,
                        )
                    else:
                        return TypeInfo(
                            kind="model",
                            input_dim=dims[0],
                            output_dim=dims[-1],
                        )
            return TypeInfo(kind="model")

        if name in ("nn.forward", "nn.forward_dense") and len(args) == 2:
            self._check_layer(args[0], args[1].shape, name, expr)
            layer = args[0]
            if layer.kind == "layer" and layer.output_dim is not None:
                if name == "nn.forward_dense":
                    return TypeInfo(kind="dense_cache")
                return TypeInfo(shape=(layer.output_dim,))
            return TypeInfo()

        if name == "nn.forward_mlp" and len(args) == 2:
            self._check_model_input(args[0], args[1].shape, name, expr)
            model = args[0]
            if model.kind == "model" and model.output_dim is not None:
                # The runtime returns a cache; its final element is the output.
                return TypeInfo()

        if name in ("nn.train", "nn.evaluate") and len(args) >= 3:
            self._check_training_data(args[0], args[1], args[2], name, expr)
            return TypeInfo()

        return TypeInfo()

    def _statements(self, statements, symbols):
        for statement in statements:
            if isinstance(statement, Assign):
                value = self._expression(statement.value, symbols)
                if isinstance(statement.target, Var):
                    if value.kind == "value" and value.shape is None:
                        symbols.pop(statement.target.name, None)
                    else:
                        symbols[statement.target.name] = value
                else:
                    self._expression(statement.target, symbols)
                    mutated = self._root_variable(statement.target)
                    if mutated:
                        symbols.pop(mutated, None)
            elif isinstance(statement, (ExprStmt,)):
                self._expression(statement.expr, symbols)
            elif isinstance(statement, Print):
                for value in statement.values:
                    self._expression(value, symbols)
            elif isinstance(statement, Return):
                if statement.value is not None:
                    self._expression(statement.value, symbols)
            elif isinstance(statement, Del):
                self._expression(statement.target, symbols)
                mutated = self._root_variable(statement.target)
                if mutated:
                    symbols.pop(mutated, None)
            elif isinstance(statement, If):
                self._expression(statement.cond, symbols)
                self._statements(statement.then_block.statements, dict(symbols))
                if statement.else_block:
                    self._statements(statement.else_block.statements, dict(symbols))
            elif isinstance(statement, While):
                self._expression(statement.cond, symbols)
                self._statements(statement.body.statements, dict(symbols))
            elif isinstance(statement, For):
                if statement.iterable is not None:
                    self._expression(statement.iterable, symbols)
                if statement.start is not None:
                    self._statements([statement.start], dict(symbols))
                loop_symbols = dict(symbols)
                if isinstance(statement.var, str):
                    loop_symbols.pop(statement.var, None)
                elif isinstance(statement.var, list):
                    for name in statement.var:
                        loop_symbols.pop(name, None)
                if statement.body:
                    self._statements(statement.body.statements, loop_symbols)
                if statement.step:
                    self._statements([statement.step], loop_symbols)
            elif isinstance(statement, FuncDecl):
                symbols[statement.name] = TypeInfo(kind="function")
                local = dict(symbols)
                for param in statement.params:
                    local.pop(param, None)
                self._statements(statement.body.statements, local)
            elif isinstance(statement, MethodDecl):
                local = dict(symbols)
                for param in statement.params:
                    local.pop(param, None)
                self._statements(statement.body.statements, local)
            elif isinstance(statement, TypeDecl):
                symbols[statement.name] = TypeInfo(kind="type")
                for method in statement.methods:
                    self._statements([method], dict(symbols))
            elif isinstance(statement, Try):
                self._statements(statement.try_block.statements, dict(symbols))
                catch_symbols = dict(symbols)
                catch_symbols.pop(statement.catch_var, None)
                self._statements(statement.catch_block.statements, catch_symbols)
            elif isinstance(statement, Match):
                self._expression(statement.expr, symbols)
                for _, block in statement.cases:
                    self._statements(block.statements, dict(symbols))
                if statement.default_block:
                    self._statements(
                        statement.default_block.statements, dict(symbols)
                    )
            elif isinstance(statement, Throw):
                self._expression(statement.value, symbols)
            elif isinstance(statement, Block):
                self._statements(statement.statements, dict(symbols))
            elif isinstance(statement, Import):
                if statement.name:
                    symbols.pop(statement.name, None)
            elif isinstance(statement, (Break, Continue)):
                continue
            elif isinstance(statement, Lambda):
                continue


def check_program(program, checker=None):
    checker = checker or StaticChecker()
    symbols = checker.check(program)
    return symbols
