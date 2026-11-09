# ast_nodes.py

class Node:
    pass


class Program(Node):
    def __init__(self, statements):
        self.statements = statements

    def __repr__(self):
        return f"Program({self.statements})"


class Import(Node):
    def __init__(self, name, path=None):
        self.name = name
        self.path = path

    def __repr__(self):
        return f"Import({self.name!r}, path={self.path!r})"


class Assign(Node):
    def __init__(self, target, value):
        self.target = target
        self.value = value

    def __repr__(self):
        return f"Assign({self.target}, {self.value})"


class Del(Node):
    def __init__(self, target):
        self.target = target

    def __repr__(self):
        return f"Del({self.target})"


class Print(Node):
    def __init__(self, values):
        self.values = values

    def __repr__(self):
        return f"Print({self.values})"


class If(Node):
    def __init__(self, cond, then_block, else_block=None):
        self.cond = cond
        self.then_block = then_block
        self.else_block = else_block

    def __repr__(self):
        return f"If({self.cond}, {self.then_block}, {self.else_block})"


class While(Node):
    def __init__(self, cond, body):
        self.cond = cond
        self.body = body

    def __repr__(self):
        return f"While({self.cond}, {self.body})"


class For(Node):
    def __init__(self, var, iterable, body):
        self.var = var
        self.iterable = iterable
        self.body = body

    def __repr__(self):
        return f"For({self.var!r}, {self.iterable}, {self.body})"


class Break(Node):
    def __repr__(self):
        return "Break()"


class Continue(Node):
    def __repr__(self):
        return "Continue()"


class FuncDecl(Node):
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body

    def __repr__(self):
        return f"FuncDecl({self.name!r}, {self.params}, {self.body})"


class Return(Node):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Return({self.value})"


class Block(Node):
    def __init__(self, statements):
        self.statements = statements

    def __repr__(self):
        return f"Block({self.statements})"


# --- выражения ---

class Number(Node):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Number({self.value})"


class Str(Node):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Str({self.value!r})"


class Bool(Node):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Bool({self.value})"


class Var(Node):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Var({self.name!r})"


class ListLit(Node):
    def __init__(self, items):
        self.items = items

    def __repr__(self):
        return f"ListLit({self.items})"


class TupleLit(Node):
    def __init__(self, items):
        self.items = items

    def __repr__(self):
        return f"TupleLit({self.items})"


class DictLit(Node):
    def __init__(self, pairs):
        self.pairs = pairs

    def __repr__(self):
        return f"DictLit({self.pairs})"


class Index(Node):
    def __init__(self, target, index):
        self.target = target
        self.index = index

    def __repr__(self):
        return f"Index({self.target}, {self.index})"


class Attr(Node):
    def __init__(self, target, name):
        self.target = target
        self.name = name

    def __repr__(self):
        return f"Attr({self.target}, {self.name!r})"


class Call(Node):
    def __init__(self, callee, args):
        self.callee = callee
        self.args = args

    def __repr__(self):
        return f"Call({self.callee}, {self.args})"


class BinOp(Node):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self):
        return f"BinOp({self.op!r}, {self.left}, {self.right})"


class UnaryOp(Node):
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

    def __repr__(self):
        return f"UnaryOp({self.op!r}, {self.operand})"
