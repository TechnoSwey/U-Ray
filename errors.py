# errors.py


class URayError(Exception):
    """Базовая ошибка U-Ray. Ловит всё: лексер, парсер, рантайм."""
    pass


class LexerError(URayError):
    pass


class ParseError(URayError):
    pass


class RuntimeErrorUray(URayError):
    pass
