from core.lexer.token import TokenType, Token
from core.lexer.lexer import Lexer


class Error:
    @staticmethod
    def build_lexer_error_msg(lexer: Lexer) -> str:
        for token in lexer.tokens:
            if token.type != TokenType.UNEXPECTED_TOKEN:
                continue

            lines = lexer.buffer[:token.position].splitlines()
            line = len(lines) if lines else 1
            col = len(lines[-1]) + 1 if lines else 1

            line_start: int = lexer.buffer.rfind('\n', 0, token.position) + 1
            line_end: int = lexer.buffer.find('\n', token.position)

            if line_end == -1:
                line_end = len(lexer.buffer)

            snippet: str = lexer.buffer[line_start:line_end]
            caret_line = ' ' * (col - 1) + '^'
            msg: str = (
                f"error: unexpected token '{token.value}'\n"
                f" --> input:{line}:{col}\n"
                f"  |\n"
                f"{line} | {snippet}\n"
                f"  | {caret_line}"
            )
            
            return msg
        
        return "error: unhandled error occurred"

    @staticmethod
    def build_parser_error_msg(error: SyntaxError, buffer: str) -> str:
        # Extrait le token depuis le message d'erreur
        import re
        match = re.search(r"Token\(type=TokenType\.(\w+), position=(\d+), length=(\d+), value='([^']*)'\)", str(error))
        if not match:
            return f"error: {error}"

        token_type, pos, length, value = match.groups()
        pos = int(pos)
        length = int(length)

        lines = buffer[:pos].splitlines()
        line = len(lines) if lines else 1
        col = len(lines[-1]) + 1 if lines else 1

        line_start = buffer.rfind('\n', 0, pos) + 1
        line_end = buffer.find('\n', pos)
        if line_end == -1:
            line_end = len(buffer)

        snippet = buffer[line_start:line_end]
        caret_line = ' ' * (col - 1) + '^'
        msg = (
            f"error: unexpected token '{value}'\n"
            f" --> input:{line}:{col}\n"
            f"  |\n"
            f"{line} | {snippet}\n"
            f"  | {caret_line}"
        )

        return msg

    @staticmethod
    def build_ast_error_msg(token: Token, buffer: str, context: str = "") -> str:
        line = buffer[:token.position].count('\n') + 1
        line_start = buffer.rfind('\n', 0, token.position) + 1
        line_end = buffer.find('\n', token.position)
        if line_end == -1:
            line_end = len(buffer)

        col = token.position - line_start + 1
        snippet = buffer[line_start:line_end]
        caret_line = ' ' * (col - 1) + '^'
        return (
            f"error: AST error at token '{token.value}' {context}\n"
            f" --> input:{line}:{col}\n"
            f"  |\n"
            f"{line} | {snippet}\n"
            f"  | {caret_line}"
        )
