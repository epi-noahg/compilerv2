from core.lexer.token import TokenType
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
    def build_parser_error_msg() -> str:
        return ""
