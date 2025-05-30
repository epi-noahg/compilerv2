from lexer.lexer import Token
from parser.ast import AstNode


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens

    def parse(self):
        pass
