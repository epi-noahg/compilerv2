# core/lexer/token.py
from typing import Optional
from enum import Enum

class TokenType(Enum):
    TYPE = 1
    IDENTIFIER = 2
    FOR = 3
    WHILE = 4
    IF = 5
    ELSE = 6
    RETURN = 7
    NUMBER = 8

    COMPARE = 9
    ASSIGN = 10
    PLUS = 11
    MINUS = 12
    MULTIPLY = 13

    SEMICOLON = 14
    COMMA = 15
    LEFT_PAREN = 16
    RIGHT_PAREN = 17
    LEFT_BRACE = 18
    RIGHT_BRACE = 19

    EOF = 20
    UNEXPECTED_TOKEN = 21

class Token:
    def __init__(self, type: TokenType, position: int, length: int, value: Optional[str] = None) -> None:
        self.type: TokenType = type
        self.position: int = position
        self.length: int = length
        self.value: Optional[str] = value

    def __str__(self) -> str:
        return f"Token(type={self.type}, position={self.position}, length={self.length}, value='{self.value}')"

    def __repr__(self) -> str:
        return self.__str__()
