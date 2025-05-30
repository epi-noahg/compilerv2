from typing import Optional
from enum import Enum

class TokenType(Enum):
    # Keywords
    TYPE = 1        # type
    IDENTIFIER = 2  # id
    FOR = 3         # for
    WHILE = 4  # while
    IF = 5  # if
    ELSE = 6  # else
    RETURN = 7  # return
    NUMBER = 8  # num

    # Operators
    COMPARE = 9  # ==
    ASSIGN = 10  # =
    PLUS = 11  # +
    MINUS = 12  # -
    MULTIPLY = 13  # *

    # Separators
    SEMICOLON = 14  # ;
    COMMA = 15  # ,
    LEFT_PAREN = 16  # (
    RIGHT_PAREN = 17  # )
    LEFT_BRACE = 18  # {
    RIGHT_BRACE = 19  # }

    EOF = 20 # End of File

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
