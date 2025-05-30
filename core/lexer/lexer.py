from core.lexer.token import Token, TokenType
from typing import Dict
from enum import Enum


class ScanState(Enum):
    UNINITIALIZED = 0
    SUCCESS = 1
    FAILURE = 2


class Lexer:
    keywords: Dict[str, TokenType] = {
        'type': TokenType.TYPE,
        'for': TokenType.FOR,
        'while': TokenType.WHILE,
        'if': TokenType.IF,
        'else': TokenType.ELSE,
        'return': TokenType.RETURN,
        'num': TokenType.NUMBER,
        'id': TokenType.IDENTIFIER,
    }

    symbols: Dict[str, TokenType] = {
        '==': TokenType.COMPARE,
        '=': TokenType.ASSIGN,
        '+': TokenType.PLUS,
        '-': TokenType.MINUS,
        '*': TokenType.MULTIPLY,
        ';': TokenType.SEMICOLON,
        ',': TokenType.COMMA,
        '(': TokenType.LEFT_PAREN,
        ')': TokenType.RIGHT_PAREN,
        '{': TokenType.LEFT_BRACE,
        '}': TokenType.RIGHT_BRACE,
    }

    def __init__(self, buffer: str) -> None:
        self.buffer: str = buffer
        self.position: int = 0
        self.tokens: list[Token] = []
        self.scan_state: ScanState = ScanState.UNINITIALIZED

    def __str__(self) -> str:
        string: str = ''
        for token in self.tokens:
            string += f'{token}\n'
        return string

    def __repr__(self) -> str:
        return self.__str__()

    def scan(self) -> list[Token]:
        buffer: str = self.buffer
        matched: bool = False

        def add_token(token_type: TokenType, length: int) -> None:
            nonlocal buffer, matched
            token_value = buffer[:length]
            self.tokens.append(Token(token_type, self.position, length, token_value))
            buffer = buffer[length:]
            self.position += length
            matched = True

        while buffer:
            if buffer[0].isspace():
                buffer = buffer[1:]
                self.position += 1
                continue
            
            matched = False
            
            for keyword, token_type in self.keywords.items():
                if buffer.startswith(keyword):
                    add_token(token_type, len(keyword))
                    break

            if matched:
                continue
            
            for symbol, token_type in self.symbols.items():
                if buffer.startswith(symbol):
                    add_token(token_type, len(symbol))
                    break
            
            if not matched:
                add_token(TokenType.UNEXPECTED_TOKEN, 1)
                self.scan_state = ScanState.FAILURE
                break
        
        if self.scan_state == ScanState.UNINITIALIZED:
            self.scan_state = ScanState.SUCCESS
        return self.tokens
