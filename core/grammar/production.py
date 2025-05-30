from dataclasses import dataclass
from typing import Tuple, Union
from core.lexer.token import TokenType  # exemple d’import de TokenType

Symbol = str | TokenType  # union de str et TokenType (Python 3.10+)

@dataclass(frozen=True)
class Production:
    lhs: str
    rhs: Tuple[Symbol, ...]  # tuple de symboles (non-terminaux ou terminaux)

    def __str__(self) -> str:
        return f"{self.lhs} -> {' '.join(str(s) for s in self.rhs)}"
