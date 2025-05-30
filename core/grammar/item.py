from dataclasses import dataclass
from core.grammar.production import Production

@dataclass(frozen=True)
class Item:
    production: Production
    dot: int  # position du point dans rhs

    def __str__(self) -> str:
        before = ' '.join(str(sym) for sym in self.production.rhs[:self.dot])
        after  = ' '.join(str(sym) for sym in self.production.rhs[self.dot:])
        return f"{self.production.lhs} -> {before}·{(' ' + after) if after else ''}"
