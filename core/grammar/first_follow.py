# core/grammar/first_follow.py
from core.lexer.token import TokenType
from core.grammar.production import Production
from core.grammar.production import Symbol

def compute_first_follow(grammar: list[Production]) -> tuple[dict[str, set[TokenType]], dict[str, set[TokenType]]]:
    first: dict[str, set[TokenType]] = {p.lhs: set() for p in grammar}
    follow: dict[str, set[TokenType]] = {p.lhs: set() for p in grammar}
    nullable: dict[str, bool] = {p.lhs: False for p in grammar}

    start = grammar[0].lhs
    follow[start].add(TokenType.EOF)

    # Compute nullable
    changed = True
    while changed:
        changed = False
        for p in grammar:
            if not p.rhs:
                if not nullable[p.lhs]:
                    nullable[p.lhs] = True
                    changed = True
            elif all((isinstance(sym, str) and nullable.get(sym, False)) or isinstance(sym, TokenType) for sym in p.rhs):
                if not nullable[p.lhs]:
                    nullable[p.lhs] = True
                    changed = True

    # Compute FIRST sets
    changed = True
    while changed:
        changed = False
        for p in grammar:
            lhs = p.lhs
            rhs = p.rhs
            trailer = set()
            for sym in rhs:
                if isinstance(sym, TokenType):
                    if sym not in first[lhs]:
                        first[lhs].add(sym)
                        changed = True
                    break
                else:
                    before = len(first[lhs])
                    first[lhs].update(first[sym])
                    if before != len(first[lhs]):
                        changed = True
                    if not nullable[sym]:
                        break
            else:
                pass  # All symbols are nullable — could include ε if needed

    # Compute FOLLOW sets
    changed = True
    while changed:
        changed = False
        for p in grammar:
            trailer = follow[p.lhs].copy()
            for sym in reversed(p.rhs):
                if isinstance(sym, TokenType):
                    trailer = {sym}
                else:
                    before = len(follow[sym])
                    follow[sym].update(trailer)
                    if len(follow[sym]) != before:
                        changed = True
                    if nullable.get(sym, False):
                        trailer = trailer.union(first[sym])
                    else:
                        trailer = first[sym]

    return first, follow
