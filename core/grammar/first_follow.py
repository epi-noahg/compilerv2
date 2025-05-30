from core.lexer.token import TokenType
from core.grammar.production import Production
from core.grammar.production import Symbol

def compute_first_follow(grammar: list[Production]) -> tuple[dict[str, set[TokenType]], dict[str, set[TokenType]]]:
    # Initialisation
    first: dict[str, set[TokenType]] = {p.lhs: set() for p in grammar}
    follow: dict[str, set[TokenType]] = {p.lhs: set() for p in grammar}
    # Supposons un non-terminal de départ S
    start = grammar[0].lhs
    follow[start].add(TokenType.EOF)  # $ dans FOLLOW(start)
    
    # Calcul de FIRST
    changed = True
    while changed:
        changed = False
        for p in grammar:
            X = p.lhs
            rhs = p.rhs
            if not rhs:
                continue
            # si le premier symbole est terminal
            first_sym = rhs[0]
            new_first = set()
            if isinstance(first_sym, TokenType):
                new_first.add(first_sym)
            else:
                # si non-terminal, ajouter FIRST(non-terminal)
                new_first |= first[first_sym]
            if not new_first.issubset(first[X]):
                first[X] |= new_first
                changed = True
    
    # Calcul de FOLLOW (après FIRST)
    changed = True
    while changed:
        changed = False
        for p in grammar:
            X = p.lhs
            rhs = p.rhs
            trailer = follow[X].copy()
            for sym in reversed(rhs):
                if isinstance(sym, TokenType):
                    trailer = {sym}
                else:
                    # sym est non-terminal
                    if not trailer.issubset(follow[sym]):
                        follow[sym] |= trailer
                        changed = True
                    # si FIRST(sym) contient ε, trailer reste + FOLLOW(X), sinon trailer = FIRST(sym)
                    # (ici on néglige ε pour simplifier)
                    trailer = trailer | first[sym]
    return first, follow
