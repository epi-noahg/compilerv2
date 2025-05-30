# core/grammar/action_goto_table.py
from core.grammar.production import Production
from core.grammar.item import Item
from core.grammar.closure import closure
from core.lexer.token import TokenType
from core.tools import find_state_index, goto

def build_action_goto(states: list[set[Item]], grammar: list[Production], follow: dict[str, set[TokenType]]):
    action = {}
    goto_table = {}
    for i, I in enumerate(states):
        for item in I:
            A = item.production.lhs
            # SHIFT
            if item.dot < len(item.production.rhs):
                sym = item.production.rhs[item.dot]
                j = find_state_index(goto(I, sym, grammar), states)
                if j == -1:
                    continue
                if isinstance(sym, TokenType):
                    action[(i, sym)] = f"shift {j}"
                else:
                    goto_table[(i, sym)] = j
            else:
                # REDUCE or ACCEPT
                if A != "S'":
                    key = f"{A}->{' '.join(str(x.value if isinstance(x, TokenType) else x) for x in item.production.rhs)}"
                    for a in follow[A]:
                        action[(i, a)] = f"reduce {key}"
                else:
                    action[(i, TokenType.EOF)] = "accept"
    return action, goto_table
