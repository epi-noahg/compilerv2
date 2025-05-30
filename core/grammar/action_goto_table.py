from core.grammar.production import Production
from core.grammar.item import Item
from core.grammar.closure import closure
from core.lexer.token import TokenType
from core.tools import find_state_index, goto

def build_action_goto(states: list[set[Item]], grammar: list[Production], follow: dict[str, set[TokenType]]):
    action = {}  # ex. {(state_idx, terminal): action_str}
    goto_table = {}  # ex. {(state_idx, nonterminal): next_state}
    for i, I in enumerate(states):
        for item in I:
            A = item.production.lhs
            # SHIFT
            if item.dot < len(item.production.rhs):
                sym = item.production.rhs[item.dot]
                if isinstance(sym, TokenType):
                    # transition avec ce terminal
                    j = find_state_index(goto(I, sym, grammar), states)
                    action[(i, sym)] = f"shift {j}"
                else:
                    # sym est non-terminal
                    j = find_state_index(goto(I, sym, grammar), states)
                    goto_table[(i, sym)] = j
            else:
                # ITEM complet A -> alpha·
                if A != "S'":  # Si ce n'est pas l'item augmenté final
                    for a in follow[A]:
                        action[(i, a)] = f"reduce {A}->{''.join(str(x) for x in item.production.rhs)}"
                else:
                    # A == S' (start symbol), on accepte sur $
                    action[(i, TokenType.EOF)] = "accept"
    return action, goto_table
