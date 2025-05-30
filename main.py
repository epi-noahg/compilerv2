from core.grammar.production import Production
from core.grammar.first_follow import compute_first_follow
from core.grammar.closure import closure
from core.grammar.lr0_states import build_lr0_states
from core.grammar.action_goto_table import build_action_goto
from core.lexer.token import TokenType

SMALL_PRODUCTIONS: list[Production] = [
    Production('S', ('E',)),
    Production('E', ('E', TokenType.PLUS, 'T')),
    Production('E', ('T',)),
    Production('T', ('T', TokenType.MULTIPLY, 'F')),
    Production('T', ('F',)),
    Production('F', (TokenType.LEFT_PAREN, 'E', TokenType.RIGHT_PAREN)),
    Production('F', (TokenType.IDENTIFIER,)),
]


grammar = SMALL_PRODUCTIONS
first, follow = compute_first_follow(grammar)
states = build_lr0_states(grammar)
action, goto_table = build_action_goto(states, grammar, follow)
def print_action_goto(action, goto_table):
    print("Action Table:")
    for (state, terminal), act in action.items():
        print(f"State {state}, Terminal {terminal}: {act}")
    
    print("\nGoto Table:")
    for (state, nonterminal), next_state in goto_table.items():
        print(f"State {state}, Nonterminal {nonterminal}: Goto State {next_state}")
print_action_goto(action, goto_table)
