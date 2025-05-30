from core.grammar.production import Production
from core.lexer.lexer import Lexer, ScanState
from core.lexer.token import TokenType
from core.error.error import Error

from core.grammar.production import Production
from core.grammar.first_follow import compute_first_follow
from core.grammar.closure import closure
from core.grammar.lr0_states import build_lr0_states
from core.grammar.action_goto_table import build_action_goto
from core.lexer.token import TokenType
from core.parser.parser import Parser
from core.grammar.production import PRODUCTIONS



def parse(buffer: str) -> str:
    grammar = PRODUCTIONS
    lexer = Lexer(buffer)
    tokens = lexer.scan()
    if lexer.scan_state == ScanState.FAILURE:
        return Error.build_lexer_error_msg(lexer)
    parser = Parser(tokens)
    ast = parser.parse(debug=True)
    return f"{ast}"


def print_action_goto(action, goto_table):
    print("Action Table:")
    for (state, terminal), act in action.items():
        print(f"State {state}, Terminal {terminal}: {act}")
    
    print("\nGoto Table:")
    for (state, nonterminal), next_state in goto_table.items():
        print(f"State {state}, Nonterminal {nonterminal}: Goto State {next_state}")
