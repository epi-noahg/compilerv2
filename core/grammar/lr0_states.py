from collections import defaultdict
from core.grammar.production import Production
from core.grammar.item import Item
from core.grammar.closure import closure
from core.tools import goto, all_symbols

def build_lr0_states(grammar: list[Production]) -> list[set[Item]]:
    # grammar[0] est S (déjà augmenté); S' ajouté par le code appelant si besoin
    start_prod = grammar[0]
    initial_item = Item(start_prod, 0)
    states = []
    state0 = closure({initial_item}, grammar)
    states.append(state0)
    while True:
        new_state_added = False
        for I in states:
            for X in all_symbols(grammar):  # liste de tous symboles (str et TokenType)
                J = goto(I, X, grammar)
                if J and J not in states:
                    states.append(J)
                    new_state_added = True
        if not new_state_added:
            break
    return states
