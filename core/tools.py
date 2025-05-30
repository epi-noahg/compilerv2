
from core.grammar.production import Production, Symbol
from core.grammar.item import Item
from core.grammar.closure import closure

def find_state_index(I: set[Item], C: list[set[Item]]) -> int:
    for idx, state in enumerate(C):
        if state == I:
            return idx
    return -1


def all_symbols(grammar: list[Production]) -> set[Symbol]:
    symboles = set()
    for p in grammar:
        symboles.update(p.rhs)
    return symboles


def goto(I: set[Item], X: Symbol, grammar: list[Production]) -> set[Item]:
    moved = set()
    for item in I:
        if item.dot < len(item.production.rhs) and item.production.rhs[item.dot] == X:
            moved.add(Item(item.production, item.dot + 1))
    return closure(moved, grammar)
