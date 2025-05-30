from core.grammar.production import Production
from core.grammar.item import Item


def closure(items: set[Item], grammar: list[Production]) -> set[Item]:
    closure_set = set(items)
    changed = True
    while changed:
        changed = False
        new_items = set()
        for item in closure_set:
            if item.dot < len(item.production.rhs):
                symbol = item.production.rhs[item.dot]
                if isinstance(symbol, str):  # non-terminal
                    for prod in grammar:
                        if prod.lhs == symbol:
                            new_item = Item(prod, 0)
                            if new_item not in closure_set:
                                new_items.add(new_item)
        if new_items:
            closure_set.update(new_items)
            changed = True
    return closure_set
