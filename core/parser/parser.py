# core/parser/parser.py
from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass

# --- run-time imports kept local to avoid circular dependencies -------------
from core.lexer.token import Token, TokenType
from core.grammar.production import Production, PRODUCTIONS
from core.grammar.first_follow import compute_first_follow
from core.grammar.lr0_states import build_lr0_states
from core.grammar.action_goto_table import build_action_goto
from core.parser.ast import (
    AstNode, Identifier, Literal, Program, DeclList, Decl, Type, VarDecl,
    FuncDecl, ParamList, Param, Block, StmtList, IfStmt, WhileStmt, ForStmt,
    ReturnStmt, ExprStmt, BinaryOp, UnaryOp
)
# ---------------------------------------------------------------------------


def _ensure_index(tok):
    if not hasattr(tok, "index"):
        if hasattr(tok, "position"):
            setattr(tok, "index", tok.position)
        else:
            # fallback : si pas de position, donne une valeur par défaut ou lève une erreur
            setattr(tok, "index", -1)



class Parser:
    """SLR(1) shift–reduce parser that turns a token stream into an AST."""

    # Convenience inner class: one stack frame holds the LR state
    # and the semantic value (AstNode or Token) produced for that symbol.
    @dataclass
    class _Frame:
        state: int
        node: Any          # AstNode | Token | None

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens: List[Token] = tokens

        # Append an explicit EOF token if the caller hasn’t done so
        if not self.tokens or self.tokens[-1].type is not TokenType.EOF:
            self.tokens.append(
                Token(TokenType.EOF,
                      self.tokens[-1].position + self.tokens[-1].length if self.tokens else 0,
                      0, "")
            )

        # Build action / goto tables once for this parser instance
        # (we include an augmented production S' → Program).
        augmented = [Production("S'", ("Program",))] + PRODUCTIONS
        first, follow = compute_first_follow(augmented)
        states = build_lr0_states(augmented)
        self._action, self._goto = build_action_goto(states, augmented, follow)

        # Map "A->αβ" strings (made by build_action_goto) back to Production
        self._prod_map: Dict[str, Production] = {
            f"{p.lhs}->{''.join(str(x) for x in p.rhs)}": p
            for p in augmented
        }

    def parse(self, debug: bool = False) -> AstNode:
        """Parse the token stream and return the AST root.
        Set *debug=True* to see a step-by-step trace of shift/reduce actions."""
        stack: List[Parser._Frame] = [self._Frame(0, None)]
        i: int = 0

        while True:
            state = stack[-1].state
            lookahead: Token = self.tokens[i]
            act: Optional[str] = self._action.get((state, lookahead.type))

            if act is None:
                raise SyntaxError(f"Unexpected token {lookahead} in state {state}")

            if act.startswith("shift"):
                tgt_state = int(act.split()[1])

                if lookahead.type is TokenType.IDENTIFIER:
                    _ensure_index(lookahead)
                    node: Any = Identifier(lookahead)
                elif lookahead.type is TokenType.NUMBER:
                    _ensure_index(lookahead)
                    node = Literal(lookahead)
                else:
                    node = lookahead

                stack.append(self._Frame(tgt_state, node))
                i += 1

                if debug:
                    self._dbg("SHIFT", lookahead, stack)

            elif act.startswith("reduce"):
                prod_key = act[len("reduce "):]
                prod = self._prod_map[prod_key]
                k = len(prod.rhs)

                children: List[Any] = []
                for _ in range(k):
                    children.append(stack.pop().node)
                children.reverse()

                lhs_node = self._make_node(prod, children)
                goto_state = self._goto[(stack[-1].state, prod.lhs)]
                stack.append(self._Frame(goto_state, lhs_node))

                if debug:
                    self._dbg(f"REDUCE {prod.lhs} -> {' '.join(map(str, prod.rhs))}", lhs_node, stack)

            elif act == "accept":
                if debug:
                    self._dbg("ACCEPT", stack[-1].node, stack)
                return stack[-1].node

            else:
                raise RuntimeError(f"Unknown parser action {act}")

    def _dbg(self, op: str, obj: Any, stack: List[_Frame]) -> None:
        def _name(x: Any) -> str:
            if isinstance(x, AstNode):
                return x.__class__.__name__
            if isinstance(x, Token):
                return x.type.name
            return str(x)
        st = "[" + ", ".join(str(f.state) for f in stack) + "]"
        sym = _name(obj)
        print(f"{op:<10} {sym:<20}  stack={st}")

    def _make_node(self, p: Production, c: List[Any]) -> Any:
        """Return an AST node (or other semantic value) for production *p* with
        RHS children *c* (already in left-to-right order)."""

        if p.lhs == "Program":
            return Program(c[0] if c else None)

        if p.lhs == "DeclList":
            if not c:
                return DeclList([])
            decl, lst = c
            if isinstance(lst, DeclList):
                return DeclList([decl] + lst.decls)
            return DeclList([decl])

        if p.lhs == "Decl":
            return Decl(c[0])

        if p.lhs == "VarDecl":
            type_tok = c[0]
            _ensure_index(type_tok)
            type_node = Type(type_tok)
            name_tok = c[1]
            if len(p.rhs) == 3:
                return VarDecl(type_node, name_tok, None)
            expr = c[3]
            return VarDecl(type_node, name_tok, expr)

        if p.lhs == "FuncDecl":
            type_tok, name_tok, _lp, params, _rp, block = c
            _ensure_index(type_tok)
            type_node = Type(type_tok)
            return FuncDecl(type_node, name_tok, params, block)

        if p.lhs == "ParamList":
            if not c:
                return ParamList([])
            param, tail = c
            return ParamList([param] + tail)

        if p.lhs == "ParamTail":
            if not c:
                return []
            _comma, param, tail = c
            return [param] + tail

        if p.lhs == "Param":
            type_tok, name_tok = c
            _ensure_index(type_tok)
            return Param(Type(type_tok), name_tok)

        if p.lhs == "Block":
            _lb, stmt_list, _rb = c
            return Block(stmt_list)

        if p.lhs == "StmtList":
            if not c:
                return StmtList([])
            stmt, tail = c
            return StmtList([stmt] + tail.statements)

        if p.lhs in ("Stmt", "MatchedStmt", "UnmatchedStmt"):
            return c[0]

        if p.lhs == "MatchedStmt" and len(p.rhs) == 7 and p.rhs[0] is TokenType.IF:
            _if, _lp, cond, _rp, then_m, _else, else_m = c
            return IfStmt(cond, then_m, else_m)

        if p.lhs in ("MatchedStmt", "UnmatchedStmt") and p.rhs[0] is TokenType.WHILE:
            _wh, _lp, cond, _rp, body = c
            return WhileStmt(cond, body)

        if p.lhs in ("MatchedStmt", "UnmatchedStmt") and p.rhs[0] is TokenType.FOR:
            (_for, _lp, init, _sc1, cond, _sc2, update, _rp, body) = c
            return ForStmt(init, cond, update, body)

        if p.lhs == "MatchedStmt" and p.rhs[0] is TokenType.RETURN:
            _ret, expr, _semi = c
            return ReturnStmt(expr)

        if p.lhs == "MatchedStmt" and p.rhs[0] in ("VarDecl", "ExprStmt", "Block"):
            return c[0]

        if p.lhs == "UnmatchedStmt" and p.rhs[0] is TokenType.IF and len(p.rhs) == 5:
            _if, _lp, cond, _rp, stmt = c
            return IfStmt(cond, stmt, None)

        if p.lhs == "UnmatchedStmt" and p.rhs[0] is TokenType.IF and len(p.rhs) == 7:
            _if, _lp, cond, _rp, then_m, _else, else_um = c
            return IfStmt(cond, then_m, else_um)

        if p.lhs == "ExprStmt":
            id_tok, assign_tok, expr, _semi = c
            _ensure_index(id_tok)
            binop = BinaryOp(Identifier(id_tok), assign_tok, expr)
            return ExprStmt(binop)

        if len(p.rhs) == 1 and p.rhs[0] not in (TokenType.MINUS,):
            return c[0]

        if len(p.rhs) == 3 and isinstance(p.rhs[1], TokenType):
            left, op_tok, right = c
            return BinaryOp(left, op_tok, right)

        if p.lhs == "UnaryExpr" and len(c) == 2:
            op_tok, expr = c
            return UnaryOp(op_tok, expr)

        if p.lhs == "PrimaryExpr" and len(c) == 1:
            return c[0]

        # Add more production rules as needed...

        # Fallback: return first child or None
        return c[0] if c else None
