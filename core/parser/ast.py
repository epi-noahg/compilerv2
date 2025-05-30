# core/parser/ast.py
from __future__ import annotations
from typing import List, Optional, Union
from dataclasses import dataclass
from core.lexer.token import Token


class AstNode:
    """Base class for all AST nodes."""
    pass


@dataclass
class Identifier(AstNode):
    token: Token

    def __str__(self):
        return f"Identifier({self.token.text})"


@dataclass
class Literal(AstNode):
    token: Token

    def __str__(self):
        return f"Literal({self.token.text})"


@dataclass
class Program(AstNode):
    decl_list: Optional[DeclList]

    def __str__(self):
        return f"Program({self.decl_list})"


@dataclass
class DeclList(AstNode):
    decls: List[Decl]

    def __str__(self):
        return f"DeclList({self.decls})"


@dataclass
class Decl(AstNode):
    decl: Union[VarDecl, FuncDecl]

    def __str__(self):
        return f"Decl({self.decl})"


@dataclass
class Type(AstNode):
    token: Token

    def __str__(self):
        return f"Type({self.token.text})"


@dataclass
class VarDecl(AstNode):
    type_: Type
    name: Token
    init_expr: Optional[AstNode]

    def __str__(self):
        return f"VarDecl(type={self.type_}, name={self.name.text}, init={self.init_expr})"


@dataclass
class FuncDecl(AstNode):
    return_type: Type
    name: Token
    params: ParamList
    body: Block

    def __str__(self):
        return f"FuncDecl(return_type={self.return_type}, name={self.name.text}, params={self.params}, body={self.body})"


@dataclass
class ParamList(AstNode):
    params: List[Param]

    def __str__(self):
        return f"ParamList({self.params})"


@dataclass
class Param(AstNode):
    type_: Type
    name: Token

    def __str__(self):
        return f"Param(type={self.type_}, name={self.name.text})"


@dataclass
class Block(AstNode):
    statements: StmtList

    def __str__(self):
        return f"Block({self.statements})"


@dataclass
class StmtList(AstNode):
    statements: List[AstNode]

    def __str__(self):
        return f"StmtList({self.statements})"


@dataclass
class IfStmt(AstNode):
    condition: AstNode
    then_branch: AstNode
    else_branch: Optional[AstNode] = None

    def __str__(self):
        return f"IfStmt(cond={self.condition}, then={self.then_branch}, else={self.else_branch})"


@dataclass
class WhileStmt(AstNode):
    condition: AstNode
    body: AstNode

    def __str__(self):
        return f"WhileStmt(cond={self.condition}, body={self.body})"


@dataclass
class ForStmt(AstNode):
    init: Optional[AstNode]
    condition: Optional[AstNode]
    update: Optional[AstNode]
    body: AstNode

    def __str__(self):
        return f"ForStmt(init={self.init}, cond={self.condition}, update={self.update}, body={self.body})"


@dataclass
class ReturnStmt(AstNode):
    expr: AstNode

    def __str__(self):
        return f"ReturnStmt({self.expr})"


@dataclass
class ExprStmt(AstNode):
    expr: AstNode

    def __str__(self):
        return f"ExprStmt({self.expr})"


@dataclass
class BinaryOp(AstNode):
    left: AstNode
    op: Token
    right: AstNode

    def __str__(self):
        return f"BinaryOp({self.left} {self.op.text} {self.right})"


@dataclass
class UnaryOp(AstNode):
    op: Token
    operand: AstNode

    def __str__(self):
        return f"UnaryOp({self.op.text}{self.operand})"
