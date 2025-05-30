from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TokenType(Enum):
    """
    Enumeration of all token types recognized by the lexer.
    These represent the fundamental building blocks of the language syntax.

    The tokens are grouped into categories:
    - Keywords (TYPE, FOR, WHILE, IF, ELSE, RETURN)
    - Identifiers and literals (IDENTIFIER, NUMBER)
    - Operators (COMPARE, ASSIGN, PLUS, MINUS, MULTIPLY)
    - Punctuation (SEMICOLON, COMMA, parentheses, braces)
    - Special tokens (EOF, UNEXPECTED_TOKEN)
    """
    TYPE = 1
    FOR = 2
    WHILE = 3
    IF = 4
    ELSE = 5
    RETURN = 6

    IDENTIFIER = 7
    NUMBER = 8

    COMPARE = 9
    ASSIGN = 10
    PLUS = 11
    MINUS = 12
    MULTIPLY = 13

    SEMICOLON = 14
    COMMA = 15
    LEFT_PAREN = 16
    RIGHT_PAREN = 17
    LEFT_BRACE = 18
    RIGHT_BRACE = 19

    EOF = 20
    UNEXPECTED_TOKEN = 21

class Token:
    """
    Represents a lexical token in the source code.

    A token is the smallest meaningful unit in the language syntax, such as keywords,
    identifiers, operators, and punctuation marks. Each token has a type, position in the
    source code, length, and an optional string value.
    """
    def __init__(self, type: TokenType, position: int, length: int, value: Optional[str] = None) -> None:
        """
        Initialize a new Token instance.

        Args:
            type: The TokenType enum value representing the token's category
            position: The starting position (index) of the token in the source code
            length: The length of the token in characters
            value: The string value of the token (optional)
        """
        self.type: TokenType = type
        self.position: int = position
        self.length: int = length
        self.value: Optional[str] = value

    def __str__(self) -> str:
        """Returns a string representation of the token for debugging purposes."""
        return f"Token(type={self.type}, position={self.position}, length={self.length}, value='{self.value}')"

    def __repr__(self) -> str:
        """Returns the same string representation as __str__ for consistency."""
        return self.__str__()


Symbol = str | TokenType
"""
A Symbol can be either a string (representing a non-terminal symbol) or a TokenType (representing a terminal symbol).
This type alias is used throughout the grammar definition and parsing algorithms.
"""

class Production:
    """
    Represents a production rule in the context-free grammar.

    A production rule defines how a non-terminal symbol (lhs) can be expanded into a sequence
    of terminal and non-terminal symbols (rhs). These rules form the basis of the language's
    syntax definition and are used by the parser to analyze the structure of the source code.
    """
    lhs: Symbol  # Left-hand side: a non-terminal symbol
    rhs: tuple[Symbol, ...]  # Right-hand side: sequence of symbols (terminals and non-terminals)

    def __init__(self, lhs: Symbol, rhs: tuple[Symbol, ...]):
        """
        Initialize a new Production instance.

        Args:
            lhs: The left-hand side non-terminal symbol
            rhs: A tuple of symbols (terminals and non-terminals) that the lhs expands to
        """
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        """Returns a string representation of the production in standard notation: LHS -> RHS"""
        return f"{self.lhs} -> {' '.join(str(s) for s in self.rhs)}"


"""
The complete context-free grammar for the language.

This grammar defines the syntax of the language using production rules. Each rule specifies
how a non-terminal symbol can be expanded into a sequence of terminal and non-terminal symbols.

The grammar includes rules for:
- Program structure (Program, DeclList)
- Declarations (variable and function declarations)
- Statements (if, while, for, return, blocks)
- Expressions (equality, additive, multiplicative, unary, primary)
- Parameters and arguments

The grammar is designed to handle the dangling-else problem using matched and unmatched statements.
"""
PRODUCTIONS: list[Production] = [
    Production('Program', ('DeclList',)),
    Production('DeclList', ('Decl', 'DeclList')),
    Production('DeclList', ()),
    Production('Decl', ('VarDecl',)),
    Production('Decl', ('FuncDecl',)),
    Production('VarDecl', (TokenType.TYPE, TokenType.IDENTIFIER, TokenType.SEMICOLON)),
    Production('VarDecl', (TokenType.TYPE, TokenType.IDENTIFIER, TokenType.ASSIGN, 'Expr', TokenType.SEMICOLON)),
    Production('FuncDecl', (TokenType.TYPE, TokenType.IDENTIFIER, TokenType.LEFT_PAREN, 'ParamList', TokenType.RIGHT_PAREN, 'Block')),
    Production('ParamList', ('Param', 'ParamTail')),
    Production('ParamList', ()),
    Production('ParamTail', (TokenType.COMMA, 'Param', 'ParamTail')),
    Production('ParamTail', ()),
    Production('Param', (TokenType.TYPE, TokenType.IDENTIFIER)),
    Production('Block', (TokenType.LEFT_BRACE, 'StmtList', TokenType.RIGHT_BRACE)),
    Production('StmtList', ('Stmt', 'StmtList')),
    Production('StmtList', ()),
    Production('Stmt', ('MatchedStmt',)),
    Production('Stmt', ('UnmatchedStmt',)),
    Production('MatchedStmt', (TokenType.IF, TokenType.LEFT_PAREN, 'Expr', TokenType.RIGHT_PAREN, 'MatchedStmt', TokenType.ELSE, 'MatchedStmt')),
    Production('MatchedStmt', (TokenType.WHILE, TokenType.LEFT_PAREN, 'Expr', TokenType.RIGHT_PAREN, 'MatchedStmt')),
    Production('MatchedStmt', (TokenType.FOR, TokenType.LEFT_PAREN, 'Expr', TokenType.SEMICOLON, 'Expr', TokenType.SEMICOLON, 'Expr', TokenType.RIGHT_PAREN, 'MatchedStmt')),
    Production('MatchedStmt', (TokenType.RETURN, 'Expr', TokenType.SEMICOLON)),
    Production('MatchedStmt', ('VarDecl',)),
    Production('MatchedStmt', ('ExprStmt',)),
    Production('MatchedStmt', ('Block',)),
    Production('UnmatchedStmt', (TokenType.IF, TokenType.LEFT_PAREN, 'Expr', TokenType.RIGHT_PAREN, 'Stmt')),
    Production('UnmatchedStmt', (TokenType.IF, TokenType.LEFT_PAREN, 'Expr', TokenType.RIGHT_PAREN, 'MatchedStmt', TokenType.ELSE, 'UnmatchedStmt')),
    Production('UnmatchedStmt', (TokenType.WHILE, TokenType.LEFT_PAREN, 'Expr', TokenType.RIGHT_PAREN, 'UnmatchedStmt')),
    Production('UnmatchedStmt', (TokenType.FOR, TokenType.LEFT_PAREN, 'Expr', TokenType.SEMICOLON, 'Expr', TokenType.SEMICOLON, 'Expr', TokenType.RIGHT_PAREN, 'UnmatchedStmt')),
    Production('ExprStmt', (TokenType.IDENTIFIER, TokenType.ASSIGN, 'Expr', TokenType.SEMICOLON)),
    Production('Expr', ('EqualityExpr',)),
    Production('EqualityExpr', ('AdditiveExpr',)),
    Production('EqualityExpr', ('AdditiveExpr', TokenType.COMPARE, 'EqualityExpr')),
    Production('AdditiveExpr', ('MultiplicativeExpr',)),
    Production('AdditiveExpr', ('AdditiveExpr', TokenType.PLUS, 'MultiplicativeExpr')),
    Production('MultiplicativeExpr', ('UnaryExpr',)),
    Production('MultiplicativeExpr', ('MultiplicativeExpr', TokenType.MULTIPLY, 'UnaryExpr')),
    Production('UnaryExpr', (TokenType.MINUS, 'UnaryExpr')),
    Production('UnaryExpr', ('PrimaryExpr',)),
    Production('PrimaryExpr', (TokenType.IDENTIFIER, TokenType.LEFT_PAREN, 'ArgList', TokenType.RIGHT_PAREN)),
    Production('PrimaryExpr', (TokenType.IDENTIFIER,)),
    Production('PrimaryExpr', (TokenType.NUMBER,)),
    Production('PrimaryExpr', (TokenType.LEFT_PAREN, 'Expr', TokenType.RIGHT_PAREN)),
    Production('ArgList', ('Expr', TokenType.COMMA, 'ArgList')),
    Production('ArgList', ('Expr',)),
    Production('ArgList', ()),
]

@dataclass(frozen=True)
class Item:
    """
    Represents an LR(0) item used in the construction of the parsing tables.

    An LR(0) item is a production rule with a "dot" indicating how much of the rule
    has been recognized so far during parsing. The dot position is crucial for determining
    parser states and actions.

    For example, if we have a production A -> XYZ, the possible items would be:
    - A -> •XYZ (dot at position 0)
    - A -> X•YZ (dot at position 1)
    - A -> XY•Z (dot at position 2)
    - A -> XYZ• (dot at position 3)

    This class is frozen (immutable) to allow its use as dictionary keys and in sets.
    """
    production: Production  # The underlying production rule
    dot: int  # Position of the dot in the right-hand side (0 to len(rhs))

    def __str__(self) -> str:
        """
        Returns a string representation of the item with the dot (•) showing the current position.
        For example: "A -> X • Y Z"
        """
        before = ' '.join(str(sym) for sym in self.production.rhs[:self.dot])
        after  = ' '.join(str(sym) for sym in self.production.rhs[self.dot:])
        return f"{self.production.lhs} -> {before}·{(' ' + after) if after else ''}"

class ScanState(Enum):
    """
    Represents the possible states of the lexical scanner (lexer).

    These states are used to track the progress and outcome of the lexical analysis process:
    - UNINITIALIZED: Initial state before scanning begins
    - SUCCESS: Scanning completed successfully with no errors
    - FAILURE: Scanning encountered an error (e.g., unexpected character)
    """
    UNINITIALIZED = 0
    SUCCESS = 1
    FAILURE = 2

class Lexer:
    """
    Lexical analyzer (scanner) that converts source code text into a sequence of tokens.

    The lexer reads the input character by character, recognizing patterns that match
    language elements like keywords, identifiers, operators, and punctuation. It produces
    a stream of tokens that can be consumed by the parser.
    """
    # Dictionary mapping keyword strings to their corresponding token types
    keywords: dict[str, TokenType] = {
        'type': TokenType.TYPE,
        'for': TokenType.FOR,
        'while': TokenType.WHILE,
        'if': TokenType.IF,
        'else': TokenType.ELSE,
        'return': TokenType.RETURN,
        'num': TokenType.NUMBER,
        'id': TokenType.IDENTIFIER,
    }

    # Dictionary mapping symbol strings to their corresponding token types
    symbols: dict[str, TokenType] = {
        '==': TokenType.COMPARE,
        '=': TokenType.ASSIGN,
        '+': TokenType.PLUS,
        '-': TokenType.MINUS,
        '*': TokenType.MULTIPLY,
        ';': TokenType.SEMICOLON,
        ',': TokenType.COMMA,
        '(': TokenType.LEFT_PAREN,
        ')': TokenType.RIGHT_PAREN,
        '{': TokenType.LEFT_BRACE,
        '}': TokenType.RIGHT_BRACE,
    }

    def __init__(self, buffer: str) -> None:
        """
        Initialize a new Lexer instance.

        Args:
            buffer: The source code string to be tokenized
        """
        self.buffer: str = buffer  # The input source code
        self.position: int = 0     # Current position in the buffer
        self.tokens: list[Token] = []  # List of tokens found during scanning
        self.scan_state: ScanState = ScanState.UNINITIALIZED  # Current state of the scanner

    def __str__(self) -> str:
        """Returns a string representation of all tokens found during scanning."""
        return '\n'.join(str(token) for token in self.tokens)

    def __repr__(self) -> str:
        """Returns the same string representation as __str__ for consistency."""
        return self.__str__()

    def scan(self) -> list[Token]:
        """
        Performs lexical analysis on the input buffer, converting it into a sequence of tokens.

        The scanning algorithm works as follows:
        1. Skip whitespace characters
        2. Try to match keywords from the language
        3. If no keyword matches, try to match symbols/operators
        4. If neither matches, report an unexpected token error

        Returns:
            A list of Token objects representing the tokenized input
        """
        buffer: str = self.buffer
        matched: bool = False

        def add_token(token_type: TokenType, length: int) -> None:
            """
            Helper function to add a new token to the token list and update the buffer position.

            Args:
                token_type: The type of token to add
                length: The length of the token in characters
            """
            nonlocal buffer, matched
            token_value = buffer[:length]
            self.tokens.append(Token(token_type, self.position, length, token_value))
            buffer = buffer[length:]
            self.position += length
            matched = True

        while buffer:
            # Skip whitespace characters
            if buffer[0].isspace():
                buffer = buffer[1:]
                self.position += 1
                continue

            matched = False

            # Try to match keywords
            for keyword, token_type in self.keywords.items():
                if buffer.startswith(keyword):
                    add_token(token_type, len(keyword))
                    break

            if matched:
                continue

            # Try to match symbols/operators
            for symbol, token_type in self.symbols.items():
                if buffer.startswith(symbol):
                    add_token(token_type, len(symbol))
                    break

            # If no match found, report an unexpected token
            if not matched:
                add_token(TokenType.UNEXPECTED_TOKEN, 1)
                self.scan_state = ScanState.FAILURE
                break

        # If scanning completed without errors, mark as successful
        if self.scan_state == ScanState.UNINITIALIZED:
            self.scan_state = ScanState.SUCCESS
        return self.tokens

class AstNode:
    """
    Base class for all Abstract Syntax Tree (AST) nodes.

    The AST represents the hierarchical structure of the program after parsing.
    Each node in the tree corresponds to a syntactic construct in the source code,
    such as expressions, statements, declarations, etc.

    This base class provides common functionality for all AST nodes, particularly
    the ability to pretty-print the tree structure for visualization and debugging.
    """
    def pretty(self, indent: str = "", is_last: bool = True) -> str:
        lines = []
        prefix = indent + ("└── " if is_last else "├── ")
        lines.append(prefix + self.__class__.__name__)

        children = [v for v in self.__dict__.values() if isinstance(v, AstNode) or isinstance(v, list)]
        for i, child in enumerate(children):
            is_child_last = i == len(children) - 1
            new_indent = indent + ("    " if is_last else "│   ")

            if isinstance(child, AstNode):
                lines.append(child.pretty(new_indent, is_child_last))
            elif isinstance(child, list):
                for j, elem in enumerate(child):
                    is_elem_last = j == len(child) - 1
                    if isinstance(elem, AstNode):
                        lines.append(elem.pretty(new_indent, is_elem_last))
                    else:
                        val = f"{elem.type.name}('{elem.value}')" if hasattr(elem, "type") else str(elem)
                        lines.append(new_indent + ("└── " if is_elem_last else "├── ") + val)

        return "\n".join(lines)

@dataclass
class Identifier(AstNode):
    """
    AST node representing an identifier (variable or function name).

    Identifiers are names used to refer to variables, functions, and other
    named entities in the program.
    """
    token: Token  # The token containing the identifier's name and position

    def __str__(self):
        """Returns a string representation of the identifier."""
        return f"Identifier({self.token.value})"

@dataclass
class Literal(AstNode):
    """
    AST node representing a literal value (e.g., a number).

    Literals are constant values that appear directly in the source code,
    such as numeric constants.
    """
    token: Token  # The token containing the literal's value and position

    def __str__(self):
        """Returns a string representation of the literal."""
        return f"Literal({self.token.value})"

@dataclass
class Type(AstNode):
    """
    AST node representing a type specifier.

    Type specifiers are used in variable and function declarations to
    indicate the data type of the declared entity.
    """
    token: Token  # The token containing the type name and position

    def __str__(self):
        """Returns a string representation of the type."""
        return f"Type({self.token.value})"

@dataclass
class VarDecl(AstNode):
    """
    AST node representing a variable declaration.

    A variable declaration specifies a new variable with a type, name, and
    optional initializer expression. For example: "type x = 5;"
    """
    type_: Type              # The type of the variable
    name: Token              # The token containing the variable name
    init_expr: Optional[AstNode]  # Optional initializer expression (can be None)

    def __str__(self):
        """Returns a string representation of the variable declaration."""
        return f"VarDecl(type={self.type_}, name={self.name.value}, init={self.init_expr})"

@dataclass
class StmtList(AstNode):
    """
    AST node representing a list of statements.

    A statement list contains zero or more statements that are executed
    sequentially. Statement lists appear in blocks and function bodies.
    """
    statements: list[AstNode]  # List of statement nodes

    def __str__(self):
        """Returns a string representation of the statement list."""
        return f"StmtList({self.statements})"

@dataclass
class Block(AstNode):
    """
    AST node representing a block of statements enclosed in braces.

    A block groups multiple statements together and creates a new scope.
    Blocks are used in function bodies, if/else statements, loops, etc.
    """
    statements: StmtList  # The list of statements in the block

    def __str__(self):
        """Returns a string representation of the block."""
        return f"Block({self.statements})"

@dataclass
class Param(AstNode):
    """
    AST node representing a function parameter.

    A parameter is part of a function declaration and specifies the type and
    name of an argument that can be passed to the function.
    """
    type_: Type   # The type of the parameter
    name: Token   # The token containing the parameter name

    def __str__(self):
        """Returns a string representation of the parameter."""
        return f"Param(type={self.type_}, name={self.name.value})"

@dataclass
class ParamList(AstNode):
    """
    AST node representing a list of function parameters.

    A parameter list contains zero or more parameters that define the
    arguments a function can accept.
    """
    params: list[Param]  # List of parameter nodes

    def __str__(self):
        """Returns a string representation of the parameter list."""
        return f"ParamList({self.params})"

@dataclass
class FuncDecl(AstNode):
    """
    AST node representing a function declaration.

    A function declaration defines a new function with a return type, name,
    parameter list, and body. For example: "type func(type param) { ... }"
    """
    return_type: Type    # The return type of the function
    name: Token          # The token containing the function name
    params: ParamList    # The list of parameters
    body: Block          # The function body (block of statements)

    def __str__(self):
        """Returns a string representation of the function declaration."""
        return f"FuncDecl(return_type={self.return_type}, name={self.name.value}, params={self.params}, body={self.body})"

@dataclass
class Decl(AstNode):
    """
    AST node representing a declaration (either variable or function).

    This is a wrapper node that can contain either a variable declaration
    or a function declaration.
    """
    decl: VarDecl | FuncDecl  # The actual declaration (either variable or function)

    def __str__(self):
        """Returns a string representation of the declaration."""
        return f"Decl({self.decl})"

@dataclass
class DeclList(AstNode):
    """
    AST node representing a list of declarations.

    A declaration list contains zero or more declarations (variables or functions)
    that appear at the top level of a program.
    """
    decls: list[Decl]  # List of declaration nodes

    def __str__(self):
        """Returns a string representation of the declaration list."""
        return f"DeclList({self.decls})"

@dataclass
class Program(AstNode):
    """
    AST node representing the entire program.

    This is the root node of the AST and contains the list of all top-level
    declarations in the program.
    """
    decl_list: Optional[DeclList]  # The list of all declarations in the program

    def __str__(self):
        """Returns a string representation of the program."""
        return f"Program({self.decl_list})"

@dataclass
class IfStmt(AstNode):
    """
    AST node representing an if statement.

    An if statement conditionally executes code based on a condition.
    It can optionally include an else branch that executes when the
    condition is false.
    """
    condition: AstNode              # The condition expression
    then_branch: AstNode            # The statement to execute if condition is true
    else_branch: Optional[AstNode] = None  # Optional statement to execute if condition is false

    def __str__(self):
        """Returns a string representation of the if statement."""
        return f"IfStmt(cond={self.condition}, then={self.then_branch}, else={self.else_branch})"

@dataclass
class WhileStmt(AstNode):
    """
    AST node representing a while loop.

    A while loop repeatedly executes its body as long as the condition
    expression evaluates to true.
    """
    condition: AstNode  # The loop condition expression
    body: AstNode       # The loop body (statement to execute repeatedly)

    def __str__(self):
        """Returns a string representation of the while statement."""
        return f"WhileStmt(cond={self.condition}, body={self.body})"

@dataclass
class ForStmt(AstNode):
    """
    AST node representing a for loop.

    A for loop has three components: initialization, condition, and update.
    It executes the initialization once, then repeatedly checks the condition,
    executes the body, and executes the update until the condition is false.
    """
    init: Optional[AstNode]      # Optional initialization expression
    condition: Optional[AstNode]  # Optional loop condition expression
    update: Optional[AstNode]     # Optional update expression
    body: AstNode                # The loop body (statement to execute repeatedly)

    def __str__(self):
        """Returns a string representation of the for statement."""
        return f"ForStmt(init={self.init}, cond={self.condition}, update={self.update}, body={self.body})"

@dataclass
class ReturnStmt(AstNode):
    """
    AST node representing a return statement.

    A return statement specifies the value to be returned from a function
    and terminates the function's execution.
    """
    expr: AstNode  # The expression whose value will be returned

    def __str__(self):
        """Returns a string representation of the return statement."""
        return f"ReturnStmt({self.expr})"

@dataclass
class ExprStmt(AstNode):
    """
    AST node representing an expression statement.

    An expression statement is an expression used as a statement, typically
    for its side effects (like an assignment).
    """
    expr: AstNode  # The expression to evaluate

    def __str__(self):
        """Returns a string representation of the expression statement."""
        return f"ExprStmt({self.expr})"

@dataclass
class BinaryOp(AstNode):
    """
    AST node representing a binary operation.

    A binary operation applies an operator to two operands, such as addition,
    multiplication, comparison, or assignment. The result is a new value.
    """
    left: AstNode   # The left operand expression
    op: Token       # The operator token (e.g., +, *, ==, =)
    right: AstNode  # The right operand expression

    def __str__(self):
        """Returns a string representation of the binary operation."""
        return f"BinaryOp({self.left} {self.op.value} {self.right})"

@dataclass
class UnaryOp(AstNode):
    """
    AST node representing a unary operation.

    A unary operation applies an operator to a single operand, such as
    negation (-). The result is a new value.
    """
    op: Token       # The operator token (e.g., -)
    operand: AstNode  # The operand expression

    def __str__(self):
        """Returns a string representation of the unary operation."""
        return f"UnaryOp({self.op.value}{self.operand})"

def closure(items: set[Item], grammar: list[Production]) -> set[Item]:
    """
    Computes the closure of a set of LR(0) items.

    The closure operation expands a set of items by adding all items that could be
    derived from the symbols appearing after the dot in the original items. This is
    a key operation in constructing LR parsing tables.

    For example, if we have an item A -> α•Bβ, and B -> γ is a production,
    then B -> •γ would be added to the closure.

    Args:
        items: The initial set of LR(0) items
        grammar: The grammar productions

    Returns:
        The closure set containing all derivable items
    """
    closure_set = set(items)  # Initialize with the input items
    changed = True
    # Continue until no more items can be added
    while changed:
        changed = False
        new_items = set()
        for item in closure_set:
            # If the dot is not at the end of the production
            if item.dot < len(item.production.rhs):
                # Get the symbol after the dot
                symbol = item.production.rhs[item.dot]
                # If it's a non-terminal (string), add its productions to the closure
                if isinstance(symbol, str):
                    for prod in grammar:
                        if prod.lhs == symbol:
                            # Create a new item with the dot at the beginning
                            new_item = Item(prod, 0)
                            if new_item not in closure_set:
                                new_items.add(new_item)
        if new_items:
            # Add the new items to the closure set
            closure_set.update(new_items)
            changed = True
    return closure_set


def find_state_index(I: set[Item], C: list[set[Item]]) -> int:
    """
    Finds the index of a state in the collection of LR(0) states.

    This function is used during the construction of parsing tables to identify
    which state corresponds to a particular set of items.

    Args:
        I: The set of items representing a state
        C: The collection of all states (sets of items)

    Returns:
        The index of the state in the collection, or -1 if not found
    """
    for idx, state in enumerate(C):
        if state == I:
            return idx
    return -1


def all_symbols(grammar: list[Production]) -> set[Symbol]:
    """
    Extracts all symbols (terminals and non-terminals) from a grammar.

    This function collects all symbols that appear on the right-hand side of
    any production in the grammar. These symbols are used in the construction
    of the LR parsing tables.

    Args:
        grammar: The list of grammar productions

    Returns:
        A set containing all symbols in the grammar
    """
    symboles = set()
    for p in grammar:
        symboles.update(p.rhs)
    return symboles


def goto(I: set[Item], X: Symbol, grammar: list[Production]) -> set[Item]:
    """
    Computes the goto function for LR parsing table construction.

    The goto function determines the next state in the LR automaton when
    a particular symbol is encountered. It works by advancing the dot past
    the specified symbol in all applicable items, then computing the closure
    of the resulting set.

    For example, if state I contains an item A -> α•Xβ, then goto(I,X) would
    include the item A -> αX•β (and its closure).

    Args:
        I: The current set of items (state)
        X: The symbol to transition on
        grammar: The grammar productions

    Returns:
        The new set of items after transitioning on symbol X
    """
    moved = set()
    # Find all items where the dot is before symbol X
    for item in I:
        if item.dot < len(item.production.rhs) and item.production.rhs[item.dot] == X:
            # Move the dot past X
            moved.add(Item(item.production, item.dot + 1))
    # Compute the closure of the new set
    return closure(moved, grammar)

def compute_first_follow(grammar: list[Production]) -> tuple[dict[str, set[TokenType]], dict[str, set[TokenType]]]:
    """
    Computes the FIRST and FOLLOW sets for all non-terminals in the grammar.

    These sets are essential for constructing predictive parsers and for resolving
    conflicts in LR parsing tables:

    - FIRST(X): The set of terminals that can appear as the first symbol of any
      string derived from X.
    - FOLLOW(X): The set of terminals that can appear immediately after X in some
      sentential form.

    The algorithm also computes which non-terminals are nullable (can derive the
    empty string).

    Args:
        grammar: The list of grammar productions

    Returns:
        A tuple containing:
        - first: Dictionary mapping non-terminals to their FIRST sets
        - follow: Dictionary mapping non-terminals to their FOLLOW sets
    """
    # Initialize FIRST and FOLLOW sets for all non-terminals
    first: dict[str, set[TokenType]] = {p.lhs: set() for p in grammar}
    follow: dict[str, set[TokenType]] = {p.lhs: set() for p in grammar}
    nullable: dict[str, bool] = {p.lhs: False for p in grammar}

    # Add EOF to FOLLOW set of the start symbol
    start = grammar[0].lhs
    follow[start].add(TokenType.EOF)

    # Compute nullable non-terminals
    changed = True
    while changed:
        changed = False
        for p in grammar:
            # Empty production means the non-terminal is nullable
            if not p.rhs:
                if not nullable[p.lhs]:
                    nullable[p.lhs] = True
                    changed = True
            # If all symbols in the production are nullable, the LHS is nullable
            elif all((isinstance(sym, str) and nullable.get(sym, False)) or isinstance(sym, TokenType) for sym in p.rhs):
                if not nullable[p.lhs]:
                    nullable[p.lhs] = True
                    changed = True

    # Compute FIRST sets
    changed = True
    while changed:
        changed = False
        for p in grammar:
            lhs = p.lhs
            rhs = p.rhs
            trailer = set()
            for sym in rhs:
                # If we find a terminal, add it to FIRST and stop
                if isinstance(sym, TokenType):
                    if sym not in first[lhs]:
                        first[lhs].add(sym)
                        changed = True
                    break
                else:
                    # Add FIRST(sym) to FIRST(lhs)
                    before = len(first[lhs])
                    first[lhs].update(first[sym])
                    if before != len(first[lhs]):
                        changed = True
                    # If sym is not nullable, stop here
                    if not nullable[sym]:
                        break
            else:
                # All symbols in the production are nullable
                pass

    # Compute FOLLOW sets
    changed = True
    while changed:
        changed = False
        for p in grammar:
            # Start with FOLLOW(lhs)
            trailer = follow[p.lhs].copy()
            # Process the RHS from right to left
            for sym in reversed(p.rhs):
                if isinstance(sym, TokenType):
                    # Terminal becomes the new trailer
                    trailer = {sym}
                else:
                    # Add trailer to FOLLOW(sym)
                    before = len(follow[sym])
                    follow[sym].update(trailer)
                    if len(follow[sym]) != before:
                        changed = True
                    # Update trailer based on whether sym is nullable
                    if nullable.get(sym, False):
                        trailer = trailer.union(first[sym])
                    else:
                        trailer = first[sym]

    return first, follow


def build_lr0_states(grammar: list[Production]) -> list[set[Item]]:
    """
    Constructs the collection of LR(0) states (sets of items) for the grammar.

    This is a key step in building an LR parser. The states represent the possible
    configurations of the parser as it processes input. Each state is a set of items
    that share a common prefix of recognized symbols.

    The algorithm works as follows:
    1. Start with the initial state containing the closure of the starting item
    2. For each state and each grammar symbol, compute the goto function
    3. If the goto function produces a new state, add it to the collection
    4. Repeat until no new states can be added

    Args:
        grammar: The list of grammar productions

    Returns:
        A list of all LR(0) states for the grammar
    """
    # Start with the initial production
    start_prod = grammar[0]
    initial_item = Item(start_prod, 0)
    states = []

    # Create the initial state (closure of the starting item)
    state0 = closure({initial_item}, grammar)
    states.append(state0)

    # Iteratively build the collection of states
    while True:
        new_state_added = False
        # For each existing state
        for I in states:
            # For each grammar symbol
            for X in all_symbols(grammar):
                # Compute the goto function
                J = goto(I, X, grammar)
                # If it produces a non-empty state that's not already in the collection
                if J and J not in states:
                    # Add the new state
                    states.append(J)
                    new_state_added = True
        # If no new states were added in this iteration, we're done
        if not new_state_added:
            break
    return states


def build_action_goto(states: list[set[Item]], grammar: list[Production], follow: dict[str, set[TokenType]]):
    """
    Builds the ACTION and GOTO tables for an LR parser.

    The ACTION table determines what the parser should do (shift, reduce, accept)
    based on the current state and lookahead token. The GOTO table determines
    which state to transition to after a reduction.

    The tables are constructed as follows:
    - For items A -> α•aβ (where a is a terminal), add shift actions
    - For items A -> α• (where the dot is at the end), add reduce actions
    - For the item S' -> S• (where S is the start symbol), add the accept action
    - For items A -> α•Bβ (where B is a non-terminal), add goto transitions

    Args:
        states: The collection of LR(0) states
        grammar: The list of grammar productions
        follow: The FOLLOW sets for all non-terminals

    Returns:
        A tuple containing:
        - action: Dictionary mapping (state, token) to actions ("shift N", "reduce A->α", "accept")
        - goto_table: Dictionary mapping (state, non-terminal) to next state
    """
    action = {}      # Maps (state, terminal) to actions
    goto_table = {}  # Maps (state, non-terminal) to next state

    # Process each state
    for i, I in enumerate(states):
        for item in I:
            A = item.production.lhs

            # If the dot is not at the end of the production
            if item.dot < len(item.production.rhs):
                # Get the symbol after the dot
                sym = item.production.rhs[item.dot]
                # Find the next state using the goto function
                j = find_state_index(goto(I, sym, grammar), states)
                if j == -1:
                    continue

                # If the symbol is a terminal, add a shift action
                if isinstance(sym, TokenType):
                    action[(i, sym)] = f"shift {j}"
                # If the symbol is a non-terminal, add a goto transition
                else:
                    goto_table[(i, sym)] = j
            else:
                # If the dot is at the end of the production
                if A != "S'":  # Not the augmented start symbol
                    # Create a string representation of the production for the reduce action
                    key = f"{A}->{' '.join(str(x.value if isinstance(x, TokenType) else x) for x in item.production.rhs)}"
                    # Add reduce actions for all terminals in FOLLOW(A)
                    for a in follow[A]:
                        action[(i, a)] = f"reduce {key}"
                else:  # Augmented start symbol
                    # Add the accept action
                    action[(i, TokenType.EOF)] = "accept"

    print_action_goto(action, goto_table)
    return action, goto_table



class Error:
    """
    Utility class for generating user-friendly error messages.

    This class provides methods to format error messages with source code context,
    making it easier for users to understand and locate errors in their code.
    The error messages include:
    - The error description
    - The file location (line and column)
    - A snippet of the source code with the error highlighted
    """

    @staticmethod
    def build_lexer_error_msg(lexer: Lexer) -> str:
        """
        Builds an error message for lexical errors (unexpected tokens).

        This method finds the first unexpected token in the lexer's token list
        and generates a formatted error message showing the location and context
        of the error.

        Args:
            lexer: The lexer containing the tokens and source buffer

        Returns:
            A formatted error message string
        """
        for token in lexer.tokens:
            if token.type != TokenType.UNEXPECTED_TOKEN:
                continue

            # Calculate line and column numbers
            lines = lexer.buffer[:token.position].splitlines()
            line = len(lines) if lines else 1
            col = len(lines[-1]) + 1 if lines else 1

            # Extract the line containing the error
            line_start: int = lexer.buffer.rfind('\n', 0, token.position) + 1
            line_end: int = lexer.buffer.find('\n', token.position)

            if line_end == -1:
                line_end = len(lexer.buffer)

            # Create the error message with source context
            snippet: str = lexer.buffer[line_start:line_end]
            caret_line = ' ' * (col - 1) + '^'
            msg: str = (
                f"error: unexpected token '{token.value}'\n"
                f" --> input:{line}:{col}\n"
                f"  |\n"
                f"{line} | {snippet}\n"
                f"  | {caret_line}"
            )

            return msg

        return "error: unhandled error occurred"

    @staticmethod
    def build_parser_error_msg(error: SyntaxError, buffer: str) -> str:
        """
        Builds an error message for syntax errors during parsing.

        This method extracts token information from the syntax error message
        and generates a formatted error message showing the location and context
        of the error.

        Args:
            error: The SyntaxError exception
            buffer: The source code buffer

        Returns:
            A formatted error message string
        """
        import re
        # Extract token information from the error message
        match = re.search(r"Token\(type=TokenType\.(\w+), position=(\d+), length=(\d+), value='([^']*)'\)", str(error))
        if not match:
            return f"error: {error}"

        token_type, pos, length, value = match.groups()
        pos = int(pos)
        length = int(length)

        # Calculate line and column numbers
        lines = buffer[:pos].splitlines()
        line = len(lines) if lines else 1
        col = len(lines[-1]) + 1 if lines else 1

        # Extract the line containing the error
        line_start = buffer.rfind('\n', 0, pos) + 1
        line_end = buffer.find('\n', pos)
        if line_end == -1:
            line_end = len(buffer)

        # Create the error message with source context
        snippet = buffer[line_start:line_end]
        caret_line = ' ' * (col - 1) + '^'
        msg = (
            f"error: unexpected token '{value}'\n"
            f" --> input:{line}:{col}\n"
            f"  |\n"
            f"{line} | {snippet}\n"
            f"  | {caret_line}"
        )

        return msg

    @staticmethod
    def build_ast_error_msg(token: Token, buffer: str, context: str = "") -> str:
        """
        Builds an error message for AST construction errors.

        This method generates a formatted error message for errors that occur
        during the construction of the abstract syntax tree, showing the location
        and context of the error.

        Args:
            token: The token where the error occurred
            buffer: The source code buffer
            context: Additional context information about the error

        Returns:
            A formatted error message string
        """
        # Calculate line and column numbers
        line = buffer[:token.position].count('\n') + 1
        line_start = buffer.rfind('\n', 0, token.position) + 1
        line_end = buffer.find('\n', token.position)
        if line_end == -1:
            line_end = len(buffer)

        # Create the error message with source context
        col = token.position - line_start + 1
        snippet = buffer[line_start:line_end]
        caret_line = ' ' * (col - 1) + '^'
        return (
            f"error: AST error at token '{token.value}' {context}\n"
            f" --> input:{line}:{col}\n"
            f"  |\n"
            f"{line} | {snippet}\n"
            f"  | {caret_line}"
        )


def _ensure_index(tok):
    """
    Ensures that a token has an 'index' attribute.

    This is a helper function used during AST construction to make sure
    all tokens have an index attribute, which is used for error reporting
    and source location tracking.

    Args:
        tok: The token to check and potentially modify
    """
    if not hasattr(tok, "index"):
        if hasattr(tok, "position"):
            setattr(tok, "index", tok.position)
        else:
            setattr(tok, "index", -1)


class Parser:
    """
    LR parser that converts a sequence of tokens into an abstract syntax tree (AST).

    This parser implements a bottom-up LR parsing algorithm, which is powerful enough
    to handle most programming language grammars. It uses precomputed ACTION and GOTO
    tables to determine the parsing steps.

    The parsing process works as follows:
    1. Initialize a stack with the start state
    2. For each input token:
       a. Look up the action for the current state and token
       b. If the action is "shift", push the token and next state onto the stack
       c. If the action is "reduce", pop items from the stack and construct an AST node
       d. If the action is "accept", return the completed AST
    """

    @dataclass
    class _Frame:
        """
        Internal class representing a frame on the parser stack.

        Each frame contains:
        - state: The current parser state number
        - node: The associated AST node or token
        """
        state: int
        node: any

    def __init__(self, tokens: list[Token]) -> None:
        """
        Initialize a new Parser instance.

        Args:
            tokens: The list of tokens to parse
        """
        # Store the input tokens
        self.tokens: list[Token] = tokens

        # Ensure the token list ends with EOF
        if not self.tokens or self.tokens[-1].type is not TokenType.EOF:
            self.tokens.append(Token(TokenType.EOF,
                                     self.tokens[-1].position + self.tokens[-1].length if self.tokens else 0,
                                     0, ""))

        # Augment the grammar with a new start production S' -> Program
        augmented = [Production("S'", ("Program",))] + PRODUCTIONS

        # Compute the FIRST and FOLLOW sets
        first, follow = compute_first_follow(augmented)

        # Build the LR(0) states
        states = build_lr0_states(augmented)

        # Construct the ACTION and GOTO tables
        self._action, self._goto = build_action_goto(states, augmented, follow)

        # Create a map from production string representations to Production objects
        self._prod_map: dict[str, Production] = {
            f"{p.lhs}->{' '.join(str(x.value if isinstance(x, TokenType) else x) for x in p.rhs)}": p
            for p in augmented
        }

    def parse(self, debug: bool = False) -> AstNode:
        """
        Parses the token stream and constructs an abstract syntax tree.

        This method implements the LR parsing algorithm, which uses a stack to
        keep track of states and symbols. It processes tokens one by one, performing
        shift and reduce actions according to the precomputed parsing tables.

        Args:
            debug: Whether to print debug information during parsing

        Returns:
            The root node of the constructed AST

        Raises:
            SyntaxError: If the input contains a syntax error
            RuntimeError: If there's an internal error in the parser
        """
        # Initialize the parser stack with the start state
        stack: list[Parser._Frame] = [self._Frame(0, None)]
        i: int = 0  # Current token index

        while True:
            # Get the current state from the top of the stack
            state = stack[-1].state
            # Get the current lookahead token
            lookahead: Token = self.tokens[i]
            # Look up the action for this state and token
            act: Optional[str] = self._action.get((state, lookahead.type))

            # If no action is defined, there's a syntax error
            if act is None:
                raise SyntaxError(f"Unexpected token {lookahead} in state {state}")

            # Handle shift actions
            if act.startswith("shift"):
                # Extract the target state number
                tgt_state = int(act.split()[1])

                # Create an appropriate AST node for the token
                if lookahead.type is TokenType.IDENTIFIER:
                    _ensure_index(lookahead)
                    node: any = Identifier(lookahead)
                elif lookahead.type is TokenType.NUMBER:
                    _ensure_index(lookahead)
                    node = Literal(lookahead)
                else:
                    node = lookahead

                # Push the new state and node onto the stack
                stack.append(self._Frame(tgt_state, node))
                # Move to the next token
                i += 1

                # Print debug information if requested
                if debug:
                    self._dbg("SHIFT", lookahead, stack)

            # Handle reduce actions
            elif act.startswith("reduce"):
                # Extract the production to reduce by
                prod_key = act[len("reduce "):]
                prod = self._prod_map[prod_key]
                k = len(prod.rhs)

                # Pop k items from the stack (k = length of the production's RHS)
                children: list[any] = []
                for _ in range(k):
                    children.append(stack.pop().node)
                children.reverse()  # Reverse to get the correct order

                # Construct an AST node for the reduction
                lhs_node = self._make_node(prod, children)
                # Look up the goto state for the current state and the LHS non-terminal
                goto_state = self._goto[(stack[-1].state, prod.lhs)]
                # Push the new state and node onto the stack
                stack.append(self._Frame(goto_state, lhs_node))

                # Print debug information if requested
                if debug:
                    self._dbg(f"REDUCE {prod.lhs} -> {' '.join(map(str, prod.rhs))}", lhs_node, stack)

            # Handle accept action
            elif act == "accept":
                # Print debug information if requested
                if debug:
                    self._dbg("ACCEPT", stack[-1].node, stack)
                # Return the completed AST
                return stack[-1].node

            # Handle unknown actions (should never happen)
            else:
                raise RuntimeError(f"Unknown parser action {act}")

    def _dbg(self, op: str, obj: any, stack: list[_Frame]) -> None:
        """
        Prints debug information during parsing.

        This method is called when debug mode is enabled to show the parser's
        operations (shift, reduce, accept) and the current state of the stack.

        Args:
            op: The operation being performed (e.g., "SHIFT", "REDUCE", "ACCEPT")
            obj: The object being processed (token or AST node)
            stack: The current parser stack
        """
        def _name(x: any) -> str:
            """Helper function to get a readable name for an object."""
            if isinstance(x, AstNode):
                return x.__class__.__name__
            if isinstance(x, Token):
                return x.type.name
            return str(x)

        # Format the stack as a list of state numbers
        st = "[" + ", ".join(str(f.state) for f in stack) + "]"
        # Get a readable name for the object
        sym = _name(obj)
        # Print the debug information
        print(f"{op:<10} {sym:<20}  stack={st}")

    def _make_node(self, p: Production, c: list[any]) -> any:
        """
        Constructs an AST node based on a production rule and its children.

        This method is called during the reduce step of parsing to create the
        appropriate AST node for a production. It handles all the different
        production rules in the grammar and constructs the corresponding AST nodes.

        Args:
            p: The production rule being reduced
            c: The list of children (tokens or AST nodes) from the right-hand side

        Returns:
            The constructed AST node

        Raises:
            SyntaxError: If there's an error in the AST construction
        """
        # Program node
        if p.lhs == "Program":
            return Program(c[0] if c else None)

        # Declaration list
        if p.lhs == "DeclList":
            if not c:
                return DeclList([])  # Empty declaration list
            decl, lst = c
            if isinstance(lst, DeclList):
                return DeclList([decl] + lst.decls)  # Add declaration to existing list
            return DeclList([decl])  # Create new list with single declaration

        # Declaration
        if p.lhs == "Decl":
            return Decl(c[0])

        # Variable declaration
        if p.lhs == "VarDecl":
            type_tok = c[0]
            _ensure_index(type_tok)
            type_node = Type(type_tok)
            name_tok = c[1]
            if len(p.rhs) == 3:
                return VarDecl(type_node, name_tok, None)  # Declaration without initialization
            expr = c[3]
            return VarDecl(type_node, name_tok, expr)  # Declaration with initialization

        # Function declaration
        if p.lhs == "FuncDecl":
            type_tok, name_tok, _lp, params, _rp, block = c
            _ensure_index(type_tok)
            type_node = Type(type_tok)
            return FuncDecl(type_node, name_tok, params, block)

        # Parameter list
        if p.lhs == "ParamList":
            if not c:
                return ParamList([])  # Empty parameter list
            param, tail = c
            return ParamList([param] + (tail if isinstance(tail, list) else []))

        # Parameter tail (for handling multiple parameters)
        if p.lhs == "ParamTail":
            if not c:
                return []  # End of parameter list
            _comma, param, tail = c
            return [param] + tail  # Add parameter to the list

        # Parameter
        if p.lhs == "Param":
            type_tok, name_tok = c
            _ensure_index(type_tok)
            return Param(Type(type_tok), name_tok)

        # Block of statements
        if p.lhs == "Block":
            _lb, stmt_list, _rb = c
            return Block(stmt_list)

        # Statement list
        if p.lhs == "StmtList":
            if not c:
                return StmtList([])  # Empty statement list
            stmt, tail = c
            if isinstance(stmt, Token):
                raise SyntaxError(Error.build_ast_error_msg(stmt, self.buffer, context="in StmtList"))
            return StmtList([stmt] + tail.statements)  # Add statement to the list

        # Return statement
        if p.lhs == "MatchedStmt" and p.rhs == (TokenType.RETURN, 'Expr', TokenType.SEMICOLON):
            _ret, expr, _semi = c
            return ReturnStmt(expr)

        # If-else statement (matched)
        if p.lhs == "MatchedStmt" and p.rhs == (TokenType.IF, TokenType.LEFT_PAREN, 'Expr', TokenType.RIGHT_PAREN, 'MatchedStmt', TokenType.ELSE, 'MatchedStmt'):
            _if, _lp, cond, _rp, then_m, _else, else_m = c
            return IfStmt(cond, then_m, else_m)

        # If statement without else (unmatched)
        if p.lhs == "UnmatchedStmt" and p.rhs == (TokenType.IF, TokenType.LEFT_PAREN, 'Expr', TokenType.RIGHT_PAREN, 'Stmt'):
            _if, _lp, cond, _rp, stmt = c
            return IfStmt(cond, stmt, None)

        # If-else statement with unmatched else branch
        if p.lhs == "UnmatchedStmt" and p.rhs == (TokenType.IF, TokenType.LEFT_PAREN, 'Expr', TokenType.RIGHT_PAREN, 'MatchedStmt', TokenType.ELSE, 'UnmatchedStmt'):
            _if, _lp, cond, _rp, then_m, _else, else_m = c
            return IfStmt(cond, then_m, else_m)

        # While loop
        if p.lhs in ("MatchedStmt", "UnmatchedStmt") and p.rhs[0] == TokenType.WHILE:
            _wh, _lp, cond, _rp, body = c
            return WhileStmt(cond, body)

        # For loop
        if p.lhs in ("MatchedStmt", "UnmatchedStmt") and p.rhs[0] == TokenType.FOR:
            _for, _lp, init, _sc1, cond, _sc2, update, _rp, body = c
            return ForStmt(init, cond, update, body)

        # Generic statement
        if p.lhs in ("Stmt", "MatchedStmt", "UnmatchedStmt"):
            if isinstance(c[0], Token):
                raise SyntaxError(Error.build_ast_error_msg(c[0], self.buffer, context=f"in {p.lhs}"))
            return c[0]

        # Expression statement (assignment)
        if p.lhs == "ExprStmt":
            id_tok, assign_tok, expr, _semi = c
            _ensure_index(id_tok)
            binop = BinaryOp(Identifier(id_tok), assign_tok, expr)
            return ExprStmt(binop)

        # Simple production with one symbol on the right-hand side
        if len(p.rhs) == 1 and p.rhs[0] not in (TokenType.MINUS,):
            if isinstance(c[0], Token):
                raise SyntaxError(Error.build_ast_error_msg(c[0], self.buffer, context=f"as {p.lhs}"))
            return c[0]

        # Binary operation
        if len(p.rhs) == 3 and isinstance(p.rhs[1], TokenType) and p.rhs[1] in {TokenType.PLUS, TokenType.MULTIPLY, TokenType.COMPARE, TokenType.ASSIGN}:
            left, op_tok, right = c
            return BinaryOp(left, op_tok, right)

        # Unary operation
        if p.lhs == "UnaryExpr" and len(c) == 2:
            op_tok, expr = c
            return UnaryOp(op_tok, expr)

        # Parenthesized expression
        if p.lhs == "PrimaryExpr" and p.rhs == (TokenType.LEFT_PAREN, 'Expr', TokenType.RIGHT_PAREN):
            _lp, expr, _rp = c
            return expr

        # Primary expression
        if p.lhs == "PrimaryExpr" and len(c) == 1:
            return c[0]

        # Default case: return the first child or None
        return c[0] if c else None

def parse(buffer: str) -> str:
    """
    Main parsing function that processes source code and returns the AST or error message.

    This function orchestrates the entire parsing process:
    1. Tokenize the input using the lexer
    2. Check for lexical errors
    3. Parse the tokens to build an AST
    4. Format the result (either the AST or an error message)

    Args:
        buffer: The source code string to parse

    Returns:
        A string containing either:
        - A pretty-printed representation of the AST (if parsing succeeds)
        - A formatted error message (if lexical or syntax errors are found)
    """
    # Get the grammar productions
    grammar = PRODUCTIONS

    # Tokenize the input
    lexer = Lexer(buffer)
    tokens = lexer.scan()

    # Check for lexical errors
    if lexer.scan_state == ScanState.FAILURE:
        return Error.build_lexer_error_msg(lexer)

    try:
        # Parse the tokens to build an AST
        parser = Parser(tokens)
        ast = parser.parse()
        # Return a pretty-printed representation of the AST
        return ast.pretty()
    except SyntaxError as e:
        # If a syntax error occurs, return a formatted error message
        return Error.build_parser_error_msg(e, lexer.buffer)

SYMBOL_MAP = {
    TokenType.IDENTIFIER: 'id',
    TokenType.NUMBER: 'num',
    TokenType.SEMICOLON: ';',
    TokenType.COMMA: ',',
    TokenType.LEFT_PAREN: '(',
    TokenType.RIGHT_PAREN: ')',
    TokenType.LEFT_BRACE: '{',
    TokenType.RIGHT_BRACE: '}',
    TokenType.PLUS: '+',
    TokenType.MINUS: '-',
    TokenType.MULTIPLY: '*',
    TokenType.COMPARE: '==',
    TokenType.ASSIGN: '=',
    TokenType.EOF: '$',
}

def _tok(sym):
    return SYMBOL_MAP.get(sym, sym.name if hasattr(sym, "name") else str(sym))

def print_action_table(action):
    rows = [ (state, _tok(term), act)
             for (state, term), act in action.items() ]
    rows.sort(key=lambda r: (r[0], r[1]))

    w_state = max(len("State"), max(len(str(r[0])) for r in rows))
    w_sym   = max(len("Sym"),   max(len(r[1])      for r in rows))
    w_act   = max(len("Action"),max(len(r[2])      for r in rows))

    header = f"{'State'.ljust(w_state)} | {'Sym'.ljust(w_sym)} | {'Action'.ljust(w_act)}"
    sep    = f"{'-'*w_state}-+-{'-'*w_sym}-+-{'-'*w_act}"
    print("\n=== ACTION TABLE ===")
    print(header)
    print(sep)
    for s, sym, act in rows:
        print(f"{str(s).ljust(w_state)} | {sym.ljust(w_sym).lower()} | {act.ljust(w_act)}")

def print_goto_table(goto_table):
    rows = [ (state, nt, nxt)
             for (state, nt), nxt in goto_table.items() ]
    rows.sort(key=lambda r: (r[0], r[1]))

    w_state = max(len("State"), max(len(str(r[0])) for r in rows))
    w_nt    = max(len("NonT"),  max(len(r[1])      for r in rows))
    w_nxt   = max(len("Next"),  max(len(str(r[2])) for r in rows))

    header = f"{'State'.ljust(w_state)} | {'NonT'.ljust(w_nt)} | {'Next'.ljust(w_nxt)}"
    sep    = f"{'-'*w_state}-+-{'-'*w_nt}-+-{'-'*w_nxt}"
    print("\n=== GOTO TABLE ===")
    print(header)
    print(sep)
    for s, nt, nxt in rows:
        print(f"{str(s).ljust(w_state)} | {nt.ljust(w_nt)} | {str(nxt).ljust(w_nxt)}")

def print_action_goto(action, goto_table):
    print_action_table(action)
    print_goto_table(goto_table)
