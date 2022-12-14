import lib.ast_nodes as ast
from lark import lark


grammar = """
start: expr_list

expr: paren_expr
    | substitution_expr
    | weight_expr
    | steps_expr

?paren_expr: "(" expr_list ")"
?expr_list: expr*

weight_expr: expr ":" (weight | range{weight})
weight: WEIGHT | substitution_expr

steps_expr: expr range{STEP | substitution_expr}

definition_expr: "$" SYMBOL "=" expr expr_list
substitution_expr: "$" SYMBOL

TEXT: (/[^\\[\\]():|=$]/ | "\\" /[\\[\\]():|=$]/)*

range{NUMBER}: "[" NUMBER? "," NUMBER? "]"

STEP: SIGN INTEGER
WEIGHT: SIGN FLOAT

FLOAT: (INTEGER "."? | INTEGER? "." INTEGER)
SIGN: /[+-]/? 
INTEGER: "0" | "1".."9" DIGIT_CHAR*

SYMBOL: ALPHA_CHAR (ALPHA_CHAR | DIGIT_CHAR)*
ALPHA_CHAR: "a".."z" | "A".."Z" | "_"
DIGIT_CHAR: "0".."9"

%import common.WS -> WHITESPACE
%ignore WHITESPACE
"""


def transpile_prompt(prompt, steps):
    expression, prompt = parse_expression(prompt.lstrip())
    return expression.evaluate((0, steps))
