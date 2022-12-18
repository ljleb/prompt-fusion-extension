import ast_nodes as ast
from lark import lark, v_args, Transformer


expression_grammar = r"""
start: list_expr_opt
list_expr: expr+ -> list_expr
list_expr_opt: list_expr? -> list_expr_opt

expr: substitution_expr
    | weight_range_expr
    | steps_range_expr
    | text_expr

weight_range_expr: "(" list_expr_opt SINGLE_COLON (weight | range{weight}) ")" -> weight_expr
?weight: weight_num | substitution_expr
weight_num: (FLOAT | INTEGER) -> float_expr

steps_range_expr: "[" list_expr_opt steps_colon (step | range{step}) "]" -> range_expr
?step: step_num | substitution_expr
step_num: INTEGER -> int_expr

?steps_colon: SINGLE_COLON | DOUBLE_COLON
SINGLE_COLON: ":"
DOUBLE_COLON: "::"

definition_expr: "$" SYMBOL "=" list_expr -> assignment_expr
substitution_expr: "$" SYMBOL -> substitution_expr

text_expr: TEXT -> text_expr
TEXT: (/[^\[\]():|=$\s]/ | "\\" /[\[\]():|=$\s]/)+

range{number}: range_begin{number} "," number? -> tuple_expr
range_begin{number}: number? -> range_begin

%import common.CNAME -> SYMBOL
%import common.FLOAT
%import common.INT -> INTEGER

%import common.WS -> WHITESPACE
%ignore WHITESPACE
"""

@v_args(inline=True)
class ExpressionTransformer(Transformer):
    def range_begin(self, value=None):
        return value

    def tuple_expr(self, left, right=None):
        return left, right

    def int_expr(self, value):
        return ast.LiftExpression(int(value) + 1)

    def float_expr(self, value):
        return ast.LiftExpression(float(value))

    def list_expr(self, *expressions):
        expressions = [children for expression in expressions for children in expression.children]
        return ast.ListExpression(expressions)

    def list_expr_opt(self, list_expr=ast.ListExpression([])):
        return list_expr

    def weight_expr(self, nested, _colon, weight):
        if type(weight) is tuple:
            return ast.WeightInterpolationExpression(
                nested,
                ast.ConversionExpression(weight[0], float) if weight[0] is not None else None,
                ast.ConversionExpression(weight[1], float) if weight[1] is not None else None)
        else:
            return ast.WeightedExpression(nested, weight)

    def range_expr(self, expr, colon, steps):
        if type(steps) is tuple:
            return ast.RangeExpression(expr, steps[0], steps[1])
        elif str(colon) == ":":
            return ast.RangeExpression(expr, steps, None)
        else:
            return ast.RangeExpression(expr, None, steps)

    def assignment_expr(self, symbol, value):
        return ast.DeclarationExpression(symbol, value)

    def substitution_expr(self, symbol):
        return ast.SubstitutionExpression(symbol)

    def text_expr(self, value):
        return ast.LiftExpression(str(value))


expr_parser = lark.Lark(expression_grammar, parser='lalr', transformer=ExpressionTransformer())
parse_expression = expr_parser.parse

def transpile_prompt(prompt, steps):
    expression, prompt = parse_expression(prompt.lstrip())
    return expression.evaluate((0, steps))


if __name__ == '__main__':
    prompt = '[(abc:2,3):,1]'
    for e in parse_expression(prompt).children:
        print(e.evaluate((0, 20)))
