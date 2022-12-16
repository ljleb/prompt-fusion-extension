import ast_nodes as ast
from lark import lark, v_args, Transformer


grammar = r"""
start: list_expr?
list_expr: expr+ -> list_expr

expr: paren_expr
    | substitution_expr
    | weight_expr
    | steps_expr
    | text_expr

?paren_expr: "(" list_expr ")"

weight_expr: expr ":" (weight | range{weight}) -> weight_expr
weight: weight_num | substitution_expr
weight_num: (FLOAT | INTEGER) -> float_expr

steps_expr: expr range{steps_expr_range} -> range_expr
steps_expr_range: step | substitution_expr
step: INTEGER -> int_expr

definition_expr: "$" SYMBOL "=" list_expr -> assignment_expr
substitution_expr: "$" SYMBOL -> substitution_expr

text_expr: TEXT -> text_expr
TEXT: (/(!<:\[\s*[0-9.]*)/ "," | /[^\[\](),:|=$]/ | "\\" /[\[\](),:|=$]/)+

range{number}: range_begin{number} "," number? "]" -> tuple_expr
range_begin{number}: "[" number? -> range_begin

%import common.CNAME -> SYMBOL
%import common.FLOAT
%import common.INT -> INTEGER

%import common.WS -> WHITESPACE
%ignore WHITESPACE
"""

@v_args(inline=True)
class CalculateTree(Transformer):
    def range_begin(self, value=None):
        return value

    def tuple_expr(self, left, right=None):
        left = left.children[0] if left is not None else None
        right = right.children[0] if right is not None else None
        return left, right

    def int_expr(self, value):
        return ast.TextExpression(int(value))

    def float_expr(self, value):
        return ast.TextExpression(float(value))

    def list_expr(self, expressions):
        expressions = expressions.children
        return ast.ListExpression(expressions)

    def weight_expr(self, nested, weight):
        nested = nested.children[0]
        if type(weight) is tuple:
            return ast.WeightInterpolationExpression(
                nested,
                ast.ConversionExpression(weight[0], float) if weight[0] is not None else None,
                ast.ConversionExpression(weight[1], float) if weight[1] is not None else None)
        else:
            return ast.WeightedExpression(nested, ast.ConversionExpression(weight, float))

    def range_expr(self, expr, range_):
        print(range_)
        return ast.RangeExpression(expr, range_[0], range_[1])

    def assignment_expr(self, symbol, value):
        return ast.DeclarationExpression(symbol, value)

    def substitution_expr(self, symbol):
        return ast.SubstitutionExpression(symbol)

    def text_expr(self, value):
        return ast.TextExpression(value)


expr_parser = lark.Lark(grammar, parser='lalr', transformer=CalculateTree())
parse_expression = expr_parser.parse

def transpile_prompt(prompt, steps):
    expression, prompt = parse_expression(prompt.lstrip())
    return expression.evaluate((0, steps))


if __name__ == '__main__':
    prompt = 'abc[1,]'
    for e in parse_expression(prompt).children:
        print(e.evaluate((0, 20)))
