import ast_nodes as ast
from lark import lark, v_args, Transformer


expression_grammar = r'''
start: list_expr_opt
?list_expr: expr+ -> list_expr
?list_expr_opt: expr* -> list_expr

?expr: substitution_expr
    | definition_expr
    | weight_range_expr
    | steps_range_expr
    | text_expr

?weight_range_expr: "(" list_expr_opt ":" weight_range_weight ")" -> weight_expr
?weight_range_weight: weight | range{weight}
?weight: weight_num | substitution_expr -> flatten_opt
weight_num: sign (FLOAT | INTEGER) -> float_expr

?steps_range_expr: "[" steps_range_exprs steps_range_steps "]" -> range_expr
?steps_range_steps: step ("," step)* -> flatten_list
?steps_range_exprs: (list_expr_opt ":")+ -> flatten_list
?step: (step_num | substitution_expr)? -> flatten_opt
step_num: sign INTEGER -> step_expr

?definition_expr: "$" SYMBOL "=" expr list_expr -> assignment_expr
substitution_expr: "$" SYMBOL -> substitution_expr

?text_expr: (TEXT | DIGIT | COMMA)+ -> text_expr
TEXT: /[^\[\]():|=$\s\d\-+]/+

COMMA: ","

range{number}: range_number{number} "," range_number{number} -> flatten_list
range_number{number}: number? -> flatten_opt

sign: SIGN? -> flatten_opt
SIGN: "-" | "+"
%import common.DIGIT
%import common.CNAME -> SYMBOL
%import common.FLOAT
%import common.INT -> INTEGER

%import common.WS -> WHITESPACE
%ignore WHITESPACE
'''


class ExpressionTransformer(Transformer):
    @v_args(inline=True)
    def flatten_list(self, *args):
        return args

    @v_args(inline=True)
    def flatten_opt(self, arg=None):
        return arg

    def step_expr(self, args):
        args = [arg for arg in args if arg is not None]

        # `step + 1` because original language is off by 1
        return ast.LiftExpression(int(''.join(args)) + 1)

    def float_expr(self, args):
        args = filter(lambda arg: arg is not None, args)
        return ast.LiftExpression(float(''.join(args)))

    def list_expr(self, args):
        return ast.ListExpression(args)

    def weight_expr(self, args):
        if type(args[1]) is tuple:
            return ast.WeightInterpolationExpression(args[0], *args[1])
        else:
            return ast.WeightedExpression(*args)

    def range_expr(self, args):
        return ast.RangeExpression(*args)

    def assignment_expr(self, args):
        return ast.DeclarationExpression(*args)

    def substitution_expr(self, args):
        return ast.SubstitutionExpression(*args)

    def text_expr(self, args):
        return ast.LiftExpression(str(' '.join(args)))


expr_parser = lark.Lark(expression_grammar, parser='lalr', transformer=ExpressionTransformer())
parse_expression = expr_parser.parse

def transpile_prompt(prompt, steps):
    expression, prompt = parse_expression(prompt.lstrip())
    return expression.evaluate((0, steps))


if __name__ == '__main__':
    for i, prompt in enumerate([
        'some space separated text',
        '(legacy weighted prompt:-2.1)',
        'mixed (legacy weight:3.6) and text',
        'legacy [range begin:0] thingy',
        'legacy [range end::3] thingy',
        'legacy [[nested range::3]:2] thingy',
        'sugar [range:2,3] thingy',
        'sugar [(weight interpolation:1,2):0,1] thingy',
        'sugar [(weight interpolation:1,2):0,2] thingy',
        'sugar [(weight interpolation:1,2):0,3] thingy',
        'legacy [from:to:2] thingy',
    ]):
        for e in parse_expression(prompt).children:
            print(str(i) + ':\t' + e.evaluate((0, 5)))
