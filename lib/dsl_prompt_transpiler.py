import lib.ast_nodes as ast
import re
from lark import lark, v_args, Transformer


expression_grammar = r'''
?start: list_expr_opt
?list_expr: expr+ -> list_expr
?list_expr_opt: expr* -> list_expr

?expr: substitution_expr
     | definition_expr
     | weight_range_expr
     | interpolation_expr
     | text_expr

?weight_range_expr: "(" list_expr_opt weight_range_weight ")" -> weight_expr
                  | "[" list_expr_opt "]" -> negative_weight_expr
?weight_range_weight: (":" (weight | range{weight}))? -> flatten_opt
?weight: weight_num | substitution_expr
weight_num: weight_num_txt -> float_expr
?weight_num_txt: SIGN? (FLOAT | INTEGER) -> concat

?interpolation_expr: "[" interpolation_subexprs interpolation_single_step "]" -> interpolation_expr
                   | "[" interpolation_subexprs interpolation_steps interpolation_parameter "]" -> interpolation_expr
?interpolation_subexprs: (list_expr_opt ":")+ -> flatten_list
?interpolation_single_step: step -> flatten_list
?interpolation_steps: step ("," step)+ -> flatten_list
?step: (step_num | substitution_expr)? -> flatten_opt
step_num: step_num_txt -> step_expr
step_num_txt: SIGN? INTEGER -> concat

?interpolation_parameter: (":" INTERPOLATION_FUNCTION)? -> flatten_opt
INTERPOLATION_FUNCTION: "linear"
                      | "catmull"
                      | "bezier"

?definition_expr: "$" SYMBOL "=" expr list_expr -> assignment_expr
substitution_expr: "$" SYMBOL -> substitution_expr

?text_expr: TEXT+ -> text_expr
TEXT: /([^\[\]\(\):$\\\s]|\\.)+/
COMMA.1: ","

range{number}: range_number{number} "," range_number{number} -> flatten_list
range_number{number}: number? -> flatten_opt

SIGN: "-" | "+"

INTEGER.1: FREE_INTEGER /\b/
FLOAT.1: FREE_FLOAT /\b/

%import common.CNAME -> SYMBOL
%import common.FLOAT -> FREE_FLOAT
%import common.INT -> FREE_INTEGER

%import common.WS -> WHITESPACE
%ignore WHITESPACE
'''


class ExpressionLarkTransformer(Transformer):
    @v_args(inline=True)
    def flatten_list(self, *args):
        return args

    @v_args(inline=True)
    def flatten_opt(self, arg=None):
        return arg

    @v_args(inline=True)
    def concat(self, *args):
        return ''.join(args)

    def step_expr(self, args):
        args = [arg for arg in args if arg is not None]

        # `step + 1` because original language is off by 1
        return ast.LiftExpression(int(''.join(args)) + 1)

    def float_expr(self, args):
        args = filter(lambda arg: arg is not None, args)
        return ast.LiftExpression(float(''.join(args)))

    def list_expr(self, args):
        if len(args) == 1:
            return args[0]
        return ast.ListExpression(args)

    def weight_expr(self, args):
        if type(args[1]) is tuple:
            return ast.WeightInterpolationExpression(args[0], *args[1])
        else:
            return ast.WeightedExpression(*args)

    def negative_weight_expr(self, args):
        return ast.WeightedExpression(*args, positive=False)

    def interpolation_expr(self, args):
        function_name = str(args[2]) if args[2:] else None
        return ast.InterpolationExpression(args[0], args[1], function_name)

    def assignment_expr(self, args):
        return ast.DeclarationExpression(*args)

    def substitution_expr(self, args):
        return ast.SubstitutionExpression(*args)

    def text_expr(self, args):
        backslash_pattern = re.compile(r'\\(.)')
        return ast.LiftExpression(backslash_pattern.sub(r'\1', ' '.join(args)))


expr_parser = lark.Lark(expression_grammar, parser='lalr', transformer=ExpressionLarkTransformer())
parse_expression = expr_parser.parse


def transpile_prompt(prompt, steps):
    expression = parse_prompt(prompt)
    return expression.evaluate((0, steps), None)


def parse_prompt(prompt):
    expression = parse_expression(prompt.lstrip())
    if not hasattr(expression, 'get_interpolation_conditioning'):
        return ast.InterpolationExpression([expression], [ast.LiftExpression(0.)])
    else:
        return expression


if __name__ == '__main__':
    for i, prompt in enumerate([
        ['single']*2,
        ['some space separated text']*2,
        ['(legacy weighted prompt:-2.1)']*2,
        ['mixed (legacy weight:3.6) and text']*2,
        ['legacy [range begin:0] thingy']*2,
        ['legacy [range end::3] thingy']*2,
        ['legacy [[nested range::3]:2] thingy']*2,
        ['legacy [[nested range:2]::3] thingy']*2,
        ('sugar [range:2,3] thingy', 'sugar [[range:2]::3] thingy'),
        ('sugar [range:2,] thingy', 'sugar [range:2] thingy'),
        ('sugar [range:,3] thingy', 'sugar [range::3] thingy'),
        # (r'sugar [range:,abc:3] thingy', 'sugar [range:,abc:3] thingy'),
        ('sugar [(weight interpolation:0,12):0,1] thingy', 'sugar [[(weight interpolation:0.0):0]::1] thingy'),
        ('sugar [(weight interpolation:0,12):0,2] thingy', 'sugar [[[(weight interpolation:0.0)::1][(weight interpolation:6.0):1]:0]::2] thingy'),
        ('sugar [(weight interpolation:0,12):0,3] thingy', 'sugar [[[(weight interpolation:0.0)::1][[(weight interpolation:4.0):1]::2][(weight interpolation:8.0):2]:0]::3] thingy'),
        ['legacy [from:to:2] thingy']*2,
        ['legacy [negative weight]']*2,
        ['legacy (positive weight)']*2,
        ['[abc:1girl:2]']*2,
        ['1girl']*2,
        ['dashes-in-text']*2,
        ['text, separated with, comas']*2,
        ['{prompt}']*2,
        ['[abc|def ghi|jkl]']*2,
        ['merging this AND with this']*2,
        ('$a = (prompt value:1) $a', '(prompt value:1.0)'),
        ('$a = (prompt value:1) $b = $a $b', '(prompt value:1.0)'),
        (r'\:', ':'),

        # ['[top level:interpolatin:lik a pro:1,3,5: linear]']*2,
    ]):
        try:
            e = parse_expression(prompt[0])
            v = e.evaluate((0, 5), None)
            assert v == prompt[1], f"'{v}' != '{prompt[1]}'"
        except Exception as e:
            print(prompt[0])
            raise e
