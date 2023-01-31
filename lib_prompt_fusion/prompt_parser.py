if __name__ == '__main__':
    import sys
    sys.path.append('.')

import lib_prompt_fusion.ast_nodes as ast
from lib_prompt_fusion.interpolation_tensor import InterpolationTensorBuilder
import re
from lark import lark, v_args, Transformer


expression_grammar = r'''
?start: expr* and_weight_expr? -> list_expr
?list_expr: expr+ -> list_expr
?list_expr_opt: expr* -> list_expr

?and_weight_expr: ":" FREE_FLOAT -> and_weight_expr

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

?interpolation_expr: "[" interpolation_subexprs "]" -> interpolation_expr
?interpolation_subexprs: list_expr_opt (":" list_expr_opt)+ -> flatten_list

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

    @v_args(inline=True)
    def and_weight_expr(self, weight):
        return ast.LiftExpression(f':{weight}')

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

    @v_args(inline=True)
    def interpolation_expr(self, subexprs):
        if str(subexprs[-1]) in {'linear', 'catmull', 'bezier'}:
            function_name = str(subexprs[-1])
            subexprs = subexprs[:-1]
        else:
            function_name = None

        steps = [None if not step
                 else ast.SubstitutionExpression(step[1:]) if step.startswith('$')
                 # `step + 1` because original language is off by 1
                 else ast.LiftExpression(float(step) if 0 < float(step) < 1 else int(float(step) + 1))
                 for step in str(subexprs[-1]).split(',')]
        subexprs = subexprs[:-1]
        if len(steps) > 1:
            return ast.InterpolationExpression(subexprs, steps, function_name)
        else:
            assert function_name is None, 'bad prompt editing syntax'
            return ast.EditingExpression(subexprs, steps[0])

    def assignment_expr(self, args):
        return ast.DeclarationExpression(*args)

    def substitution_expr(self, args):
        return ast.SubstitutionExpression(*args)

    def text_expr(self, args):
        backslash_pattern = re.compile(r'\\(.)')
        return ast.LiftExpression(backslash_pattern.sub(r'\1', ' '.join(args)))


expr_parser = lark.Lark(expression_grammar, parser='lalr', transformer=ExpressionLarkTransformer())
parse_expression = expr_parser.parse


def parse_prompt(prompt):
    return parse_expression(prompt.lstrip())


def run_parser_tests(total_steps):
    for i, prompt in enumerate([
        ('single',)*2,
        ('some space separated text',)*2,
        ('(legacy weighted prompt:-2.1)',)*2,
        ('mixed (legacy weight:3.6) and text',)*2,
        ('legacy [range begin:0] thingy',)*2,
        ('legacy [range end::3] thingy',)*2,
        ('legacy [[nested range::3]:2] thingy',)*2,
        ('legacy [[nested range:2]::3] thingy',)*2,
        ('sugar [range:,abc:3] thingy',)*2,
        ('sugar [[(weight interpolation:0,12):0]::1] thingy', 'sugar [[(weight interpolation:0.0):0]::1] thingy'),
        ('sugar [[(weight interpolation:0,12):0]::2] thingy', 'sugar [[[(weight interpolation:0.0)::1][(weight interpolation:12.0):1]:0]::2] thingy'),
        ('sugar [[(weight interpolation:0,12):0]::3] thingy', 'sugar [[[(weight interpolation:0.0)::1][[(weight interpolation:6.0):1]::2][(weight interpolation:12.0):2]:0]::3] thingy'),
        ('legacy [from:to:2] thingy',)*2,
        ('legacy [negative weight]',)*2,
        ('legacy (positive weight)',)*2,
        ('[abc:1girl:2]',)*2,
        ('1girl',)*2,
        ('dashes-in-text',)*2,
        ('text, separated with, comas',)*2,
        ('{prompt}',)*2,
        ('[abc|def ghi|jkl]',)*2,
        ('merging this AND with this',)*2,
        ('$a = (prompt value:1) $a', '(prompt value:1.0)'),
        ('$a = (prompt value:1) $b = $a $b', '(prompt value:1.0)'),
        (r'\:', ':'),
        ('a [b:c:-1, 10] d', {'a b d', 'a c d'}),
        ('a [b:c:5, 6] d', {'a b d', 'a c d'}),
        ('a [b:c:0.25, 0.5] d', {'a b d', 'a c d'}),
        ('a [b:c:.25, .5] d', {'a b d', 'a c d'}),
        ('a [b:c:,] d', {'a b d', 'a c d'}),
        ('0[1.0:1.1:,]2[3.0:3.1:,]4', {'0 1.0 2 3.0 4', '0 1.1 2 3.0 4', '0 1.0 2 3.1 4', '0 1.1 2 3.1 4'}),
        ('0[1.0:1.1:1.2:,.5,]2[3.0:3.1:,]4', {
            '0 1.0 2 3.0 4', '0 1.0 2 3.1 4',
            '0 1.1 2 3.0 4', '0 1.1 2 3.1 4',
            '0 1.2 2 3.0 4', '0 1.2 2 3.1 4',
        }),
        ('[0.0:0.1:,][1.0:1.1:,][2.0:2.1:,]', {
            '0.0 1.0 2.0', '0.0 1.0 2.1',
            '0.1 1.0 2.0', '0.1 1.0 2.1',
            '0.0 1.1 2.0', '0.0 1.1 2.1',
            '0.1 1.1 2.0', '0.1 1.1 2.1',
        }),
        ('[top level:interpolatin:lik a pro:1,3,5: linear]', {'top level', 'interpolatin', 'lik a pro'}),
    ]):
        try:
            expr = parse_expression(prompt[0])
            tensor_builder = InterpolationTensorBuilder()
            expr.extend_tensor(tensor_builder, (0, total_steps), total_steps, dict())

            actual = tensor_builder.get_prompt_database()

            if type(prompt[1]) is set:
                message = f"parse('{expr}') != {prompt[1]}"
                assert set(actual) == prompt[1], message
            else:
                message = f"parse('{expr}') != '{prompt[1]}'"
                assert len(actual) == 1 and actual[0] == prompt[1], message
        except Exception as e:
            raise e


if __name__ == '__main__':
    run_parser_tests(total_steps=100)
