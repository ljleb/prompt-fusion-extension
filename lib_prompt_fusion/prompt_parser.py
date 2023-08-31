import lib_prompt_fusion.ast_nodes as ast
from collections import namedtuple
import re


ParseResult = namedtuple('ParseResult', ['prompt', 'expr'])


def parse_prompt(prompt):
    prompt = prompt.lstrip()
    prompt, list_expr = parse_list_expression(prompt, set())
    return list_expr


def parse_list_expression(prompt, stoppers):
    exprs = []
    try:
        while True:
            prompt, expr = parse_expression(prompt, stoppers)
            exprs.append(expr)
    except ValueError:
        return ParseResult(prompt=prompt, expr=ast.ListExpression(exprs))


def parse_expression(prompt, stoppers):
    for parse in _parsers():
        try:
            return parse(prompt, stoppers)
        except ValueError:
            pass

    raise ValueError


def _parsers():
    return (
        parse_text,
        parse_declaration,
        parse_substitution,
        parse_positive_attention,
        parse_negative_attention,
        parse_editing,
        parse_interpolation,
        parse_unrestricted_text,
    )


def parse_text(prompt, stoppers):
    return parse_unrestricted_text(prompt, set_concat(stoppers, {'[', '(', '$'}))


def parse_unrestricted_text(prompt, stoppers):
    escaped_stoppers = ''.join(re.escape(stopper) for stopper in stoppers)
    regex = rf'(?:[^{escaped_stoppers}\\\s]|\$(?![a-zA-Z_])|\\.)+'
    prompt, expr = parse_token(prompt, whitespace_tail_regex(regex, stoppers))
    return ParseResult(prompt=prompt, expr=ast.LiftExpression(expr))


def parse_substitution(prompt, stoppers):
    prompt, symbol = parse_symbol(prompt, stoppers)
    prompt, arguments = parse_arguments(prompt, stoppers)
    return ParseResult(prompt=prompt, expr=ast.SubstitutionExpression(symbol, arguments))


def parse_arguments(prompt, stoppers):
    try:
        prompt, _ = parse_open_paren(prompt, stoppers)
        prompt, arguments = parse_inner_arguments(prompt, stoppers)
        prompt, _ = parse_close_paren(prompt, stoppers)
    except ValueError:
        arguments = []
    return ParseResult(prompt=prompt, expr=arguments)


def parse_inner_arguments(prompt, stoppers):
    arguments = []
    try:
        while True:
            prompt, arg = parse_list_expression(prompt, {',', ')'})
            arguments.append(arg)
            prompt, _ = parse_comma(prompt, stoppers)
    except ValueError:
        pass

    return ParseResult(prompt=prompt, expr=arguments)


def parse_declaration(prompt, stoppers):
    prompt, symbol = parse_symbol(prompt, stoppers)
    prompt, parameters = parse_parameters(prompt, stoppers)
    prompt, _ = parse_equals(prompt, stoppers)
    prompt, value = parse_list_expression(prompt, set_concat(stoppers, '\n'))
    prompt, _ = parse_newline(prompt, stoppers)
    prompt, expr = parse_list_expression(prompt, stoppers)
    return ParseResult(prompt=prompt, expr=ast.DeclarationExpression(symbol, parameters, value, expr))


def parse_parameters(prompt, stoppers):
    try:
        prompt, _ = parse_open_paren(prompt, stoppers)
        prompt, parameters = parse_inner_parameters(prompt, stoppers)
        prompt, _ = parse_close_paren(prompt, stoppers)
    except ValueError:
        parameters = []
    return ParseResult(prompt=prompt, expr=parameters)


def parse_inner_parameters(prompt, stoppers):
    parameters = []
    try:
        while True:
            prompt, param = parse_symbol(prompt, stoppers)
            parameters.append(param)
            prompt, _ = parse_comma(prompt, stoppers)
    except ValueError:
        pass

    return ParseResult(prompt=prompt, expr=parameters)


def parse_interpolation(prompt, stoppers):
    prompt, _ = parse_open_square(prompt, stoppers)
    prompt, exprs = parse_interpolation_exprs(prompt, stoppers)
    prompt, steps = parse_interpolation_steps(prompt, stoppers)
    prompt, function_name = parse_interpolation_function_name(prompt, stoppers)
    prompt, _ = parse_close_square(prompt, stoppers)

    max_len = min(len(exprs), len(steps))
    exprs = exprs[:max_len]
    steps = steps[:max_len]

    return ParseResult(prompt=prompt, expr=ast.InterpolationExpression(exprs, steps, function_name))


def parse_interpolation_exprs(prompt, stoppers):
    exprs = []

    try:
        while True:
            prompt_tmp, expr = parse_list_expression(prompt, {':', ']'})
            if parse_interpolation_function_name(prompt_tmp, stoppers).expr is not None:
                raise ValueError

            prompt, _ = parse_colon(prompt_tmp, stoppers)
            exprs.append(expr)
    except ValueError:
        pass

    return ParseResult(prompt=prompt, expr=exprs)


def parse_interpolation_function_name(prompt, stoppers):
    try:
        prompt, _ = parse_colon(prompt, stoppers)
        function_names = ('linear', 'catmull', 'bezier')
        return parse_token(prompt, whitespace_tail_regex('|'.join(function_names), stoppers))
    except ValueError:
        return ParseResult(prompt=prompt, expr=None)


def parse_interpolation_steps(prompt, stoppers):
    steps = []

    try:
        while True:
            prompt, step = parse_interpolation_step(prompt, stoppers)
            steps.append(step)
            prompt, _ = parse_comma(prompt, stoppers)
    except ValueError:
        pass

    return ParseResult(prompt=prompt, expr=steps)


def parse_interpolation_step(prompt, stoppers):
    try:
        return parse_step(prompt, stoppers)
    except ValueError:
        pass

    if prompt[0] in {',', ':', ']'}:
        return ParseResult(prompt=prompt, expr=None)

    raise ValueError


def parse_editing(prompt, stoppers):
    prompt, _ = parse_open_square(prompt, stoppers)
    prompt, exprs = parse_editing_exprs(prompt, stoppers)
    try:
        prompt, step = parse_step(prompt, stoppers)
    except ValueError:
        step = None

    prompt, _ = parse_close_square(prompt, stoppers)
    return ParseResult(prompt=prompt, expr=ast.EditingExpression(exprs, step))


def parse_editing_exprs(prompt, stoppers):
    exprs = []

    try:
        for _ in range(2):
            prompt_tmp, expr = parse_list_expression(prompt, {':', ']'})
            prompt, _ = parse_colon(prompt_tmp, stoppers)
            exprs.append(expr)
    except ValueError:
        pass

    return ParseResult(prompt=prompt, expr=exprs)


def parse_negative_attention(prompt, stoppers):
    prompt, _ = parse_open_square(prompt, stoppers)
    prompt, expr = parse_list_expression(prompt, set_concat(stoppers, {':', ']'}))
    prompt, _ = parse_close_square(prompt, stoppers)
    return ParseResult(prompt=prompt, expr=ast.WeightedExpression(expr, positive=False))


def parse_positive_attention(prompt, stoppers):
    prompt, _ = parse_open_paren(prompt, stoppers)
    prompt, expr = parse_list_expression(prompt, {':', ')'})
    prompt, weight_exprs = parse_attention_weights(prompt, stoppers)
    prompt, _ = parse_close_paren(prompt, stoppers)
    if len(weight_exprs) >= 2:
        return ParseResult(prompt=prompt, expr=ast.WeightInterpolationExpression(expr, *weight_exprs[:2]))
    else:
        return ParseResult(prompt=prompt, expr=ast.WeightedExpression(expr, *weight_exprs[:1]))


def parse_attention_weights(prompt, stoppers):
    weights = []
    try:
        prompt, _ = parse_colon(prompt, stoppers)
    except ValueError:
        return ParseResult(prompt=prompt, expr=weights)

    while True:
        try:
            prompt, weight_expr = parse_weight(prompt, stoppers)
            weights.append(weight_expr)
            prompt, _ = parse_comma(prompt, stoppers)
        except ValueError:
            return ParseResult(prompt=prompt, expr=weights)


def parse_step(prompt, stoppers):
    try:
        prompt, step = parse_float(prompt, stoppers)
        return ParseResult(prompt=prompt, expr=ast.LiftExpression(step))
    except ValueError:
        pass

    return parse_substitution(prompt, stoppers)


def parse_weight(prompt, stoppers):
    try:
        prompt, step = parse_float(prompt, stoppers)
        return ParseResult(prompt=prompt, expr=ast.LiftExpression(step))
    except ValueError:
        pass

    return parse_substitution(prompt, stoppers)


def parse_symbol(prompt, stoppers):
    prompt, _ = parse_dollar(prompt)
    return parse_symbol_text(prompt, stoppers)


def parse_symbol_text(prompt, stoppers):
    return parse_token(prompt, whitespace_tail_regex('[a-zA-Z_][a-zA-Z0-9_]*', stoppers))


def parse_float(prompt, stoppers):
    return parse_token(prompt, whitespace_tail_regex(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)', stoppers))


def parse_dollar(prompt):
    dollar_sign = re.escape('$')
    return parse_token(prompt, f'({dollar_sign})')


def parse_equals(prompt, stoppers):
    return parse_token(prompt, whitespace_tail_regex(re.escape('='), stoppers))


def parse_comma(prompt, stoppers):
    return parse_token(prompt, whitespace_tail_regex(re.escape(','), stoppers))


def parse_colon(prompt, stoppers):
    return parse_token(prompt, whitespace_tail_regex(re.escape(':'), stoppers))


def parse_open_square(prompt, stoppers):
    return parse_token(prompt, whitespace_tail_regex(re.escape('['), stoppers))


def parse_close_square(prompt, stoppers):
    return parse_token(prompt, whitespace_tail_regex(re.escape(']'), stoppers))


def parse_open_paren(prompt, stoppers):
    return parse_token(prompt, whitespace_tail_regex(re.escape('('), stoppers))


def parse_close_paren(prompt, stoppers):
    return parse_token(prompt, whitespace_tail_regex(re.escape(')'), stoppers))


def parse_newline(prompt, stoppers):
    return parse_token(prompt, whitespace_tail_regex('\n|$', stoppers))


def parse_token(prompt, regex):
    match = re.match(regex, prompt)
    if match is None:
        raise ValueError

    return ParseResult(prompt=prompt[len(match.group()):], expr=match.groups()[-1])


def whitespace_tail_regex(regex, stoppers):
    if '\n' in stoppers:
        return rf'({regex})[ \t\f\r]*'

    return rf'({regex})\s*'


def set_concat(left, right):
    result = set(left)
    result.update(right)
    return result
