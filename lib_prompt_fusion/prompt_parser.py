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
    for parse in [parse_restricted_text, parse_positive_attention, parse_negative_attention, parse_editing, parse_interpolation, parse_text]:
        try:
            return parse(prompt, stoppers)
        except ValueError:
            pass

    raise ValueError


def parse_restricted_text(prompt, stoppers):
    return parse_text(prompt, set_concat(stoppers, {'[', '('}))


def parse_text(prompt, stoppers):
    stoppers = ''.join(re.escape(stopper) for stopper in stoppers)
    prompt, expr = parse_token(prompt, whitespace_tail_regex(rf'[^{stoppers}\s]+'))
    return ParseResult(prompt=prompt, expr=ast.LiftExpression(expr))


def parse_interpolation(prompt, stoppers):
    prompt, _ = parse_open_square(prompt)
    prompt, exprs = parse_interpolation_exprs(prompt, stoppers)
    prompt, steps = parse_interpolation_steps(prompt)
    prompt, function_name = parse_interpolation_function_name(prompt)
    prompt, _ = parse_close_square(prompt)

    max_len = min(len(exprs), len(steps))
    exprs = exprs[:max_len]
    steps = steps[:max_len]

    return ParseResult(prompt=prompt, expr=ast.InterpolationExpression(exprs, steps))


def parse_interpolation_function_name(prompt):
    try:
        prompt, _ = parse_colon(prompt)
        return parse_token(prompt, whitespace_tail_regex('|'.join(interpolation_function_names)))
    except ValueError:
        return ParseResult(prompt=prompt, expr=None)


interpolation_function_names = (
    'linear',
    'catmull',
    'bezier',
)


def parse_interpolation_exprs(prompt, stoppers):
    exprs = []

    try:
        while True:
            prompt_tmp, expr = parse_list_expression(prompt, set_concat(stoppers, {':', ']'}))
            prompt_tmp, _ = parse_colon(prompt_tmp)
            if prompt_tmp.startswith(interpolation_function_names):
                raise ValueError

            prompt = prompt_tmp
            exprs.append(expr)
    except ValueError:
        pass

    return ParseResult(prompt=prompt, expr=exprs)


def parse_interpolation_steps(prompt):
    exprs = []

    try:
        while True:
            prompt, expr = parse_interpolation_step(prompt)
            exprs.append(expr)
            prompt, _ = parse_comma(prompt)
    except ValueError:
        pass

    return ParseResult(prompt=prompt, expr=exprs)


def parse_interpolation_step(prompt):
    try:
        return parse_step(prompt)
    except ValueError:
        pass

    if prompt[0] in {',', ']'}:
        return ParseResult(prompt=prompt, expr=None)

    raise ValueError


def parse_editing(prompt, stoppers):
    prompt, _ = parse_open_square(prompt)
    prompt, exprs = parse_editing_exprs(prompt, stoppers)
    prompt, step = parse_step(prompt)
    prompt, _ = parse_close_square(prompt)
    return ParseResult(prompt=prompt, expr=ast.EditingExpression(exprs, step))


def parse_editing_exprs(prompt, stoppers):
    exprs = []

    try:
        for _ in range(2):
            prompt_tmp, expr = parse_list_expression(prompt, set_concat(stoppers, {':', ']'}))
            prompt, _ = parse_colon(prompt_tmp)
            exprs.append(expr)
    except ValueError:
        pass

    return ParseResult(prompt=prompt, expr=exprs)


def parse_negative_attention(prompt, stoppers):
    prompt, _ = parse_open_square(prompt)
    prompt, expr = parse_list_expression(prompt, set_concat(stoppers, {':', ']'}))
    prompt, _ = parse_close_square(prompt)
    return ParseResult(prompt=prompt, expr=ast.WeightedExpression(expr, positive=False))


def parse_positive_attention(prompt, stoppers):
    prompt, _ = parse_open_paren(prompt)
    prompt, expr = parse_list_expression(prompt, set_concat(stoppers, {':', ')'}))
    prompt, weight_exprs = parse_attention_weights(prompt)
    prompt, _ = parse_close_paren(prompt)
    if len(weight_exprs) >= 2:
        return ParseResult(prompt=prompt, expr=ast.WeightInterpolationExpression(expr, *weight_exprs[:2]))
    else:
        return ParseResult(prompt=prompt, expr=ast.WeightedExpression(expr, *weight_exprs[:1]))


def parse_attention_weights(prompt):
    weights = []
    try:
        prompt, _ = parse_colon(prompt)
    except ValueError:
        return ParseResult(prompt=prompt, expr=weights)

    while True:
        try:
            prompt, weight_expr = parse_weight(prompt)
            weights.append(weight_expr)
            prompt, _ = parse_comma(prompt)
        except ValueError:
            return ParseResult(prompt=prompt, expr=weights)


def parse_step(prompt):
    prompt, step = parse_float(prompt)
    step = step if 0 < float(step) < 1 else str(float(step)+1)
    return ParseResult(prompt=prompt, expr=ast.LiftExpression(step))


def parse_weight(prompt):
    prompt, step = parse_float(prompt)
    return ParseResult(prompt=prompt, expr=ast.LiftExpression(step))


def parse_float(prompt):
    return parse_token(prompt, whitespace_tail_regex(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)'))


def parse_comma(prompt):
    return parse_token(prompt, whitespace_tail_regex(','))


def parse_colon(prompt):
    return parse_token(prompt, whitespace_tail_regex(r'\:'))


def parse_open_square(prompt):
    return parse_token(prompt, whitespace_tail_regex(r'\['))


def parse_close_square(prompt):
    return parse_token(prompt, whitespace_tail_regex(r'\]'))


def parse_open_paren(prompt):
    return parse_token(prompt, whitespace_tail_regex(r'\('))


def parse_close_paren(prompt):
    return parse_token(prompt, whitespace_tail_regex(r'\)'))


def parse_token(prompt, regex):
    match = re.match(regex, prompt)
    if match is None:
        raise ValueError

    return ParseResult(prompt=prompt[len(match.group()):], expr=match.groups()[-1])


def whitespace_tail_regex(regex):
    return rf'({regex})\s*'


def set_concat(left, right):
    result = set(left)
    result.update(right)
    return result
