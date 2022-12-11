from extensions.promptlang.src.ast_nodes import (
    ListExpression,
    DeclarationExpression,
    WeightedExpression,
    RangeExpression,
    TextExpression,
    SubstitutionExpression,
    WeightInterpolationExpression,
)


def make_strict_token_parser(characters):
    return make_token_parser(characters, lambda s, cs: s is cs, loop=False)


def make_token_parser(characters, matcher=lambda s, cs: s in cs, loop=True):
    def parse_token(source):
        if not source:
            raise ValueError
        index = 0
        for i in range(len(source)):
            if not matcher(source[i], characters):
                if i == 0:
                    raise ValueError
                break
            index = i
            if not loop:
                break

        next_prompt_begin = index
        for i in range(index + 1, len(source)):
            if source[i] not in set(' \n\r\t'):
                break
            next_prompt_begin = i

        return source[:index + 1], source[next_prompt_begin + 1:]

    return parse_token


def transpile_prompt(prompt, steps):
    expression, prompt = parse_expression(prompt.lstrip())
    return expression.evaluate((0, steps))


def parse_expression(prompt):
    expressions = []
    while True:
        try:
            expression, prompt = parse_recursive_expression(prompt)
            expressions.append(expression)
        except ValueError:
            break
    return ListExpression(expressions), prompt


def parse_recursive_expression(prompt):
    try:
        return parse_declaration(prompt)
    except ValueError:
        expression, prompt = parse_atom_expression(prompt)
        try:
            return parse_range_expression(expression, prompt)
        except ValueError:
            pass
        try:
            return parse_weighted_expression(expression, prompt)
        except ValueError:
            pass
        return expression, prompt


def parse_declaration(prompt):
    symbol, prompt = parse_symbol(prompt)
    _, prompt = parse_assignment_separator(prompt)
    value, prompt = parse_atom_expression(prompt)
    expression, prompt = parse_expression(prompt)
    return DeclarationExpression(symbol, value, expression), prompt


parse_assignment_separator = make_strict_token_parser('=')


def parse_range_expression(expression, prompt):
    (range_begin, range_end), prompt = parse_range(prompt, int)
    return RangeExpression(expression, range_begin, range_end), prompt


def parse_number(prompt, number_type):
    number, prompt = parse_digits(prompt)
    return number_type(number), prompt


parse_digits = make_token_parser('0123456789.')


def parse_weighted_expression(expression, prompt):
    _, prompt = parse_weight_separator(prompt)
    try:
        weight, prompt = parse_number(prompt, float)
        return WeightedExpression(expression, weight), prompt
    except ValueError:
        pass
    (weight_begin, weight_end), prompt = parse_range(prompt, float)
    return WeightInterpolationExpression(expression, weight_begin, weight_end), prompt


parse_weight_separator = make_strict_token_parser(':')


def parse_range(prompt, number_type):
    _, prompt = parse_range_begin(prompt)
    try:
        range_begin, prompt = parse_number(prompt, number_type)
    except ValueError:
        range_begin = None

    _, prompt = parse_range_separator(prompt)
    try:
        range_end, prompt = parse_number(prompt, number_type)
    except ValueError:
        range_end = None

    _, prompt = parse_range_end(prompt)
    return (range_begin, range_end), prompt


parse_range_begin = make_strict_token_parser('[')
parse_range_end = make_strict_token_parser(']')
parse_range_separator = make_strict_token_parser(',')


def parse_atom_expression(prompt):
    try:
        return parse_parentheses_expression(prompt)
    except ValueError:
        pass
    try:
        return parse_substitution_expression(prompt)
    except ValueError:
        text, prompt = parse_text(prompt)
        return TextExpression(text), prompt


def parse_parentheses_expression(prompt):
    _, prompt = parse_parenthesis_begin(prompt)
    expression, prompt = parse_expression(prompt)
    _, prompt = parse_parenthesis_end(prompt)
    return expression, prompt


parse_parenthesis_begin = make_strict_token_parser('(')
parse_parenthesis_end = make_strict_token_parser(')')


def parse_substitution_expression(prompt):
    _, prompt = parse_substitution_begin(prompt)
    symbol, prompt = parse_symbol(prompt)
    return SubstitutionExpression(symbol), prompt


parse_substitution_begin = make_strict_token_parser('$')

parse_text = make_token_parser(' \n\r\t:[]()', lambda s, cs: s not in cs)

parse_symbol = make_token_parser(
    'abcdefghijklmnopqrstuvwxyz' +
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ' +
    '_')

if __name__ == "__main__":
    print(transpile_prompt("""(extremely cute, beautiful and delicate (1girl, solo)[0,7] (loli girl)[7,30]):1.2
caustics,

(frosty colors with frost):0.8
(flashy sparkling red tunic with white stripes):0.7
(extremely intricate spiky tattoos):1.2
(very bright glowing yellow eyes):1.1

(wearing a japanese hanbok):1.1""", 30))
