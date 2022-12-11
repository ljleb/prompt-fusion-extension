import extensions.promptlang.src.ast_nodes as ast


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
    return parse_alternator_expression(prompt)


def parse_alternator_expression(prompt):
    expressions = []
    head_expression, prompt = parse_list_expression(prompt)
    expressions.append(head_expression)
    try:
        while True:
            _, prompt = parse_alternator_separator(prompt)
            expression, prompt = parse_list_expression(prompt)
            expressions.append(expression)
    except ValueError:
        if len(expressions) > 1:
            return ast.AlternatorExpression(expressions), prompt
        else:
            return expressions[0], prompt


parse_alternator_separator = make_strict_token_parser('|')


def parse_list_expression(prompt):
    expressions = []
    try:
        while True:
            expression, prompt = parse_recursive_expression(prompt)
            expressions.append(expression)
    except ValueError:
        return ast.ListExpression(expressions), prompt


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
    return ast.DeclarationExpression(symbol, value, expression), prompt


parse_assignment_separator = make_strict_token_parser('=')


def parse_range_expression(expression, prompt):
    (range_begin, range_end), prompt = parse_range(prompt, int)
    return ast.RangeExpression(expression, range_begin, range_end), prompt


def parse_number(prompt, number_type):
    number, prompt = parse_digits(prompt)
    return number_type(number), prompt


parse_digits = make_token_parser('0123456789.')


def parse_weighted_expression(expression, prompt):
    _, prompt = parse_weight_separator(prompt)
    try:
        weight, prompt = parse_number(prompt, float)
        return ast.WeightedExpression(expression, weight), prompt
    except ValueError:
        pass
    (weight_begin, weight_end), prompt = parse_range(prompt, float)
    return ast.WeightInterpolationExpression(expression, weight_begin, weight_end), prompt


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
        return ast.TextExpression(text), prompt


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
    return ast.SubstitutionExpression(symbol), prompt


parse_substitution_begin = make_strict_token_parser('$')

parse_text = make_token_parser(' \n\r\t:[]()|=', lambda s, cs: s not in cs)

parse_symbol = make_token_parser(
    'abcdefghijklmnopqrstuvwxyz' +
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ' +
    '_')
