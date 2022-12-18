class ListExpression:
    def __init__(self, expressions):
        self.expressions = expressions

    def evaluate(self, steps_range, context=dict()):
        evaluations = [expression.evaluate(steps_range, context) for expression in self.expressions]
        return ' '.join([evaluation for evaluation in evaluations if evaluation])


class DeclarationExpression:
    def __init__(self, symbol, nested, expression):
        self.symbol = symbol
        self.nested = nested
        self.expression = expression

    def evaluate(self, steps_range, context=dict()):
        updated_context = dict(context)
        updated_context[self.symbol] = self.nested.evaluate(steps_range, context)
        return self.expression.evaluate(steps_range, updated_context)


class WeightedExpression:
    def __init__(self, nested, weight):
        self.nested = nested
        self.weight = weight

    def evaluate(self, steps_range, context=dict()):
        return f"({self.nested.evaluate(steps_range, context)}:{self.weight.evaluate(steps_range, context)})"


class RangeExpression:
    def __init__(self, nested, begin, end):
        self.nested = nested
        self.begin = begin
        self.end = end

    def evaluate(self, steps_range, context=dict()):
        begin = self.begin.evaluate(steps_range, context) if self.begin is not None else steps_range[0]
        end = self.end.evaluate(steps_range, context) if self.end is not None else steps_range[1]
        new_steps_range = (max(begin, steps_range[0]), min(end, steps_range[1]))
        if new_steps_range[0] >= new_steps_range[1]:
            return ''

        result = self.nested.evaluate(new_steps_range, context)
        if begin > steps_range[0]:
            result = f'[{result}:{new_steps_range[0] - 1}]'

        if end < steps_range[1]:
            result = f'[{result}::{new_steps_range[1] - 1}]'

        return result


class WeightInterpolationExpression:
    def __init__(self, nested, weight_begin, weight_end):
        self.nested = nested
        self.weight_begin = weight_begin
        self.weight_end = weight_end

    def evaluate(self, steps_range, context=dict()):
        total_steps = steps_range[1] - steps_range[0]
        result = ''
        for i in range(total_steps):
            weight_begin = self.weight_begin.evaluate(steps_range, context) if self.weight_begin is not None else 1
            weight_end = self.weight_end.evaluate(steps_range, context) if self.weight_end is not None else 1
            step = i + steps_range[0]
            inner_text = self.nested.evaluate((step, step + 1), context)
            if not inner_text: continue

            weight = weight_begin + (weight_end - weight_begin) * (i / total_steps)
            tmp_result = f'({inner_text}:{weight})'

            if step > steps_range[0]:
                tmp_result = f'[{tmp_result}:{step - 1}]'

            if step < steps_range[1] - 1:
                tmp_result = f'[{tmp_result}::{step}]'

            result += tmp_result

        return result


class SubstitutionExpression:
    def __init__(self, symbol):
        self.symbol = symbol

    def evaluate(self, steps_range, context=dict()):
        return context[self.symbol]


class ConversionExpression:
    def __init__(self, nested, converter):
        self.nested = nested
        self.converter = converter

    def evaluate(self, steps_range, context=dict()):
        return self.converter(self.nested.evaluate(steps_range, context))


class LiftExpression:
    def __init__(self, text):
        self.text = text

    def evaluate(self, steps_range, context=dict()):
        return self.text
