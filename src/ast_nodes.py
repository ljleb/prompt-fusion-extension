class ListExpression:
    def __init__(self, expressions):
        self.expressions = expressions

    def evaluate(self, steps_range, context=dict()):
        return ' '.join([expression.evaluate(steps_range, context) for expression in self.expressions])

class DeclarationExpression:
    def __init__(self, symbol, nested, expression):
        self.symbol = symbol
        self.nested = nested
        self.expression = expression

    def evaluate(self, steps_range, context=dict()):
        updated_context = dict(context)
        updated_context[self.symbol] = self.nested.evaluate(steps_range, context)
        return self.expression.evaluate(steps_range, updated_context)

class RangeExpression:
    def __init__(self, nested, begin, end):
        self.nested = nested
        self.begin = begin
        self.end = end

    def evaluate(self, steps_range, context=dict()):
        new_steps_range = (
            max(self.begin, steps_range[0]) if self.begin else steps_range[0],
            min(self.end, steps_range[1]) if self.end else steps_range[1]
        )
        if new_steps_range[0] >= new_steps_range[1]: return ''

        result = self.nested.evaluate(new_steps_range, context)
        if self.begin and self.begin > steps_range[0]:
            result = f'[{result}:{new_steps_range[0] - 1}]'

        if self.end and self.end < steps_range[1]:
            result = f'[{result}::{new_steps_range[1] - 1}]'

        return result

class WeightedExpression:
    def __init__(self, nested, weight):
        self.nested = nested
        self.weight = weight

    def evaluate(self, steps_range, context=dict()):
        return f'({self.nested.evaluate(steps_range, context)}:{self.weight})'

class WeightInterpolationExpression:
    def __init__(self, nested, weight_begin, weight_end):
        self.nested = nested
        self.weight_begin = weight_begin
        self.weight_end = weight_end

    def evaluate(self, steps_range, context=dict()):
        total_steps = steps_range[1] - steps_range[0]
        result = ''
        for i in range(total_steps):
            step = i + steps_range[0]
            tmp_result = self.nested.evaluate((step, step + 1), context)
            if not tmp_result: continue

            weight = self.weight_begin + (self.weight_end - self.weight_begin) * (i / total_steps)
            tmp_result = f'({tmp_result}:{weight})'

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

class TextExpression:
    def __init__(self, text):
        self.text = text

    def evaluate(self, steps_range, context=dict()):
        return self.text
