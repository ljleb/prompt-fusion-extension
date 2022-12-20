class ListExpression:
    def __init__(self, expressions):
        self.expressions = expressions

    def evaluate(self, steps_range, context):
        expressions = filter(
            lambda e: e,
            [expression.evaluate(steps_range, context) for expression in self.expressions])
        return ' '.join(expressions)


class InterpolationExpression:
    def __init__(self, expressions, steps, function_name='linear'):
        assert len(expressions) > 0
        assert len(steps) > 0
        self.__expressions = expressions
        self.__steps = steps
        self.__function_name = function_name

    def evaluate(self, steps_range, context):
        result = ''
        if len(self.__steps) == 1:
            if len(self.__expressions) > 2:
                raise ValueError('more than 2 control points in instantaneous interpolation')

            for expression in self.__expressions:
                result += f'{expression.evaluate(steps_range, context)}:'

            return f'[{result}{self.__steps[0].evaluate(steps_range, context) - 1}]'

        elif len(self.__steps) > 1:
            if len(self.__expressions) > 1:
                raise ValueError('interpolation subexpressions using multiple control points are not supported yet')

            begin = self.__steps[0].evaluate(steps_range, context) if self.__steps[0] is not None else steps_range[0]
            end = self.__steps[1].evaluate(steps_range, context) if self.__steps[1] is not None else steps_range[1]
            new_steps_range = (max(begin, steps_range[0]), min(end, steps_range[1]))
            if new_steps_range[0] >= new_steps_range[1]:
                return ''

            result = self.__expressions[0].evaluate(new_steps_range, context)
            if begin > steps_range[0]:
                result = f'[{result}:{new_steps_range[0] - 1}]'

            if end < steps_range[1]:
                result = f'[{result}::{new_steps_range[1] - 1}]'

            return result

    def get_interpolation_conditioning(self, model, get_learned_conditioning, steps_range, context=None):
        from lib.interpolation_conditioning import InterpolationConditioning

        total_steps = steps_range[1]

        conditionings = []
        for expression in self.__expressions:
            prompt = expression.evaluate(steps_range, context)
            conditionings.append(get_learned_conditioning(model, [prompt], total_steps)[0][0].cond)

        control_points = []
        if len(self.__steps) < 2:
            control_points.append(0.)
            control_points.append(1.)

        else:
            if self.__steps[0] is not None:
                control_points.append(0.)
            else:
                control_point = self.__steps[0].evaluate(steps_range, context)
                control_points.append((control_point + 1) / total_steps)

            for step in self.__steps[1:-1]:
                control_point = step.evaluate(steps_range, context)
                control_points.append((control_point + 1) / total_steps)

            if self.__steps[-1] is not None:
                control_points.append(1.)
            else:
                control_point = self.__steps[-1].evaluate(steps_range, context)
                control_points.append((control_point + 1) / total_steps)

        return InterpolationConditioning(conditionings, control_points, self.get_curve_function())

    def get_curve_function(self):
        from lib.catmull import compute_catmull
        from lib.bezier import compute_on_curve_with_points as compute_bezier
        from lib.linear import compute_linear

        return {
            'catmull': compute_catmull,
            'linear': compute_linear,
            'bezier': compute_bezier,
        }[self.__function_name]


class WeightedExpression:
    def __init__(self, nested, weight=None, positive=True):
        self.nested = nested
        if not positive:
            assert weight is None
        self.weight = weight
        self.positive = positive

    def evaluate(self, steps_range, context):
        result = self.nested.evaluate(steps_range, context)

        if self.positive:
            if self.weight is not None:
                result = f'{result}:{self.weight.evaluate(steps_range, context)}'
            return f'({result})'

        else:
            return f'[{result}]'


class WeightInterpolationExpression:
    def __init__(self, nested, weight_begin, weight_end):
        self.nested = nested
        self.weight_begin = weight_begin
        self.weight_end = weight_end

    def evaluate(self, steps_range, context):
        total_steps = steps_range[1] - steps_range[0]
        result = ''
        weight_begin = self.weight_begin.evaluate(steps_range, context) if self.weight_begin is not None else 1
        weight_end = self.weight_end.evaluate(steps_range, context) if self.weight_end is not None else 1

        for i in range(total_steps):
            step = i + steps_range[0]
            inner_text = self.nested.evaluate((step, step + 1), context)
            if not inner_text: continue

            weight = weight_begin + (weight_end - weight_begin) * (i / total_steps)
            equivalent_expr = WeightedExpression(LiftExpression(inner_text), LiftExpression(weight))
            equivalent_expr = InterpolationExpression([equivalent_expr], [LiftExpression(step), LiftExpression(step + 1)])
            result += equivalent_expr.evaluate(steps_range, context)

        return result


class DeclarationExpression:
    def __init__(self, symbol, nested, expression):
        self.symbol = symbol
        self.nested = nested
        self.expression = expression

    def evaluate(self, steps_range, context):
        updated_context = dict(context) if context is not None else {}
        updated_context[self.symbol] = self.nested.evaluate(steps_range, context)
        return self.expression.evaluate(steps_range, updated_context)

    def get_interpolation_conditioning(self, model, get_learned_conditioning, steps_range, context=None):
        updated_context = dict(context) if context is not None else {}
        updated_context[self.symbol] = self.nested.evaluate(steps_range, context)
        return self.expression.get_interpolation_conditioning(model, get_learned_conditioning, steps_range, updated_context)


class SubstitutionExpression:
    def __init__(self, symbol):
        self.symbol = symbol

    def evaluate(self, steps_range, context):
        context = context if context is not None else {}
        return context[self.symbol]


class LiftExpression:
    def __init__(self, text):
        self.text = text

    def evaluate(self, steps_range, context):
        return self.text
