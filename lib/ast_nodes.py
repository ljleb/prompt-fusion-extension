from lib.catmull import compute_catmull
from lib.bezier import compute_on_curve_with_points as compute_bezier
from lib.linear import compute_linear
from lib.t_scaler import scale_t


class ListExpression:
    def __init__(self, expressions):
        self.__expressions = expressions

    def append_to_tensor(self, tensor, prompt_database, interpolation_functions, steps_range, total_steps, context):
        for expression in self.__expressions:
            tensor = expression.append_to_tensor(tensor, prompt_database, interpolation_functions, steps_range, total_steps, context)

        return tensor


class InterpolationExpression:
    def __init__(self, expressions, steps, function_name=None):
        assert len(expressions) >= 2
        assert len(steps) == len(expressions), 'the number of steps must be the same as the number of expressions'
        self.__expressions = expressions
        self.__steps = steps
        self.__function_name = function_name if function_name is not None else 'linear'

    def append_to_tensor(self, tensor, prompt_database, interpolation_functions, steps_range, total_steps, context):
        extended_tensor = []
        extended_prompt_database = []
        extended_functions = []

        for expr_i, expr in enumerate(self.__expressions):
            expr_database = prompt_database[:]
            expr_functions = []

            expr_tensor = tensor if type(tensor) is int else tensor[:]
            expr_tensor = expr.append_to_tensor(expr_tensor, expr_database, expr_functions, steps_range, total_steps, context)
            expr_tensor = _tensor_add(expr_tensor, len(extended_prompt_database))
            extended_tensor.append(expr_tensor)

            extended_prompt_database.extend(expr_database)
            extended_functions.append(expr_functions)

        prompt_database[:] = extended_prompt_database
        interpolation_functions.append((self.get_interpolation_function(steps_range, total_steps, context), extended_functions))
        return extended_tensor

    def get_interpolation_function(self, steps_range, total_steps, context):
        steps = list(self.__steps)
        if steps[0] is None:
            steps[0] = LiftExpression(steps_range[0])
        if steps[-1] is None:
            steps[-1] = LiftExpression(steps_range[1])

        mock_database = ['']
        for i, step in enumerate(steps):
            step.append_to_tensor([0], mock_database, [], steps_range, total_steps, context)
            step = float(mock_database[0])
            if 0 < step < 1:
                step = steps_range[0] + step * (steps_range[1] - steps_range[0])

            steps[i] = int(step)
            mock_database[0] = ''

        interpolation_function = {
            'catmull': compute_catmull,
            'linear': compute_linear,
            'bezier': compute_bezier,
        }[self.__function_name]

        def steps_scale_t(t, conditionings):
            scaled_t = (t * total_steps - steps[0]) / (steps[-1] - steps[0])
            scaled_t = scale_t(scaled_t, steps)
            return interpolation_function(scaled_t, conditionings)

        return steps_scale_t


def _tensor_add(tensor, value):
    try:
        return tensor + value

    except TypeError:
        return [_tensor_add(e, value) for e in tensor]


class EditingExpression:
    def __init__(self, expressions, step):
        assert 1 <= len(expressions) <= 2
        self.__expressions = expressions
        self.__step = step

    def append_to_tensor(self, tensor, prompt_database, interpolation_functions, steps_range, total_steps, context):
        mock_database = ['']
        self.__step.append_to_tensor([0], mock_database, [], steps_range, total_steps, context)
        step = float(mock_database[0])
        if step == int(step):
            step = int(step)

        for i in range(len(prompt_database)):
            prompt_database[i] += '['

        for expr_index, expr in enumerate(self.__expressions):
            expr_steps_range = (steps_range[0], step) if expr_index == 0 else (step, steps_range[1])
            tensor = expr.append_to_tensor(tensor, prompt_database, interpolation_functions, expr_steps_range, total_steps, context)
            for i in range(len(prompt_database)):
                prompt_database[i] += ':'

        for i in range(len(prompt_database)):
            prompt_database[i] += f'{step - 1}]'

        return tensor


class WeightedExpression:
    def __init__(self, nested, weight=None, positive=True):
        self.__nested = nested
        if not positive:
            assert weight is None
        self.__weight = weight
        self.__positive = positive

    def append_to_tensor(self, tensor, prompt_database, interpolation_functions, steps_range, total_steps, context):
        if self.__positive:
            open_bracket = '('
            close_bracket = ')'
        else:
            open_bracket = '['
            close_bracket = ']'

        for i in range(len(prompt_database)):
            prompt_database[i] += open_bracket

        tensor = self.__nested.append_to_tensor(tensor, prompt_database, interpolation_functions, steps_range, total_steps, context)

        if self.__weight is not None:
            for i in range(len(prompt_database)):
                prompt_database[i] += ':'

            self.__weight.append_to_tensor(tensor, prompt_database, interpolation_functions, steps_range, total_steps, context)

        for i in range(len(prompt_database)):
            prompt_database[i] += close_bracket

        return tensor


class WeightInterpolationExpression:
    def __init__(self, nested, weight_begin, weight_end):
        self.__nested = nested
        self.__weight_begin = weight_begin if weight_begin is not None else LiftExpression(1.)
        self.__weight_end = weight_end if weight_end is not None else LiftExpression(1.)

    def append_to_tensor(self, tensor, prompt_database, interpolation_functions, steps_range, total_steps, context):
        steps_range_size = steps_range[1] - steps_range[0]

        mock_database = ['']
        self.__weight_begin.append_to_tensor([0], mock_database, [], steps_range, total_steps, context)
        weight_begin = float(mock_database[0])
        mock_database[0] = ''
        self.__weight_end.append_to_tensor([0], mock_database, [], steps_range, total_steps, context)
        weight_end = float(mock_database[0])

        for i in range(steps_range_size):
            step = i + steps_range[0]

            weight = weight_begin + (weight_end - weight_begin) * (i / max(steps_range_size - 1, 1))
            weight_step_expr = WeightedExpression(self.__nested, LiftExpression(weight))
            weight_step_expr = EditingExpression([weight_step_expr], LiftExpression(step))
            weight_step_expr = EditingExpression([weight_step_expr, ListExpression([])], LiftExpression(step + 1))

            tensor = weight_step_expr.append_to_tensor(tensor, prompt_database, interpolation_functions, steps_range, total_steps, context)

        return tensor


class DeclarationExpression:
    def __init__(self, symbol, nested, expression):
        self.__symbol = symbol
        self.__nested = nested
        self.__expression = expression

    def append_to_tensor(self, tensor, prompt_database, interpolation_functions, steps_range, total_steps, context):
        updated_context = dict(context)
        updated_context[self.__symbol] = self.__nested
        return self.__expression.append_to_tensor(tensor, prompt_database, interpolation_functions, steps_range, total_steps, updated_context)


class SubstitutionExpression:
    def __init__(self, symbol):
        self.__symbol = symbol

    def append_to_tensor(self, tensor, prompt_database, interpolation_functions, steps_range, total_steps, context):
        return context[self.__symbol].append_to_tensor(tensor, prompt_database, interpolation_functions, steps_range, total_steps, context)


class LiftExpression:
    def __init__(self, text):
        self.text = text

    def append_to_tensor(self, tensor, prompt_database, interpolation_functions, steps_range, total_steps, context):
        for i in range(len(prompt_database)):
            prompt_database[i] += str(self.text)

        return tensor
