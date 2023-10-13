import math
from lib_prompt_fusion import interpolation_functions
from lib_prompt_fusion.t_scaler import scale_t
from lib_prompt_fusion import interpolation_tensor


class ListExpression:
    def __init__(self, expressions):
        self.__expressions = expressions

    def extend_tensor(self, tensor_builder, steps_range, total_steps, context, is_hires):
        if not self.__expressions:
            return

        def expr_extend_tensor(expr):
            expr.extend_tensor(tensor_builder, steps_range, total_steps, context, is_hires)

        expr_extend_tensor(self.__expressions[0])
        for expression in self.__expressions[1:]:
            tensor_builder.append(' ')
            expr_extend_tensor(expression)


class InterpolationExpression:
    def __init__(self, expressions, steps, function_name=None):
        assert len(expressions) >= 2
        assert len(steps) == len(expressions), 'the number of steps must be the same as the number of expressions'
        self.__expressions = expressions
        self.__steps = steps
        self.__function_name = function_name if function_name is not None else 'linear'

    def extend_tensor(self, tensor_builder, steps_range, total_steps, context, is_hires):
        def tensor_updater(expr):
            return lambda t: expr.extend_tensor(t, steps_range, total_steps, context, is_hires)

        tensor_builder.extrude(
            [tensor_updater(expr) for expr in self.__expressions],
            self.get_interpolation_function(steps_range, total_steps, context, is_hires))

    def get_interpolation_function(self, steps_range, total_steps, context, is_hires):
        steps = list(self.__steps)
        if steps[0] is None:
            steps[0] = LiftExpression(str(steps_range[0] - 1))
        if steps[-1] is None:
            steps[-1] = LiftExpression(str(steps_range[1] - 1))

        for i, step in enumerate(steps):
            if step is None:
                continue

            step = _eval_int_or_float(step, steps_range, total_steps, context, is_hires)

            if isinstance(step, float):
                step = int((step - int(is_hires)) * total_steps)
            else:
                step += 1

            steps[i] = step

        i = 1
        while i < len(steps):
            none_len = 0
            while steps[i + none_len] is None:
                none_len += 1

            min_step, max_step = steps[i - 1], steps[i + none_len]

            for j in range(none_len):
                steps[i + j] = min_step + (max_step - min_step) * (j + 1) / (none_len + 1)

            i += 1 + none_len

        interpolation_function = {
            'linear': interpolation_functions.compute_linear,
            'bezier': interpolation_functions.compute_bezier,
            'catmull': interpolation_functions.compute_catmull,
        }[self.__function_name]

        def steps_scale_t(conds, params: interpolation_tensor.InterpolationParams):
            scaled_t = (params.t * total_steps - steps[0]) / max(1, steps[-1] - steps[0])
            scaled_t = scale_t(scaled_t, steps)

            new_params = interpolation_tensor.InterpolationParams(scaled_t, *params[1:])
            return interpolation_function(conds, new_params)

        return steps_scale_t


class AlternationExpression:
    def __init__(self, expressions, speed):
        self.__expressions = expressions
        self.__speed = speed

    def extend_tensor(self, tensor_builder, steps_range, total_steps, context, is_hires):
        if self.__speed is None:
            speed = None
        else:
            speed = _eval_int_or_float(self.__speed, steps_range, total_steps, context, is_hires)

        if speed is None:
            tensor_builder.append('[')
            for expr_i, expr in enumerate(self.__expressions):
                if expr_i >= 1:
                    tensor_builder.append('|')
                expr.extend_tensor(tensor_builder, steps_range, total_steps, context, is_hires)
            tensor_builder.append(']')
            return

        def tensor_updater(expr):
            return lambda t: expr.extend_tensor(t, steps_range, total_steps, context, is_hires)

        exprs = self.__expressions + [self.__expressions[0]]

        tensor_builder.extrude(
            [tensor_updater(expr) for expr in exprs],
            self.get_interpolation_function(speed, exprs, steps_range, total_steps))

    def get_interpolation_function(self, speed, exprs, steps_range, total_steps):
        def compute_wrap(control_points, params: interpolation_tensor.InterpolationParams):
            wrapped_t = math.fmod((params.t * total_steps - steps_range[0]) / (len(exprs) - 1) * speed, 1.0)
            if wrapped_t < 0:
                wrapped_t = wrapped_t + 1
            new_params = interpolation_tensor.InterpolationParams(wrapped_t, *params[1:])
            return interpolation_functions.compute_linear(control_points, new_params)

        return compute_wrap


class EditingExpression:
    def __init__(self, expressions, step):
        assert 1 <= len(expressions) <= 2
        self.__expressions = expressions
        self.__step = step

    def extend_tensor(self, tensor_builder, steps_range, total_steps, context, is_hires):
        if self.__step is None:
            tensor_builder.append('[')
            for expr_i, expr in enumerate(self.__expressions):
                expr.extend_tensor(tensor_builder, steps_range, total_steps, context, is_hires)
                tensor_builder.append(':')
            tensor_builder.append(']')
            return

        step = _eval_int_or_float(self.__step, steps_range, total_steps, context, is_hires)
        if isinstance(step, float):
            step = int((step - int(is_hires)) * total_steps)
        else:
            step += 1

        tensor_builder.append('[')
        for expr_i, expr in enumerate(self.__expressions):
            expr_steps_range = (steps_range[0], step) if expr_i == 0 and len(self.__expressions) >= 2 else (step, steps_range[1])
            expr.extend_tensor(tensor_builder, expr_steps_range, total_steps, context, is_hires)
            tensor_builder.append(':')

        tensor_builder.append(f'{step - 1}]')


class WeightedExpression:
    def __init__(self, nested, weight=None, positive=True):
        self.__nested = nested
        if not positive:
            assert weight is None

        self.__weight = weight
        self.__positive = positive

    def extend_tensor(self, tensor_builder, steps_range, total_steps, context, is_hires):
        open_bracket, close_bracket = ('(', ')') if self.__positive else ('[', ']')
        tensor_builder.append(open_bracket)
        self.__nested.extend_tensor(tensor_builder, steps_range, total_steps, context, is_hires)

        if self.__weight is not None:
            tensor_builder.append(':')
            self.__weight.extend_tensor(tensor_builder, steps_range, total_steps, context, is_hires)

        tensor_builder.append(close_bracket)


class WeightInterpolationExpression:
    def __init__(self, nested, weight_begin, weight_end):
        self.__nested = nested
        self.__weight_begin = weight_begin if weight_begin is not None else LiftExpression(str(1.))
        self.__weight_end = weight_end if weight_end is not None else LiftExpression(str(1.))

    def extend_tensor(self, tensor_builder, steps_range, total_steps, context, is_hires):
        steps_range_size = steps_range[1] - steps_range[0]

        weight_begin = _eval_int_or_float(self.__weight_begin, steps_range, total_steps, context, is_hires)
        weight_end = _eval_int_or_float(self.__weight_end, steps_range, total_steps, context, is_hires)

        for i in range(steps_range_size):
            step = i + steps_range[0]

            weight = weight_begin + (weight_end - weight_begin) * (i / max(steps_range_size - 1, 1))
            weight_step_expr = WeightedExpression(self.__nested, LiftExpression(str(weight)))
            if step > steps_range[0]:
                weight_step_expr = EditingExpression([weight_step_expr], LiftExpression(str(step - 1)))
            if step + 1 < steps_range[1]:
                weight_step_expr = EditingExpression([weight_step_expr, ListExpression([])], LiftExpression(str(step)))

            weight_step_expr.extend_tensor(tensor_builder, steps_range, total_steps, context, is_hires)


class DeclarationExpression:
    def __init__(self, symbol, parameters, value, target):
        self.__symbol = symbol
        self.__value = value
        self.__target = target
        self.__parameters = parameters

    def extend_tensor(self, tensor_builder, steps_range, total_steps, context, is_hires):
        updated_context = dict(context)
        updated_context[self.__symbol] = (self.__value, self.__parameters)
        self.__target.extend_tensor(tensor_builder, steps_range, total_steps, updated_context, is_hires)


class SubstitutionExpression:
    def __init__(self, symbol, arguments):
        self.__symbol = symbol
        self.__arguments = arguments

    def extend_tensor(self, tensor_builder, steps_range, total_steps, context, is_hires):
        updated_context = dict(context)
        nested, parameters = context[self.__symbol]
        for argument, parameter in zip(self.__arguments, parameters):
            updated_context[parameter] = argument, []
        nested.extend_tensor(tensor_builder, steps_range, total_steps, updated_context, is_hires)


class LiftExpression:
    def __init__(self, value):
        self.__value = value

    def extend_tensor(self, tensor_builder, *_args, **_kwargs):
        tensor_builder.append(self.__value)


def _eval_int_or_float(expression, steps_range, total_steps, context, is_hires):
    mock_database = ['']
    expression.extend_tensor(interpolation_tensor.InterpolationTensorBuilder(prompt_database=mock_database), steps_range, total_steps, context, is_hires)
    try:
        return int(mock_database[0])
    except ValueError:
        return float(mock_database[0])
