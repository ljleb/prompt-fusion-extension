from lib.t_scaler import apply_sampled_range
from modules import prompt_parser


class InterpolationConditioning:
    def __init__(self, conditionings, control_points, curve_function):
        self.conditionings = conditionings
        self.control_points = control_points
        self.curve_function = curve_function

    def to_scheduled_conditionings(self, steps):
        cond_array = []
        for i in range(steps):
            t = apply_sampled_range(i / max(1, steps - 1), self.control_points)
            cond_array.append(
                prompt_parser.ScheduledPromptConditioning(end_at_step=i, cond=self.curve_function(t, self.conditionings)))
        return cond_array
