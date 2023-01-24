import modules.scripts as scripts
import sys
base_dir = scripts.basedir()
sys.path.append(base_dir)

from lib.dsl_prompt_transpiler import parse_prompt
from lib.hijacker import prompt_parser_hijacker
from modules.prompt_parser import ScheduledPromptConditioning

import torch


@prompt_parser_hijacker.hijack('get_learned_conditioning')
def hijacked_get_learned_conditioning(model, prompts, total_steps, original_function):
    flattened_prompts, prompts_interpolation_info = parse_interpolation_info(prompts, total_steps)
    flattened_conditionings = original_function(model, flattened_prompts, total_steps)
    return schedule_conditionings(flattened_conditionings, prompts_interpolation_info, total_steps)


def parse_interpolation_info(prompts, total_steps):
    flattened_prompts = []
    prompts_interpolation_info = []

    for prompt in prompts:
        expr = parse_prompt(prompt)
        tensor = 0
        prompt_database = ['']
        interpolation_functions = []

        tensor = expr.append_to_tensor(tensor, prompt_database, interpolation_functions, (0, total_steps), total_steps, dict())

        prompts_interpolation_info.append((len(flattened_prompts), len(prompt_database), (tensor, interpolation_functions)))
        flattened_prompts.extend(prompt_database)

    return flattened_prompts, prompts_interpolation_info


def schedule_conditionings(flattened_conditionings, prompts_interpolation_info, steps):
    scheduled_conditionings = []
    for embeds_start_index, embeds_database_size, embed_interpolation_info in prompts_interpolation_info:
        indices_tensor, interpolation_functions = embed_interpolation_info
        conditioning_database = [embed[0].cond for embed in flattened_conditionings[embeds_start_index:embeds_start_index + embeds_database_size]]
        tensor = _resolve_conditionings(indices_tensor, conditioning_database)

        interpolated_conditionings = []
        for step in range(steps):
            interpolated_conditioning = _interpolate_tensor(tensor, interpolation_functions, step / max(steps - 1, 1))
            if len(interpolated_conditionings) > 0 and torch.all(torch.eq(interpolated_conditionings[-1].cond, interpolated_conditioning)):
                interpolated_conditionings[-1] = ScheduledPromptConditioning(end_at_step=step, cond=interpolated_conditionings[-1].cond)
            else:
                interpolated_conditionings.append(ScheduledPromptConditioning(end_at_step=step, cond=interpolated_conditioning))

        scheduled_conditionings.append(interpolated_conditionings)

    return scheduled_conditionings


def _resolve_conditionings(tensor, conditionings):
    if type(tensor) is int:
        return conditionings[tensor]
    else:
        return [_resolve_conditionings(e, conditionings) for e in tensor]


def _interpolate_tensor(tensor, interpolation_functions, t):
    tensor_axes = len(interpolation_functions)
    if tensor_axes == 0:
        return tensor

    interpolation_function, control_points_functions = interpolation_functions[0]
    if tensor_axes == 1:
        control_points = list(tensor)
    else:
        control_points = [_interpolate_tensor(sub_tensor, interpolation_functions[1:], t) for sub_tensor in tensor]

    for i, nested_functions in enumerate(control_points_functions):
        if nested_functions:
            control_points[i] = _interpolate_tensor(control_points[i], nested_functions, t)

    return interpolation_function(t, control_points)


class FusionScript(scripts.Script):
    def title(self):
        return "fusion"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def run(self, p, *args):
        pass
