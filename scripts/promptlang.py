import modules.scripts as scripts
import sys
base_dir = scripts.basedir()
sys.path.append(base_dir)

from lib.dsl_prompt_transpiler import parse_prompt
from lib.hijacker import prompt_parser_hijacker
from modules.prompt_parser import ScheduledPromptConditioning

import torch

def _resolve_embeds(tensor, embeds):
    if type(tensor) is int:
        return embeds[tensor]
    else:
        return [_resolve_embeds(e, embeds) for e in tensor]


def _interpolate_tensor(t, interpolation_functions, embed_tensor, tensor_axes):
    if tensor_axes == 0:
        return embed_tensor
    if tensor_axes == 1:
        return interpolation_functions[0][0](t, embed_tensor)

    control_points = [_interpolate_tensor(t, interpolation_functions[1:], e, tensor_axes - 1) for e in embed_tensor]
    return interpolation_functions[0][0](t, control_points)


@prompt_parser_hijacker.hijack('get_learned_conditioning')
def hijacked_get_learned_conditioning(model, prompts, steps, original_function):
    scheduled_conditionings = []

    flattened_prompts = []
    prompts_interpolation_info = []

    total_prompts = 0
    for prompt in prompts:
        expr = parse_prompt(prompt)
        tensor = 0
        prompt_database = ['']
        interpolation_functions = []

        tensor = expr.append_to_tensor(tensor, prompt_database, interpolation_functions, (0, steps), dict())

        flattened_prompts.extend(prompt_database)
        prompts_interpolation_info.append((total_prompts, len(prompt_database), (tensor, interpolation_functions)))
        total_prompts += len(prompt_database)

    flattened_embeds = original_function(model, flattened_prompts, steps)

    for embeds_start_index, embeds_database_size, embed_interpolation_info in prompts_interpolation_info:
        tensor, interpolation_functions = embed_interpolation_info
        embed_database = [embed[0].cond for embed in flattened_embeds[embeds_start_index:embeds_start_index + embeds_database_size]]
        embed_tensor = _resolve_embeds(tensor, embed_database)
        tensor_axes = len(interpolation_functions)

        interpolated_conditionings = []
        for step in range(steps):
            interpolated_conditioning = _interpolate_tensor(step / max(steps - 1, 1), interpolation_functions, embed_tensor, tensor_axes)
            if len(interpolated_conditionings) > 0 and torch.all(torch.eq(interpolated_conditionings[-1].cond, interpolated_conditioning)):
                interpolated_conditionings[-1] = ScheduledPromptConditioning(end_at_step=step, cond=interpolated_conditionings[-1].cond)
            else:
                interpolated_conditionings.append(ScheduledPromptConditioning(end_at_step=step, cond=interpolated_conditioning))

        scheduled_conditionings.append(interpolated_conditionings)

    return scheduled_conditionings


class FusionScript(scripts.Script):
    def title(self):
        return "fusion"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def run(self, p, *args):
        pass
