import modules.scripts as scripts
import sys
base_dir = scripts.basedir()
sys.path.append(base_dir)

from lib.interpolation_tensor import InterpolationTensorBuilder
from lib.dsl_prompt_transpiler import parse_prompt
from lib.hijacker import ModuleHijacker
import modules
from modules.script_callbacks import on_script_unloaded
from modules.prompt_parser import ScheduledPromptConditioning
import torch


fusion_hijacker_attribute = '__fusion_hijacker'
prompt_parser_hijacker = ModuleHijacker.install_or_get(modules.prompt_parser, fusion_hijacker_attribute, on_script_unloaded)


@prompt_parser_hijacker.hijack('get_learned_conditioning')
def _hijacked_get_learned_conditioning(model, prompts, total_steps, original_function):
    tensor_builders = _parse_tensor_builders(prompts, total_steps)
    flattened_prompts, consecutive_ranges = _get_flattened_prompts(tensor_builders)

    flattened_conditionings = original_function(model, flattened_prompts, total_steps)
    conditionings_tensors = [tensor_builder.build(flattened_conditionings[begin:end])
                             for begin, end, tensor_builder
                             in zip(consecutive_ranges[:-1], consecutive_ranges[1:], tensor_builders)]

    return [_schedule_conditionings(conditionings_tensor, total_steps)
            for conditionings_tensor in conditionings_tensors]


def _parse_tensor_builders(prompts, total_steps):
    tensor_builders = []

    for prompt in prompts:
        expr = parse_prompt(prompt)
        tensor_builder = InterpolationTensorBuilder()
        expr.extend_tensor(tensor_builder, (0, total_steps), total_steps, dict())
        tensor_builders.append(tensor_builder)

    return tensor_builders


def _get_flattened_prompts(tensor_builders):
    flattened_prompts = []
    consecutive_ranges = [0]

    for tensor_builder in tensor_builders:
        flattened_prompts.extend(tensor_builder.get_prompt_database())
        consecutive_ranges.append(len(flattened_prompts))

    return flattened_prompts, consecutive_ranges


def _schedule_conditionings(tensor, steps):
    interpolated_conditionings = []
    for step in range(steps):
        interpolated_conditioning = tensor.interpolate(step / steps, step)
        if len(interpolated_conditionings) > 0 and torch.all(torch.eq(interpolated_conditionings[-1].cond, interpolated_conditioning)):
            interpolated_conditionings[-1] = ScheduledPromptConditioning(end_at_step=step, cond=interpolated_conditionings[-1].cond)
        else:
            interpolated_conditionings.append(ScheduledPromptConditioning(end_at_step=step, cond=interpolated_conditioning))

    return interpolated_conditionings


class FusionScript(scripts.Script):
    def title(self):
        return "fusion"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def run(self, p, *args):
        pass
