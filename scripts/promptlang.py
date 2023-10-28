import dataclasses
import gradio as gr
import torch
from lib_prompt_fusion import global_state, interpolation_tensor, prompt_parser as prompt_fusion_parser
from modules import script_callbacks, shared
import sdlib


plugin = sdlib.register_plugin("prompt-fusion")


def on_ui_settings():
    section = ('prompt-fusion', 'Prompt Fusion')
    shared.opts.add_option('prompt_fusion_enabled', shared.OptionInfo(True, 'Enable prompt-fusion extension', section=section))
    shared.opts.add_option('prompt_fusion_slerp_scale', shared.OptionInfo(0, 'Slerp scale (0 = linear geometry, 1 = slerp geometry)', component=gr.Number, section=section))
    shared.opts.add_option('prompt_fusion_slerp_negative_origin', shared.OptionInfo(True, 'use negative prompt as slerp origin', section=section))
    shared.opts.add_option('prompt_fusion_slerp_epsilon', shared.OptionInfo(0.0001, 'Slerp epsilon (fallback on linear geometry when conds are too similar. 0 = parallel, 1 = perpendicular)', component=gr.Number, section=section))


script_callbacks.on_ui_settings(on_ui_settings)


# TODO - verify function signature in wrapper()
@plugin.wrapper
def encode_prompt(params: sdlib.EncodePromptScheduleParams):
    tensor_builder = _parse_tensor_builder(params.prompt, params.steps, params.pass_index)
    flattened_conds = yield from _encode_all_prompts(params, tensor_builder.get_prompt_database())
    cond_tensor = tensor_builder.build(flattened_conds, params.empty_cond)
    schedule = _sample_schedules(cond_tensor, params.steps, params.pass_index, params.empty_cond)
    return schedule


def _parse_tensor_builder(prompt: str, steps: int, pass_index: int):
    expr = prompt_fusion_parser.parse_prompt(prompt)
    tensor_builder = interpolation_tensor.InterpolationTensorBuilder()
    expr.extend_tensor(tensor_builder, (0, steps), steps, dict(), pass_index, use_old_scheduling=False)
    return tensor_builder


def _encode_all_prompts(params: sdlib.EncodePromptScheduleParams, prompts):
    conds = []
    for prompt in prompts:
        # TODO - memoize original function. clear cache when returning from the last wrapper
        cond = yield dataclasses.replace(params, prompt=prompt)
        conds.append(cond)

    return conds


def _sample_schedules(cond_tensor: interpolation_tensor.InterpolationTensor, steps: int, pass_index: int, empty_cond: torch.Tensor):
    schedules = []
    for step in range(steps):
        # TODO - origin cond does not work fully anymore because negative prompts are not guaranteed to be processed first
        #          -> either this plugin should be able to influence the control flow, or negative prompts are guaranteed to be processed first
        origin_cond = global_state.get_origin_cond_at(step, pass_index, empty_cond)
        params = interpolation_tensor.InterpolationParams(step / steps, step, steps, global_state.get_slerp_scale(), global_state.get_slerp_epsilon())
        cond = cond_tensor.interpolate(params, origin_cond, empty_cond)
        if schedules and (schedules[-1] == cond).all():
            schedules.append(schedules[-1])
        else:
            schedules.append(cond)

    return torch.stack(schedules, dim=-5)
