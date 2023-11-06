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
    if not shared.opts.data.get("prompt_fusion_enabled", False):
        schedule = yield params
        return schedule

    tensor_builder = _parse_tensor_builder(params)
    flattened_conds = yield from _encode_all_prompts(params, tensor_builder.get_prompt_database())
    cond_tensor = tensor_builder.build(flattened_conds, params.empty_cond)
    schedule = _sample_schedules(cond_tensor, params)
    return schedule


def _parse_tensor_builder(params: sdlib.EncodePromptScheduleParams):
    expr = prompt_fusion_parser.parse_prompt(params.prompt)
    tensor_builder = interpolation_tensor.InterpolationTensorBuilder()
    expr.extend_tensor(tensor_builder, (0, params.steps), params.steps, dict(), params.pass_index, use_old_scheduling=False)
    return tensor_builder


def _encode_all_prompts(params: sdlib.EncodePromptScheduleParams, prompts):
    conds = []
    for prompt in prompts:
        cond = yield dataclasses.replace(params, prompt=prompt)
        conds.append(cond)
    return conds


def _sample_schedules(cond_tensor: interpolation_tensor.InterpolationTensor, params: sdlib.EncodePromptScheduleParams):
    schedules = []
    for step in range(params.steps):
        origin_cond = global_state.get_origin_cond_at(step, params.empty_cond, params.negative_schedule)
        interpolation_params = interpolation_tensor.InterpolationParams(
            t=step/params.steps,
            step=step,
            total_steps=params.steps,
            slerp_scale=global_state.get_slerp_scale(),
            slerp_epsilon=global_state.get_slerp_epsilon(),
        )
        cond = cond_tensor.interpolate(interpolation_params, origin_cond, params.empty_cond)
        schedules.append(cond)
    return torch.stack(schedules)
