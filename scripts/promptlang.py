import gradio as gr
from lib_prompt_fusion import hijacker, empty_cond, global_state, interpolation_tensor, prompt_parser as prompt_fusion_parser
from modules import scripts, script_callbacks, prompt_parser, shared


fusion_hijacker_attribute = '__fusion_hijacker'
prompt_parser_hijacker = hijacker.ModuleHijacker.install_or_get(
    module=prompt_parser,
    hijacker_attribute=fusion_hijacker_attribute,
    register_uninstall=script_callbacks.on_script_unloaded)


def on_ui_settings():
    section = ('prompt-fusion', 'Prompt Fusion')
    shared.opts.add_option('prompt_fusion_enabled', shared.OptionInfo(True, 'Enable prompt-fusion extension', section=section))
    shared.opts.add_option('prompt_fusion_slerp_scale', shared.OptionInfo(0, 'Slerp scale (0 = linear geometry, 1 = slerp geometry)', component=gr.Number, section=section))
    shared.opts.add_option('prompt_fusion_slerp_negative_origin', shared.OptionInfo(True, 'use negative prompt as slerp origin', section=section))
    shared.opts.add_option('prompt_fusion_slerp_epsilon', shared.OptionInfo(0.0001, 'Slerp epsilon (fallback on linear geometry when conds are too similar. 0 = parallel, 1 = perpendicular)', component=gr.Number, section=section))


script_callbacks.on_ui_settings(on_ui_settings)


@prompt_parser_hijacker.hijack('get_learned_conditioning')
def _hijacked_get_learned_conditioning(model, prompts, total_steps, *args, original_function, **kwargs):
    if not shared.opts.prompt_fusion_enabled:
        return original_function(model, prompts, total_steps, *args, **kwargs)

    hires_steps, *_ = args if args else (None, True)
    if hires_steps is not None:
        real_total_steps = hires_steps
    else:
        real_total_steps = total_steps

    if hasattr(prompts, 'is_negative_prompt'):
        is_negative_prompt = prompts.is_negative_prompt
    else:
        is_negative_prompt = global_state.old_webui_is_negative

    empty_cond.init(model)

    tensor_builders = _parse_tensor_builders(prompts, real_total_steps)
    if hasattr(prompt_parser, 'SdConditioning'):
        empty_conditioning = prompt_parser.SdConditioning(prompts)
        empty_conditioning.clear()
    else:
        empty_conditioning = []

    flattened_prompts, consecutive_ranges = _get_flattened_prompts(tensor_builders, empty_conditioning)
    flattened_schedules = original_function(model, flattened_prompts, total_steps, *args, **kwargs)

    if isinstance(flattened_schedules[0][0].cond, dict): # sdxl
        CondWrapper = interpolation_tensor.DictCondWrapper
    else:
        CondWrapper = interpolation_tensor.TensorCondWrapper

    flattened_schedules = [
        [
            prompt_parser.ScheduledPromptConditioning(cond=CondWrapper(schedule.cond), end_at_step=schedule.end_at_step)
            for schedule in subschedules
        ]
        for subschedules in flattened_schedules
    ]

    cond_tensors = (tensor_builder.build(flattened_schedules[begin:end], empty_cond.get())
                    for begin, end, tensor_builder
                    in zip(consecutive_ranges[:-1], consecutive_ranges[1:], tensor_builders))

    schedules = [_sample_tensor_schedules(cond_tensor, real_total_steps, is_hires=hires_steps is not None)
                 for cond_tensor in cond_tensors]

    if is_negative_prompt:
        if hires_steps is not None:
            global_state.negative_schedules_hires = schedules[0]
        else:
            global_state.negative_schedules = schedules[0]

    schedules = [
        [
            prompt_parser.ScheduledPromptConditioning(cond=schedule.cond.original_cond, end_at_step=schedule.end_at_step)
            for schedule in subschedules
        ]
        for subschedules in schedules
    ]

    return schedules


@prompt_parser_hijacker.hijack('get_multicond_learned_conditioning')
def _hijacked_get_multicond_learned_conditioning(*args, original_function, **kwargs):
    res = original_function(*args, **kwargs)
    global_state.old_webui_is_negative = False
    return res


def _parse_tensor_builders(prompts, total_steps):
    tensor_builders = []

    for prompt in prompts:
        expr = prompt_fusion_parser.parse_prompt(prompt)
        tensor_builder = interpolation_tensor.InterpolationTensorBuilder()
        expr.extend_tensor(tensor_builder, (0, total_steps), total_steps, dict())
        tensor_builders.append(tensor_builder)

    return tensor_builders


def _get_flattened_prompts(tensor_builders, flattened_prompts=None):
    if flattened_prompts is None:
        flattened_prompts = []
    consecutive_ranges = [0]

    for tensor_builder in tensor_builders:
        flattened_prompts.extend(tensor_builder.get_prompt_database())
        consecutive_ranges.append(len(flattened_prompts))

    return flattened_prompts, consecutive_ranges


def _sample_tensor_schedules(tensor, steps, is_hires):
    schedules = []

    for step in range(steps):
        origin_cond = global_state.get_origin_cond_at(step, is_hires)
        params = interpolation_tensor.InterpolationParams(step / steps, step, steps, global_state.get_slerp_scale(), global_state.get_slerp_epsilon())
        schedule_cond = tensor.interpolate(params, origin_cond, empty_cond.get())
        if schedules and schedules[-1].cond == schedule_cond:
            schedules[-1] = prompt_parser.ScheduledPromptConditioning(end_at_step=step, cond=schedules[-1].cond)
        else:
            schedules.append(prompt_parser.ScheduledPromptConditioning(end_at_step=step, cond=schedule_cond))

    return schedules


class PromptFusionScript(scripts.Script):
    def title(self):
        return 'Prompt Fusion'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process(self, p, *args):
        global_state.negative_schedules = None
        global_state.old_webui_is_negative = True
