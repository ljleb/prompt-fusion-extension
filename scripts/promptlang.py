import modules.scripts as scripts
import sys
base_dir = scripts.basedir()
sys.path.append(base_dir)

from lib_prompt_fusion import interpolation_tensor, prompt_parser as prompt_fusion_parser, hijacker, empty_cond, global_state
import importlib
importlib.reload(hijacker)
importlib.reload(empty_cond)
importlib.reload(global_state)
importlib.reload(interpolation_tensor)
importlib.reload(prompt_fusion_parser)
from modules import prompt_parser, script_callbacks, shared
import torch
import gradio as gr


fusion_hijacker_attribute = '__fusion_hijacker'
prompt_parser_hijacker = hijacker.ModuleHijacker.install_or_get(
    module=prompt_parser,
    hijacker_attribute=fusion_hijacker_attribute,
    register_uninstall=script_callbacks.on_script_unloaded)


def on_ui_settings():
    section = ('prompt-fusion', 'Prompt Fusion')
    shared.opts.add_option('prompt_fusion_enabled', shared.OptionInfo(True, 'Enabled', section=section))
    shared.opts.add_option('prompt_fusion_curve_relative_negative', shared.OptionInfo(False, 'Rotate around the negative prompt', section=section))
    shared.opts.add_option('prompt_fusion_curve_scale', shared.OptionInfo(0, 'Interpolate in spherical geometry (0 = do not rotate, 1 = rotate to scale)', component=gr.Number, section=section))


script_callbacks.on_ui_settings(on_ui_settings)


@prompt_parser_hijacker.hijack('get_learned_conditioning')
def _hijacked_get_learned_conditioning(model, prompts, total_steps, original_function):
    if not shared.opts.prompt_fusion_enabled:
        return original_function(model, prompts, total_steps)

    empty_cond.init(model)

    tensor_builders = _parse_tensor_builders(prompts, total_steps)
    flattened_prompts, consecutive_ranges = _get_flattened_prompts(tensor_builders)
    flattened_conds = original_function(model, flattened_prompts, total_steps)

    cond_tensors = [tensor_builder.build(flattened_conds[begin:end])
                    for begin, end, tensor_builder
                    in zip(consecutive_ranges[:-1], consecutive_ranges[1:], tensor_builders)]

    schedules = [_sample_tensor_schedules(cond_tensor, total_steps)
              for cond_tensor in cond_tensors]

    if global_state.is_negative:
        global_state.is_negative = False
        global_state.negative_schedules = schedules[0]

    return schedules


def _parse_tensor_builders(prompts, total_steps):
    tensor_builders = []

    for prompt in prompts:
        expr = prompt_fusion_parser.parse_prompt(prompt)
        tensor_builder = interpolation_tensor.InterpolationTensorBuilder()
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


def _sample_tensor_schedules(tensor, steps):
    schedules = []

    for step in range(steps):
        origin_cond = global_state.get_origin_cond_at(step)
        params = interpolation_tensor.InterpolationParams(step / steps, step, global_state.get_curve_scale())
        schedule_cond = tensor.interpolate(params, origin_cond)
        if schedules and torch.all(torch.eq(schedules[-1].cond, schedule_cond)):
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
        global_state.is_negative = True
        global_state.negative_schedules = None


if __name__ == '__main__':
    import turtle as tr
    from lib_prompt_fusion.geometries import slerp_geometry, linear_geometry
    from lib_prompt_fusion.linear import compute_linear
    from lib_prompt_fusion.bezier import compute_on_curve_with_points
    size = 30
    turtle_tool = tr.Turtle()
    turtle_tool.speed(10)
    turtle_tool.up()

    points = torch.Tensor([[-3, 2], [1, 1]]) * 100

    for point in points:
        turtle_tool.goto([int(point[0]), int(point[1])])
        turtle_tool.dot(5, "red")

    turtle_tool.goto([0,0])
    turtle_tool.dot(5, "red")

    for i in range(size):
        t = i / size
        point = compute_linear(slerp_geometry)(t, points)
        try:
            turtle_tool.goto([int(point[0]), int(point[1])])
            turtle_tool.dot()
            print(point)
        except ValueError:
            pass

    turtle_tool.goto(100000, 100000)
    turtle_tool.dot()
    tr.done()
