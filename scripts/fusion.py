import modules.scripts as scripts
from modules import prompt_parser
import re

import sys
base_dir = scripts.basedir()
sys.path.append(base_dir)

from lib.prompt_transpiler import transpile_prompt
from lib.catmull import compute_catmull
from lib.bezier import compute_on_curve_with_points as compute_bezier
from lib.linear import compute_linear
from lib.t_scaler import apply_sampled_range
from collections import namedtuple


ClandestineCompanionObject = namedtuple('ClandestineCompanionObject', ['original_functions'])


def install_or_get_companion_object():
    clandestine_attribute = '__fusion'
    if not hasattr(prompt_parser, clandestine_attribute):
        clandestine_companion_object = ClandestineCompanionObject({
            'get_learned_conditioning': prompt_parser.get_learned_conditioning,
        })
        setattr(prompt_parser, clandestine_attribute, clandestine_companion_object)
        return clandestine_companion_object
    else:
        return getattr(prompt_parser, clandestine_attribute)


clandestine_companion_object = install_or_get_companion_object()
original_functions = clandestine_companion_object.original_functions


re_INTERPOLATE_TYPE = re.compile(r'\bINTERPOLATE(?:\((bezier|catmull|linear)\))?')
re_INTERPOLATE_SPLIT = re.compile(r'\bINTERPOLATE(?:\((?:bezier|catmull|linear)\))?')

re_INTERPOLATION_STEPS = re.compile(r'\bINTERPOLATION_STEPS\[(\s*(?:(?:\d+)?\.\d+|\d+\.?)\s*(?:,\s*(?:(?:\d+)?\.\d+|\d+\.?)\s*)*)]')
re_INTERPOLATION_STEP_VALUES = re.compile(r'\s*((?:\d+)?\.\d+|\d+\.?)\s*,?')


def hijacked_get_learned_conditioning(model, prompts, steps):
    res = []

    for prompt in prompts:
        interpolation_control_points = find_interpolation_control_points(prompt, steps)
        prompt = re_INTERPOLATION_STEPS.sub('', prompt)
        subconditionings = get_conditionings(model, prompt, steps)
        curve_function = get_curve_function(prompt)

        cond_array = []
        for i in range(steps):
            t = apply_sampled_range(i/max(1, steps-1), interpolation_control_points)
            cond_array.append(prompt_parser.ScheduledPromptConditioning(end_at_step=i, cond=curve_function(t, subconditionings)))
        res.append(cond_array)

    return res


prompt_parser.get_learned_conditioning = hijacked_get_learned_conditioning


def find_interpolation_control_points(prompt, steps):
    match = re_INTERPOLATION_STEPS.search(prompt)
    interpolation_control_points = []
    if match is not None:
        inter_steps_params = match.group(1)
        inter_params_matches = re_INTERPOLATION_STEP_VALUES.finditer(inter_steps_params)
        for match in inter_params_matches:
            interpolation_control_points.append(float(match.group(1)) / steps)
    else:
        interpolation_control_points.append(0.)
        interpolation_control_points.append(1.)

    return interpolation_control_points


def get_conditionings(model, prompt, steps):
    control_points = []
    subprompts = re_INTERPOLATE_SPLIT.split(prompt)
    for subprompt in subprompts:
        subprompt = transpile_prompt(subprompt, steps)
        subconditioning = original_functions['get_learned_conditioning'](model, [subprompt], steps)[0][0].cond
        control_points.append(subconditioning)
    return control_points


def get_curve_function(prompt):
    match = re_INTERPOLATE_TYPE.search(prompt)
    return {
        'catmull': compute_catmull,
        'linear': compute_linear,
        'bezier': compute_bezier,
    }[match.group(1) if match is not None else 'linear']


class FusionScript(scripts.Script):
    def title(self):
        return "fusion"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def run(self, p, *args):
        pass
