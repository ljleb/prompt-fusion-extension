import modules.scripts as scripts
from modules import prompt_parser
import re

import sys
base_dir = scripts.basedir()
sys.path.append(base_dir)

from lib.prompt_transpiler import transpile_prompt as transpile
from lib.catmull import compute_catmull
from lib.bezier import compute_on_curve_with_points as compute_bezier
from lib.linear import compute_linear
from lib.t_scaler import apply_sampled_range


saved_get_learned_conditioning = prompt_parser.get_learned_conditioning


re_INTERPOLATE_TYPE = re.compile(r'\bINTERPOLATE(?:\((bezier|catmull|linear)\))?')
re_INTERPOLATE_SPLIT = re.compile(r'\bINTERPOLATE(?:\((?:bezier|catmull|linear)\))?')

re_INTERPOLATION_STEPS = re.compile(r'\bINTERPOLATION_STEPS\[(\s*\d+\s*(?:,\s*\d+\s*)*)]')
re_INTERPOLATION_STEP_VALUES = re.compile(r'\s*(\d+)\s*,?')


def hijacked_get_learned_conditioning(model, prompts, steps):
    res = []

    for prompt in prompts:
        match = re_INTERPOLATION_STEPS.search(prompt)
        prompt = re_INTERPOLATION_STEPS.sub("", prompt)
        interpolation_control_points = []
        if match is not None:
            inter_steps_params = match.group(1)
            inter_params_matches = re_INTERPOLATION_STEP_VALUES.finditer(inter_steps_params)
            for match in inter_params_matches:
                interpolation_control_points.append(float(match.group(1))/steps)
        else:
            interpolation_control_points.append(0.)
            interpolation_control_points.append(1.)


        control_points = []

        match = re_INTERPOLATE_TYPE.search(prompt)
        curve_type = 'catmull'
        if match is not None:
            curve_type = match.group(1)
        subprompts = re_INTERPOLATE_SPLIT.split(prompt)
        for subprompt in subprompts:
            subconditioning = saved_get_learned_conditioning(model, [subprompt], steps)[0][0].cond
            control_points.append(subconditioning)

        cond_array = []
        for i in range(steps):
            t = apply_sampled_range(i/max(1, steps-1), interpolation_control_points)
            if curve_type == 'catmull':
                cond_array.append(prompt_parser.ScheduledPromptConditioning(end_at_step=i, cond=compute_catmull(t, control_points)))
            elif curve_type == 'linear':
                cond_array.append(prompt_parser.ScheduledPromptConditioning(end_at_step=i, cond=compute_linear(t, control_points)))
            else:
                cond_array.append(prompt_parser.ScheduledPromptConditioning(end_at_step=i, cond=compute_bezier(t, control_points)))
        res.append(cond_array)

    return res


prompt_parser.get_learned_conditioning = hijacked_get_learned_conditioning


class PromptlangScript(scripts.Script):
    def __init__(self):
        self.prompts = {}

    def title(self):
        return "fusion"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process(self, p, *args):
        for prompts in [p.all_prompts, p.all_negative_prompts]:
            for i, prompt in enumerate(prompts):
                transpiled_prompt = transpile(prompt, p.steps)
                self.prompts[transpiled_prompt] = prompt
                prompts[i] = transpiled_prompt

    def postprocess_batch(self, p, **kwargs):
        for prompts in [p.all_prompts, p.all_negative_prompts]:
            for i, prompt in enumerate(prompts):
                if prompt in self.prompts:
                    prompts[i] = self.prompts[prompt]

    def postprocess(self, *args, **kwargs):
        self.prompts = {}
