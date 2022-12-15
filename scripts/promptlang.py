import modules.scripts as scripts
from modules import prompt_parser
import re

import sys
base_dir = scripts.basedir()
sys.path.append(base_dir)

from lib.prompt_transpiler import transpile_prompt as transpile
from lib.catmull import compute_catmull
from lib.bezier import compute_on_curve_with_points as compute_bezier


saved_get_learned_conditioning = prompt_parser.get_learned_conditioning


re_INTERPOLATE_TYPE = re.compile(r'\bINTERPOLATE(?:\((bezier|catmull)\))?')
re_INTERPOLATE_SPLIT = re.compile(r'\bINTERPOLATE(?:\((?:bezier|catmull)\))?')


def hijacked_get_learned_conditioning(model, prompts, steps):
    res = []

    for prompt in prompts:
        control_points = []
        match = re_INTERPOLATE_TYPE.search(prompt)
        interpolation_function = compute_catmull
        if match is not None and match.group(1) == 'bezier':
            interpolation_function = compute_bezier
        subprompts = re_INTERPOLATE_SPLIT.split(prompt)
        for subprompt in subprompts:
            subconditioning = saved_get_learned_conditioning(model, [subprompt], steps)[0][0].cond
            control_points.append(subconditioning)

        cond_array = []
        if steps > 1:
            for i in range(steps):
                cond_array.append(prompt_parser.ScheduledPromptConditioning(end_at_step=i, cond=interpolation_function(i/(steps-1), control_points)))
        else:
            cond_array.append(prompt_parser.ScheduledPromptConditioning(end_at_step=steps, cond=interpolation_function(0.5, control_points)))
        res.append(cond_array)

    return res


prompt_parser.get_learned_conditioning = hijacked_get_learned_conditioning


class PromptlangScript(scripts.Script):
    def __init__(self):
        self.prompts = {}

    def title(self):
        return "Promptlang"

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
