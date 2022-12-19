import modules.scripts as scripts
import sys
base_dir = scripts.basedir()
sys.path.append(base_dir)

from lib.dsl_prompt_transpiler import parse_prompt
from lib.hijacker import prompt_parser_hijacker


@prompt_parser_hijacker.hijack('get_learned_conditioning')
def hijacked_get_learned_conditioning(model, prompts, steps, original_function):
    scheduled_conditionings = []

    for prompt in prompts:
        expr = parse_prompt(prompt)
        conditioning = expr.get_interpolation_conditioning(model, original_function, (0, steps))
        scheduled_conditionings.append(conditioning.to_scheduled_conditionings(steps))

    return scheduled_conditionings


class FusionScript(scripts.Script):
    def title(self):
        return "fusion"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def run(self, p, *args):
        pass
