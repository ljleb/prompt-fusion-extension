import modules.scripts as scripts
import sys
base_dir = scripts.basedir()
sys.path.append(base_dir)

from lib.dsl_prompt_transpiler import parse_prompt
from lib.hijacker import prompt_parser_hijacker
import numpy

@prompt_parser_hijacker.hijack('get_learned_conditioning')
def hijacked_get_learned_conditioning(model, prompts, steps, original_function):
    scheduled_conditionings = []

    for prompt in prompts:
        expr = parse_prompt(prompt)
        tensor = [0]
        prompt_database = ['']
        interpolation_functions = []

        tensor = expr.append_to_tensor(tensor, prompt_database, interpolation_functions, (0, steps), dict())
        print(tensor)
        print(prompt_database)
        print(interpolation_functions)
        #scheduled_conditionings.append(conditioning.to_scheduled_conditionings(steps))

    return scheduled_conditionings


class FusionScript(scripts.Script):
    def title(self):
        return "fusion"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def run(self, p, *args):
        pass
