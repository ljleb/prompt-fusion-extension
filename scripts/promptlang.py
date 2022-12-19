import modules.scripts as scripts
from modules import prompt_parser

import sys
base_dir = scripts.basedir()
sys.path.append(base_dir)

from lib.dsl_prompt_transpiler import parse_prompt


class ClandestineCompanionObject:
    def __init__(self, original_functions):
        self.original_functions = original_functions

    def hijack(self, module, attribute):
        assert attribute in self.original_functions, 'function is not backed up by this clandestine object'

        def decorator(function):
            def wrapper(*args, **kwargs):
                return function(*args, **kwargs, original_function=self.original_functions[attribute])

            setattr(module, attribute, wrapper)
            return function

        return decorator


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


@clandestine_companion_object.hijack(prompt_parser, 'get_learned_conditioning')
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

if __name__ == '__main__':
    pass
