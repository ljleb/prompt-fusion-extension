import modules.scripts as scripts

import sys
base_dir = scripts.basedir()
sys.path.append(base_dir)

from lib.prompt_transpiler import transpile_prompt as transpile


class PromptlangScript(scripts.Script):
    def title(self):
        return "Promptlang"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        pass

    def process(self, p, *args):
        for prompts in [p.all_prompts, p.all_negative_prompts]:
            for i, prompt in enumerate(prompts):
                prompts[i] = transpile(prompt, p.steps)


    def run(self, p, *args):
        pass
