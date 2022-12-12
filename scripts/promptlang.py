import modules.scripts as scripts

import sys
base_dir = scripts.basedir()
sys.path.append(base_dir)

from lib.prompt_transpiler import transpile_prompt as transpile


class PromptlangScript(scripts.Script):
    def __init__(self):
        self.prompts = {}

    def title(self):
        return "Promptlang"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process_batch(self, p, prompts, **kwargs):
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
