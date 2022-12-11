import modules.scripts as scripts
from modules.processing import create_infotext
from modules.sd_hijack import model_hijack

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

    def postprocess(self, p, processed, *args):
        comments = {}
        if len(model_hijack.comments) > 0:
            for comment in model_hijack.comments:
                comments[comment] = 1

        def infotext(iteration=0, position_in_batch=0):
            return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments, iteration, position_in_batch)

        for i, image in enumerate(processed.images):
            batch_count = len(processed.images) / p.n_iter
            print((i % batch_count, i // batch_count))
            text = infotext(i % batch_count, i // batch_count)
            image.info["parameters"] = text
            processed.infotexts[i] = text

    def run(self, p, *args):
        pass
