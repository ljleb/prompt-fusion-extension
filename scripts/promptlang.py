import modules.scripts as scripts
from modules.processing import create_infotext
from modules.sd_hijack import model_hijack
from modules.shared import opts

import sys
base_dir = scripts.basedir()
sys.path.append(base_dir)

from lib.prompt_transpiler import transpile_prompt as transpile


class PromptlangScript(scripts.Script):
    def __init__(self):
        self.original_prompts = []

    def title(self):
        return "Promptlang"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        pass

    def process(self, p, *args):
        self.original_prompts = [list(p.all_prompts), list(p.all_negative_prompts)]
        for prompts in [p.all_prompts, p.all_negative_prompts]:
            for i, prompt in enumerate(prompts):
                prompts[i] = transpile(prompt, p.steps)

    def postprocess(self, p, processed, *args):
        comments = {}
        if len(model_hijack.comments) > 0:
            for comment in model_hijack.comments:
                comments[comment] = 1

        def infotext(iteration=0, position_in_batch=0):
            print(position_in_batch + iteration * p.batch_size)
            return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments, iteration, position_in_batch)

        for prompts_i, prompts in enumerate([p.all_prompts, p.all_negative_prompts]):
            for i, prompt in enumerate(prompts):
                prompts[i] = self.original_prompts[prompts_i][i]

        processed.all_prompts = p.all_prompts
        processed.all_negative_prompts = p.all_negative_prompts

        for i, image in (enumerate(processed.images)):
            batch_count = len(processed.images) // p.n_iter
            unwanted_grid_because_of_img_count = len(processed.images) < 2 and opts.grid_only_if_multiple
            if i == len(processed.images) - 1 and (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count and opts.return_grid and opts.enable_pnginfo:
                text = infotext(0, 0)
            else:
                text = infotext(i // batch_count, i % batch_count)

            image.info["parameters"] = text
            processed.infotexts[i] = text

    def run(self, p, *args):
        pass
