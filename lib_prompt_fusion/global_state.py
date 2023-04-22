from typing import List, Optional
from modules import shared, prompt_parser
from lib_prompt_fusion import empty_cond


is_negative: bool = False
negative_schedules: Optional[List[prompt_parser.ScheduledPromptConditioning]] = None


def get_origin_cond_at(step: int):
    if not negative_schedules or not shared.opts.data.get('prompt_fusion_curve_relative_negative', False):
        return empty_cond.get()

    for schedule in negative_schedules:
        if schedule.end_at_step >= step:
            return schedule.cond

    return empty_cond.get()


def get_curve_scale():
    return shared.opts.data.get('prompt_fusion_curve_scale', 0)
