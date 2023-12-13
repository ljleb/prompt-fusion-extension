from typing import List, Optional
from modules import shared, prompt_parser
from lib_prompt_fusion import empty_cond


old_webui_is_negative: bool = False
negative_schedules: Optional[List[prompt_parser.ScheduledPromptConditioning]] = None
negative_schedules_hires: Optional[List[prompt_parser.ScheduledPromptConditioning]] = None


def get_origin_cond_at(step: int, is_hires: bool = False):
    fallback_schedules = negative_schedules_hires if is_hires else negative_schedules
    if not fallback_schedules or not shared.opts.data.get('prompt_fusion_slerp_negative_origin', False):
        return empty_cond.get()

    for schedule in fallback_schedules:
        if schedule.end_at_step >= step:
            return schedule.cond

    return empty_cond.get()


def get_slerp_scale():
    return shared.opts.data.get('prompt_fusion_slerp_scale', 0.0)


def get_slerp_epsilon():
    return shared.opts.data.get('prompt_fusion_slerp_epsilon', 0.0001)
