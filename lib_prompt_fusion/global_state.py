from typing import Sequence, Optional
from modules import shared
from torch import Tensor


old_webui_is_negative: bool = False
negative_schedules: Optional[Sequence[Sequence[Tensor]]] = None


def get_origin_cond_at(step: int, pass_index: int, empty_cond: Tensor):
    if negative_schedules is None or pass_index >= len(negative_schedules) or not shared.opts.data.get('prompt_fusion_slerp_negative_origin', False):
        return empty_cond

    return negative_schedules[pass_index][step]


def get_slerp_scale():
    return shared.opts.data.get('prompt_fusion_slerp_scale', 1)


def get_slerp_epsilon():
    return shared.opts.data.get('prompt_fusion_slerp_epsilon', 0.0001)
