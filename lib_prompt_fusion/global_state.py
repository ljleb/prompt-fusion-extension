from modules import shared
from torch import Tensor
from typing import Optional


def get_origin_cond_at(step: int, empty_cond: Tensor, negative_schedule: Optional[Tensor]):
    if negative_schedule is None or not shared.opts.data.get('prompt_fusion_slerp_negative_origin', False):
        return empty_cond

    return negative_schedule[step]


def get_slerp_scale():
    return shared.opts.data.get('prompt_fusion_slerp_scale', 1)


def get_slerp_epsilon():
    return shared.opts.data.get('prompt_fusion_slerp_epsilon', 0.0001)
