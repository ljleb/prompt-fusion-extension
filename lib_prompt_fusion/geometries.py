import math
import torch
from lib_prompt_fusion import global_state


def curved_geometry(t, step, control_points):
    p0, p1 = control_points

    origin = global_state.get_origin_cond_at(step)
    p0 = p0 - origin
    p1 = p1 - origin

    p0_norm = torch.linalg.norm(p0)
    p1_norm = torch.linalg.norm(p1)

    angle = torch.sum((p0 / p0_norm) * (p1 / p1_norm))
    base = math.sin(angle)
    mid = origin
    mid += p0 * math.sin((1 - t) * angle) / base
    mid += p1 * math.sin(t * angle) / base

    lin = linear_geometry(t, step, control_points)
    return linear_geometry(global_state.get_curve_scale(), step, [lin, mid])


def linear_geometry(t, _step, control_points):
    p0, p1 = control_points
    res = p0 + (p1 - p0) * t
    return res
