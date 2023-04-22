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
    angle = math.acos(angle) / 2
    if angle == 0:
        return control_points[0]

    t_curve = angle * (2 * t - 1)
    t_curve = math.tan(t_curve) / math.tan(angle)
    t_curve = (t_curve + 1) / 2

    p1 = p1 / p1_norm * p0_norm
    mid = p0 + (p1 - p0) * t_curve
    mid = mid / torch.linalg.norm(mid) * (p0_norm + (p1_norm - p0_norm) * t)
    mid += origin

    lin = linear_geometry(t, step, control_points)
    return linear_geometry(global_state.get_curve_scale(), step, [lin, mid])


def linear_geometry(t, _step, control_points):
    p0, p1 = control_points
    res = p0 + (p1 - p0) * t
    return res
