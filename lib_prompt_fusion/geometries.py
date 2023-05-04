import math
import torch
from lib_prompt_fusion import interpolation_tensor


def slerp_geometry(control_points, params: interpolation_tensor.InterpolationParams):
    p0, p1 = control_points
    p0_norm = torch.linalg.norm(p0)
    p1_norm = torch.linalg.norm(p1)

    similarity = torch.sum((p0 / p0_norm) * (p1 / p1_norm))
    similarity = min(1., max(-1., similarity))

    angle = math.acos(similarity) / 2
    if angle == 0:
        return control_points[0]

    t_curve = angle * (2 * params.t - 1)
    t_curve = math.tan(t_curve) / math.tan(angle)
    t_curve = (t_curve + 1) / 2

    np1 = p1 / p1_norm * p0_norm
    mid = p0 + (np1 - p0) * t_curve
    mid = mid / torch.linalg.norm(mid) * (p0_norm + (p1_norm - p0_norm) * params.t)

    lin = linear_geometry(control_points, params)
    new_params = interpolation_tensor.InterpolationParams(params.slerp_scale, *params[1:])
    return linear_geometry([lin, mid], new_params)


def linear_geometry(control_points, params: interpolation_tensor.InterpolationParams):
    p0, p1 = control_points
    res = p0 + (p1 - p0) * params.t
    return res
