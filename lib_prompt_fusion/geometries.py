import math
import torch
from lib_prompt_fusion import interpolation_tensor


def slerp_geometry(control_points, params: interpolation_tensor.InterpolationParams):
    p0, p1 = control_points
    p0_norm = torch.linalg.norm(p0)
    p1_norm = torch.linalg.norm(p1)

    similarity = torch.sum((p0 / p0_norm) * (p1 / p1_norm))
    similarity = min(1., max(-1., float(similarity)))
    if similarity <= params.slerp_epsilon - 1 or similarity >= 1 - params.slerp_epsilon:
        return linear_geometry(control_points, params)

    angle = math.acos(float(similarity)) / 2

    slerp_t = angle * (2 * params.t - 1)
    slerp_t = math.tan(slerp_t) / math.tan(angle)
    slerp_t = (slerp_t + 1) / 2

    normalized_p1 = p1 / p1_norm * p0_norm
    slerp_p = p0 + (normalized_p1 - p0) * slerp_t
    slerp_p = slerp_p / torch.linalg.norm(slerp_p) * (p0_norm + (p1_norm - p0_norm) * params.t)

    lerp_p = linear_geometry(control_points, params)
    return lerp_p + (slerp_p - lerp_p) * params.slerp_scale


def linear_geometry(control_points, params: interpolation_tensor.InterpolationParams):
    p0, p1 = control_points
    res = p0 + (p1 - p0) * params.t
    return res
