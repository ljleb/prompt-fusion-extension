import math
import torch


def curved_geometry(t, control_points):
    cp0, cp1 = control_points
    cp0_norm = torch.linalg.norm(cp0)
    cp1_norm = torch.linalg.norm(cp1)

    similarity = torch.sum((cp0 / cp0_norm) * (cp1 / cp1_norm))

    angle = math.acos(similarity) / 2
    t_curve = angle * (2 * t - 1)
    t_curve = math.tan(t_curve) / math.tan(angle)
    t_curve = (t_curve + 1) / 2

    ncp1 = cp1 / cp1_norm * cp0_norm
    mid = cp0 + (ncp1 - cp0) * t_curve
    mid = mid / torch.linalg.norm(mid) * (cp0_norm + (cp1_norm - cp0_norm) * t)
    return mid


def linear_geometry(t, control_points):
    cp0, cp1 = control_points
    return cp0 + (cp1 - cp0) * t
