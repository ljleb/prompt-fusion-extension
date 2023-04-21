import math

import torch


def curved_geometry(t, control_points):
    cp0, cp1 = control_points
    cp0_norm = torch.linalg.norm(cp0)
    cp1_norm = torch.linalg.norm(cp1)

    similarity = torch.sum((cp0 / cp0_norm) * (cp1 / cp1_norm))
    similarity = (similarity + 1) / 4

    def curve(x):
        return math.asin(2 * x - 1)

    def scaled_curve(x):
       return 1 - (curve(x) / curve(similarity) + 1) / 2

    t_cos = scaled_curve(similarity + (1 - 2 * similarity) * t)

    ncp1 = cp1 / cp1_norm * cp0_norm
    mid = cp0 + (ncp1 - cp0) * t_cos
    mid = mid / torch.linalg.norm(mid) * (cp0_norm + (cp1_norm - cp0_norm) * t)
    return mid


def linear_geometry(t, control_points):
    cp0, cp1 = control_points
    return cp0 + (cp1 - cp0) * t
