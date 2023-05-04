import copy
import math
from lib_prompt_fusion import interpolation_tensor, geometries


def compute_linear(control_points, params: interpolation_tensor.InterpolationParams):
    if len(control_points) <= 2:
        return geometries.slerp_geometry(control_points, params)
    else:
        target_curve = min(int(params.t * (len(control_points) - 1)), len(control_points) - 1)
        cp0 = control_points[target_curve]
        cp1 = control_points[target_curve + 1] if target_curve + 1 < len(control_points) else control_points[-1]
        new_params = interpolation_tensor.InterpolationParams(math.fmod(params.t * (len(control_points) - 1), 1.), *params[1:])
        return geometries.slerp_geometry([cp0, cp1], new_params)


def compute_bezier(control_points, params: interpolation_tensor.InterpolationParams):
    def compute_casteljau(ps, size):
        for i in reversed(range(1, size)):
            for j in range(i):
                ps[j] = geometries.slerp_geometry([ps[j], ps[j+1]])

        return ps[0]

    if len(control_points) == 1:
        return control_points[0]
    elif len(control_points) == 2:
        return geometries.slerp_geometry(control_points, params)
    copied_control_points = copy.deepcopy(control_points)
    return compute_casteljau(copied_control_points, len(copied_control_points))


def compute_catmull(control_points, params: interpolation_tensor.InterpolationParams):
    if len(control_points) <= 2:
        return compute_linear(control_points, params)
    else:
        target_curve = min(int(params.t * (len(control_points) - 1)), len(control_points) - 1)
        g0 = control_points[target_curve - 1] if target_curve > 0 else 2 * control_points[0] - control_points[1]
        cp0 = control_points[target_curve]
        cp1 = control_points[target_curve + 1] if target_curve + 1 < len(control_points) else control_points[-1]
        g1 = control_points[target_curve + 2] if target_curve + 2 < len(control_points) else 2 * cp1 - cp0
        ip0 = cp0 + (cp1 - g0)/6
        ip1 = cp1 + (cp0 - g1)/6

        new_params = interpolation_tensor.InterpolationParams(math.fmod(params.t * (len(control_points) - 1), 1.), *params[1:])
        return compute_bezier([cp0, ip0, ip1, cp1], new_params)


if __name__ == '__main__':
    import turtle as tr
    import torch
    size = 90
    turtle_tool = tr.Turtle()
    turtle_tool.speed(10)
    turtle_tool.up()

    points = torch.Tensor([[-2., 1.], [4., -2.], [0., -2.]]) * 100.

    def sample(slerp_scale, color):
        for i in range(size):
            t = i / size
            params = interpolation_tensor.InterpolationParams(t, i, slerp_scale)
            point = compute_linear(points, params)
            try:
                turtle_tool.goto((float(point[0]), float(point[1])))
                turtle_tool.dot(5, color)
                print(point)
            except ValueError:
                pass

    sample(0, "black")
    sample(1, "green")
    sample(2, "blue")
    sample(-1, "purple")
    sample(-2, "orange")

    for point in points:
        turtle_tool.goto((float(point[0]), float(point[1])))
        turtle_tool.dot(5, "red")

    turtle_tool.goto((0, 0))
    turtle_tool.dot(10, "red")

    turtle_tool.goto(100000, 100000)
    turtle_tool.dot()
    tr.done()
