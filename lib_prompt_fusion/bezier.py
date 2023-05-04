import copy
import numpy as np
from lib_prompt_fusion import interpolation_tensor


def compute_on_curve_with_points(geometry):
    def inner(control_points, params: interpolation_tensor.InterpolationParams):
        if len(control_points) == 1:
            return control_points[0]
        elif len(control_points) == 2:
            return geometry(control_points, params)
        copied_control_points = copy.deepcopy(control_points)
        return compute_casteljau(geometry)(copied_control_points, params, len(copied_control_points))

    return inner


def compute_casteljau(geometry):
    def inner(ps, params, size):
        for i in reversed(range(1, size)):
            for j in range(i):
                ps[j] = geometry([ps[j], ps[j+1]], params)
        return ps[0]

    return inner


if __name__ == "__main__":
    import turtle as tr
    from lib_prompt_fusion.geometries import linear_geometry
    size = 30
    turtle_tool = tr.Turtle()
    turtle_tool.speed(10)
    turtle_tool.up()

    points = np.array([[0., 0], [2, -3], [-2, 4]])*100

    for point in points:
        turtle_tool.goto(point)
        turtle_tool.dot(5, "red")

    for i in range(size):
        t = i/size
        point = compute_on_curve_with_points(linear_geometry)(t, points)
        turtle_tool.goto(point)
        turtle_tool.dot()
        print(point)

    turtle_tool.goto(100000, 100000)
    turtle_tool.dot()
    tr.done()
