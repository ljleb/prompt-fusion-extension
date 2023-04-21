import copy
import numpy as np


def compute_on_curve_with_points(geometry):
    def inner(t, control_points):
        if len(control_points) == 1:
            return control_points[0]
        elif len(control_points) == 2:
            return geometry(t, control_points)
        copied_control_points = copy.deepcopy(control_points)
        return compute_casteljau(geometry)(t, copied_control_points, len(copied_control_points))

    return inner


def compute_casteljau(geometry):
    def inner(t, cp_list, size):
        for i in reversed(range(1, size)):
            for j in range(i):
                cp_list[j] = geometry(t, [cp_list[j], cp_list[j+1]])
        return cp_list[0]

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
