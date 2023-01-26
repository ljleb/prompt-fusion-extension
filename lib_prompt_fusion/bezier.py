import copy
import numpy as np


def linear_interpolation(t, control_points):
    return control_points[0] + (control_points[1] - control_points[0]) * t


def compute_on_curve_with_points(t, control_points):
    if len(control_points) == 1:
        return control_points[0]
    elif len(control_points) == 2:
        return linear_interpolation(t, control_points)
    copied_control_points = copy.deepcopy(control_points)
    return compute_casteljau(t, copied_control_points, len(copied_control_points))


def compute_casteljau(t, cp_list, size):
    for i in reversed(range(1, size)):
        for j in range(i):
            cp_list[j] = cp_list[j] + t*(cp_list[j+1] - cp_list[j])
    return cp_list[0]


if __name__ == "__main__":
    import turtle as tr
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
        point = compute_on_curve_with_points(t, points)
        turtle_tool.goto(point)
        turtle_tool.dot()
        print(point)

    turtle_tool.goto(100000, 100000)
    turtle_tool.dot()
    tr.done()
