from lib_prompt_fusion.bezier import compute_on_curve_with_points as compute_bezier
from lib_prompt_fusion.linear import compute_linear
import numpy
import math


def compute_catmull(geometry):
    def inner(t, control_points):
        if len(control_points) <= 2:
            return compute_linear(geometry)(t, control_points)
        else:
            target_curve = min(int(t * (len(control_points) - 1)), len(control_points) - 1)
            g0 = control_points[target_curve - 1] if target_curve > 0 else 2 * control_points[0] - control_points[1]
            cp0 = control_points[target_curve]
            cp1 = control_points[target_curve + 1] if target_curve + 1 < len(control_points) else control_points[-1]
            g1 = control_points[target_curve + 2] if target_curve + 2 < len(control_points) else 2 * cp1 - cp0
            ip0 = cp0 + (cp1 - g0)/6
            ip1 = cp1 + (cp0 - g1)/6
            return compute_bezier(geometry)(math.fmod(t * (len(control_points) - 1), 1.), [cp0, ip0, ip1, cp1])

    return inner


if __name__ == '__main__':
    import turtle as tr
    from lib_prompt_fusion.geometries import linear_geometry
    size = 50
    turtle_tool = tr.Turtle()
    turtle_tool.speed(10)
    turtle_tool.up()

    points = numpy.array([[0., 0], [1, 0], [3, 1], [4, 2]])*100

    for point in points:
        turtle_tool.goto(point)
        turtle_tool.dot(5, "red")

    for i in range(size):
        t = apply_sampled_range(i/size, [0, 5, 12, 30])
        point = compute_catmull(linear_geometry)(t, points)
        turtle_tool.goto(point)
        turtle_tool.dot()
        print(point)

    turtle_tool.goto(100000, 100000)
    turtle_tool.dot()
    tr.done()
