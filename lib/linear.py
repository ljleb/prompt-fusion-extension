from lib.bezier import compute_on_curve_with_points as compute_bezier
import numpy
import math


def compute_linear(t, control_points):
    if len(control_points) < 3:
        return compute_bezier(t, control_points)
    else:
        target_curve = min(int(t * (len(control_points) - 1)), len(control_points) - 1)
        cp0 = control_points[target_curve]
        cp1 = control_points[target_curve + 1] if target_curve + 1 < len(control_points) else control_points[-1]
        return compute_bezier(math.fmod(t * (len(control_points) - 1), 1.), [cp0, cp1])


if __name__ == '__main__':
    import turtle as tr
    size = 50
    turtle_tool = tr.Turtle()
    turtle_tool.speed(10)
    turtle_tool.up()

    points = numpy.array([[0., 0], [1, 0], [3, 1], [4, 2]])*100

    for point in points:
        turtle_tool.goto(point)
        turtle_tool.dot(5, "red")

    for i in range(size):
        t = i/(size-1)
        point = compute_linear(t, points)
        turtle_tool.goto(point)
        turtle_tool.dot()
        print(point)

    turtle_tool.goto(100000, 100000)
    turtle_tool.dot()
    tr.done()
