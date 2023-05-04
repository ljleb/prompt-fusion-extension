import math
import torch


def compute_linear(geometry):
    def inner(t, step, control_points):
        if len(control_points) <= 2:
            return geometry(t, step, control_points)
        else:
            target_curve = min(int(t * (len(control_points) - 1)), len(control_points) - 1)
            cp0 = control_points[target_curve]
            cp1 = control_points[target_curve + 1] if target_curve + 1 < len(control_points) else control_points[-1]
            return geometry(math.fmod(t * (len(control_points) - 1), 1.), step, [cp0, cp1])

    return inner


if __name__ == '__main__':
    import turtle as tr
    from geometries import curved_geometry
    size = 50
    turtle_tool = tr.Turtle()
    turtle_tool.speed(10)
    turtle_tool.up()

    points = torch.Tensor([[-2., 1], [1, -3], [0., 0.1]])*100

    for point in points:
        turtle_tool.goto([int(point[0]), int(point[1])])
        turtle_tool.dot(5, "red")

    turtle_tool.goto([0,0])
    turtle_tool.dot(5, "red")

    for i in range(size):
        t = i / size
        point = compute_linear(curved_geometry)(t, i, points)
        try:
            turtle_tool.goto([int(point[0]), int(point[1])])
            turtle_tool.dot()
            print(point)
        except ValueError:
            pass

    turtle_tool.goto(100000, 100000)
    turtle_tool.dot()
    tr.done()
