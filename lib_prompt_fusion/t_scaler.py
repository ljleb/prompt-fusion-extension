def scale_t(t, positions):
    if t >= 1.:
        return 1.

    if t <= 0.:
        return 0.

    distances = []
    for i in range(len(positions)-1):
        distances.append(positions[i+1] - positions[i])

    total_distance = sum(distances)
    for i in range(len(distances)):
        distances[i] = distances[i]/total_distance

    for i in range(len(distances)-1):
        distances[i+1] = distances[i] + distances[i+1]

    distances.insert(0, 0.0)

    spline_index = 0
    for i, distance in enumerate(distances):
        if t > distance:
            spline_index = i
        else:
            break

    if spline_index >= len(distances) - 1:
        return 1

    local_ratio = (t - distances[spline_index]) / (distances[spline_index+1] - distances[spline_index])
    return (spline_index + local_ratio)/(len(distances)-1)


if __name__ == "__main__":
    total_steps = 20
    for i in range(total_steps):
        print(i, scale_t(i/total_steps, [9, 10]))
