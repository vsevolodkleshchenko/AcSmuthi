import numpy as np


def reasonable_waypoints(imag_deflection, finish=None, problems=None, offset=2):
    if problems is None:
        problems = [1.]
    min_problem = min(1., min(np.real(problems)))
    max_problem = max(1., max(np.real(problems)))

    if finish is None:
        finish = max_problem + offset

    if imag_deflection is None:
        return [0., finish]

    else:
        start_deflection = max(0., min_problem - 0.1)
        stop_deflection = max_problem + 0.2

        waypoints = [
            0.,
            start_deflection,
            start_deflection - 1j * imag_deflection,
            stop_deflection - 1j * imag_deflection,
            stop_deflection]

        if finish > stop_deflection:
            waypoints.append(finish)

        return waypoints


def k_contour(imag_deflection, step, finish=None, problems=None):
    k_waypoints = reasonable_waypoints(imag_deflection, finish=finish, problems=problems)

    path_pieces = []
    for i in range(len(k_waypoints) - 1):
        abs_dk = abs(k_waypoints[i + 1] - k_waypoints[i])
        if abs_dk > 0:
            num_samples = int(np.ceil(abs_dk / step)) + 1
            array = np.linspace(0, 1, num=num_samples, endpoint=True, dtype=complex)
            path_pieces.append(k_waypoints[i] + array * (k_waypoints[i + 1] - k_waypoints[i]))

    return np.concatenate(path_pieces)
