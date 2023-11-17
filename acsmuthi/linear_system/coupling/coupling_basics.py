import numpy as np


def fresnel_r_hard():
    return 1


def fresnel_r(k_parallel, k_medium, k_substrate, rho_medium, rho_substrate):
    kz_medium = np.emath.sqrt(k_medium ** 2 - k_parallel ** 2)
    kz_substrate = np.emath.sqrt(k_substrate ** 2 - k_parallel ** 2)
    return ((rho_substrate * kz_medium - rho_medium * kz_substrate) /
            (rho_substrate * kz_medium + rho_medium * kz_substrate))


def reasonable_waypoints(imag_deflection, finish, problems=None):
    min_problem = 1
    max_problem = 1
    if problems is not None:
        min_problem = np.min(problems)
        max_problem = np.max(problems)

    if finish is None:
        finish = max_problem + 2,
    else:
        max_problem = 1 if finish <= min_problem + 0.1 else max_problem

    if imag_deflection is None:
        return [0., finish]
    else:
        start_deflection = min(0.9, min_problem - 0.1)
        stop_deflection = max(1.1, max_problem + 0.1)
        waypoints = [
            0.,
            start_deflection,
            start_deflection - 1j * imag_deflection,
            stop_deflection - 1j * imag_deflection,
            stop_deflection,
            finish
        ]
        return waypoints


def k_contour(imag_deflection, finish, step, k_problems=None):
    k_waypoints = reasonable_waypoints(imag_deflection, finish, problems=k_problems)

    path_pieces = []
    for i in range(len(k_waypoints) - 1):
        abs_dk = abs(k_waypoints[i + 1] - k_waypoints[i])
        if abs_dk > 0:
            num_samples = int(np.ceil(abs_dk / step)) + 1
            array = np.linspace(0, 1, num=num_samples, endpoint=True, dtype=complex)
            path_pieces.append(k_waypoints[i] + array * (k_waypoints[i + 1] - k_waypoints[i]))

    return np.concatenate(path_pieces)


# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# x = np.linspace(-0.005, 0.005, 150)
# rho1, rho2, c1, c2 = 825, 1000, 1480, 1290
# z = rho2 * c2 * np.emath.sqrt(1 - c1 ** 2 * x ** 2) + rho1 * c1 * np.emath.sqrt(1 - c2 ** 2 * x ** 2)
# plt.plot(x, np.real(z))
# plt.plot(x, np.imag(z))
# plt.show()
