import numpy as np


def fresnel_r(k_parallel):
    return 1


def k_contour(k_start_deflection, k_stop_deflection, dk_imag_deflection, k_finish, dk):
    if k_start_deflection is None:
        k_waypoints = [0., k_finish]
    else:
        k_waypoints = [
            0.,
            k_start_deflection,
            k_start_deflection - 1j * dk_imag_deflection,
            k_stop_deflection - 1j * dk_imag_deflection,
            k_stop_deflection,
            k_finish
        ]

    path_pieces = []
    for i in range(len(k_waypoints) - 1):
        abs_dk = abs(k_waypoints[i + 1] - k_waypoints[i])
        if abs_dk > 0:
            num_samples = int(np.ceil(abs_dk / dk)) + 1
            array = np.linspace(0, 1, num=num_samples, endpoint=True, dtype=complex)
            path_pieces.append(k_waypoints[i] + array * (k_waypoints[i + 1] - k_waypoints[i]))

    return np.concatenate(path_pieces)