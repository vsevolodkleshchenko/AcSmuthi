import numpy as np
import scipy.special as ss


def dec_to_cyl(x, y, z):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    if phi.shape == ():
        if phi < 0:
            phi += 2 * np.pi
    else:
        phi[phi < 0] += 2 * np.pi
    return rho, phi, z


def legendre_normalized(m, n, x):
    coefficient = np.sqrt((2 * n + 1) / 4 / np.pi * ss.factorial(n - m) / ss.factorial(n + m))
    return coefficient * ss.clpmn(m, n, x, type=2)[0][-1][-1]


def fresnel_r(k_rho):
    return 1


def k_contour_old(k_start_deflection, k_stop_deflection, dk_imag_deflection, k_finish, dk):
    if k_start_deflection is None:
        return np.arange(0, k_finish, dk)

    path_pieces = []

    if k_start_deflection != 0:
        start_path = np.arange(0, k_start_deflection, dk)
    else:
        start_path = 0 - 1j * np.arange(0, dk_imag_deflection, dk)
    path_pieces.append(start_path)

    if k_stop_deflection is not None:
        deflected_path = np.arange(k_start_deflection, k_stop_deflection, dk) - 1j * dk_imag_deflection
        deflection_stop_path = k_stop_deflection + 1j * np.arange(-dk_imag_deflection, 0, dk)
        finish_path = np.arange(k_stop_deflection, k_finish, dk)
        path_pieces.extend([deflected_path, deflection_stop_path, finish_path])
    else:
        deflected_path = np.arange(k_start_deflection, k_finish, dk) - 1j * dk_imag_deflection
        path_pieces.append(deflected_path)

    return np.concatenate(path_pieces)


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
