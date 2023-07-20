import numpy as np
import scipy.special as ss
import matplotlib.pyplot as plt


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
    return coefficient * ss.clpmn(m, n, x, type=3)[0][-1][-1]


def fresnel_r(k_rho):
    return 1


def k_contour(
        k_start_deflection=0.5,
        k_stop_deflection=1.5,
        dk_imag_deflection=0.1,
        k_finish=5,
        dk=0.01
):
    if k_start_deflection is None:
        return np.arange(0, k_finish, dk)

    path_pieces = []

    if k_start_deflection != 0:
        start_path = np.arange(0, k_start_deflection, dk) + 0j
    else:
        start_path = 0 - 1j * np.arange(0, dk_imag_deflection, dk)
    path_pieces.append(start_path)

    if k_stop_deflection is not None:
        deflected_path = np.arange(k_start_deflection, k_stop_deflection, dk) - 1j * dk_imag_deflection
        deflection_stop_path = k_stop_deflection + 1j * np.arange(-dk_imag_deflection, 0, dk)
        finish_path = np.arange(k_stop_deflection, k_finish, dk) + 0j
        path_pieces.extend([deflected_path, deflection_stop_path, finish_path])
    else:
        deflected_path = np.arange(k_start_deflection, k_finish, dk) - 1j * dk_imag_deflection
        path_pieces.append(deflected_path)

    return np.concatenate(path_pieces)
