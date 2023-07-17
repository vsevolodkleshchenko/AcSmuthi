import numpy as np
import scipy.integrate as si
import scipy.special as ss
from acsmuthi.utility import wavefunctions as wvfs


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
    path_pieces = []

    if k_start_deflection != 0:
        start_path = np.arange(0, k_start_deflection, dk) + 0j
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


def reflection_element(m, n, mu, nu, k, emitter_pos, receiver_pos, k_parallel=k_contour()):
    dist = emitter_pos - receiver_pos
    d_rho, d_phi, dz = np.sqrt(dist[0] ** 2 + dist[1] ** 2), np.arctan2(dist[0], dist[1]), dist[2]

    def f_integrand(k_rho):
        gamma = np.emath.sqrt(k ** 2 - k_rho ** 2)
        return fresnel_r(k_rho) * np.exp(2j * gamma * emitter_pos[2]) * k_rho / gamma * \
            np.conj(legendre_normalized(m, n, gamma / k)) * legendre_normalized(mu, nu, - gamma / k) * \
            np.exp(1j * gamma * dz) * ss.jn(np.abs(m - mu), k_rho * d_rho)

    integrand = [f_integrand(ki) for ki in k_parallel]
    integral = si.trapz(integrand, k_parallel)

    coef = 4 * np.pi * (-1j) ** n * 1j ** (nu + np.abs(m - mu))
    return coef * np.exp(1j * (m - mu) * d_phi) / k * integral


def reflection_block(emitter_pos, receiver_pos, k, order):
    block = np.zeros(((order + 1) ** 2, (order + 1) ** 2), dtype=complex)
    for m, n in wvfs.multipoles(order):
        imn = n ** 2 + n + m
        for mu, nu in wvfs.multipoles(order):
            if mu != m:
                continue
            imunu = nu ** 2 + nu + mu
            block[imn, imunu], _ = reflection_element(m, n, mu, nu, k, emitter_pos, receiver_pos)
    return block


def test_integrator():
    m, n, mu, nu = 0, 1, 0, 1
    k, pos1, pos2 = 1.1, np.array([-1, 1, 2]), np.array([3, 4, 3])

    k_waypoint = np.linspace(k-0.5, k-0.01, 40)
    els = []
    for k_tested in k_waypoint:
        k_parallel = k_contour(
            k_start_deflection=k_tested,
            k_stop_deflection=k + 0.1,
            dk_imag_deflection=0.01,
            k_finish=6,
            dk=0.01
        )
        els.append(reflection_element(m, n, mu, nu, k, pos1, pos2, k_parallel))

    import matplotlib.pyplot as plt

    # plt.plot(k_waypoint, np.real(els))
    plt.plot(k_waypoint, np.imag(els))
    plt.show()


# test_integrator()
