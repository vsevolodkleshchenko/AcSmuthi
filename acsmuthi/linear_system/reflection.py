import numpy as np
import scipy.integrate as si
import scipy.special as ss
from acsmuthi.utility import wavefunctions as wvfs


def legendre_normalized(m, n, x):
    coefficient = np.sqrt((2 * n + 1) / 4 / np.pi * ss.factorial(n - m) / ss.factorial(n + m))
    return coefficient * ss.clpmn(m, n, x, type=3)[0][-1][-1]


def fresnel_r(k_rho):
    return 1


def reflection_element(m, n, mu, nu, k, emitter_pos, receiver_pos):
    dist = emitter_pos - receiver_pos
    d_rho, d_phi, dz = np.sqrt(dist[0] ** 2 + dist[1] ** 2), np.arctan2(dist[0], dist[1]), dist[2]

    def integrand(k_rho):
        gamma = np.emath.sqrt(k ** 2 - k_rho ** 2)
        return fresnel_r(k_rho) * np.exp(2j * gamma * emitter_pos[2]) * k_rho / gamma * \
            np.conj(legendre_normalized(m, n, gamma / k)) * legendre_normalized(mu, nu, - gamma / k) * \
            np.exp(1j * gamma * dz) * ss.jn(np.abs(m - mu), k_rho * d_rho)

    def subtracted_function(k_rho):
        gamma = np.emath.sqrt(k ** 2 - k_rho ** 2)
        return 1 / gamma + 1j / np.sqrt(k_rho ** 2 + k ** 2)

    subtracted_integral = \
        -1j * ss.iv(np.abs(m - mu) / 2, -1j * k * d_rho / 2) * ss.kv(np.abs(m - mu) / 2, -1j * k * d_rho / 2) + \
        1j * ss.iv(np.abs(m - mu) / 2, k * d_rho / 2) * ss.kv(np.abs(m - mu) / 2, k * d_rho / 2)

    singularity_coefficient = fresnel_r(k) * k * np.conj(legendre_normalized(m, n, 0)) * legendre_normalized(mu, nu, -0)

    min_limit, max_limit, step = 0, 3, 0.0001
    x = np.arange(min_limit, max_limit, step)
    y = [integrand(xi) for xi in x]
    ym = [subtracted_function(xi) * singularity_coefficient for xi in x]
    integral = si.trapz(y, x) + singularity_coefficient * subtracted_integral

    coef = 4 * np.pi * (-1j) ** n * 1j ** (nu + np.abs(m - mu))
    return coef * np.exp(1j * (m - mu) * d_phi) / k * integral, (x, y, ym)


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


def plot_integrand():
    m, n, mu, nu = 0, 1, 0, 1
    k, pos1, pos2 = 1.1, np.array([-1, 1, 2]), np.array([3, 4, 3])

    import matplotlib.pyplot as plt
    _, (x, y, ym) = reflection_element(m, n, mu, nu, k, pos1, pos2)
    plt.plot(x, np.real(y))
    plt.plot(x, np.imag(y))
    plt.plot(x, np.real(ym), linestyle='--')
    plt.plot(x, np.imag(ym), linestyle='--')
    plt.show()


# plot_integrand()
