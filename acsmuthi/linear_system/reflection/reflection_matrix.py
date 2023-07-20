import numpy as np
import scipy.integrate as si
import scipy.special as ss
from acsmuthi.utility import wavefunctions as wvfs
from basics import fresnel_r, k_contour, legendre_normalized


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

