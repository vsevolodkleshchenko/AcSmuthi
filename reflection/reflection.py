import numpy as np
import scipy.integrate as si
import scipy.special as ss
from reflection.basics import fresnel_r, k_contour, legendre_normalized, dec_to_cyl


def compute_reflection_integrand(kp, k, d_rho, dz, ds, m, n, mu, nu):
    kz = np.emath.sqrt(k ** 2 - kp ** 2)
    return fresnel_r(kp) * np.exp(1j * kz * (2 * ds + dz)) * kp / kz * ss.jn(m - mu, kp * d_rho) * \
        legendre_normalized(mu, nu, kz / k) * legendre_normalized(m, n, - kz / k)


def reflection_element_i(m, n, mu, nu, k, emitter_pos, receiver_pos, k_parallel=k_contour()):
    dist = receiver_pos - emitter_pos
    d_rho, d_phi, dz = dec_to_cyl(dist[0], dist[1], dist[2])
    ds = np.abs(emitter_pos[2])

    def f_integrand(k_rho):
        return compute_reflection_integrand(k_rho, k, d_rho, dz, ds, m, n, mu, nu)

    integrand = [f_integrand(kp) for kp in k_parallel]
    integral = si.trapz(integrand, k_parallel)

    coef = 4 * np.pi * 1j ** (nu - n + m - mu) / k
    return coef * np.exp(1j * (m - mu) * d_phi) * integral


def compute_reflection_integrand_angled(beta, k, d_rho, dz, ds, m, n, mu, nu):
    cos_b, sin_b = np.cos(beta), np.sin(beta)
    return fresnel_r(beta) * np.exp(-1j * k * cos_b * (2 * ds + dz)) * ss.jn(m - mu, k * sin_b * d_rho) * \
        legendre_normalized(mu, nu, -cos_b) * legendre_normalized(m, n, cos_b) * sin_b


def reflection_element_i_angled(m, n, mu, nu, k, emitter_pos, receiver_pos, beta_max=0.5, d_beta=0.05):
    dist = receiver_pos - emitter_pos
    d_rho, d_phi, dz = dec_to_cyl(dist[0], dist[1], dist[2])
    ds = np.abs(emitter_pos[2])

    def angle_contour():
        im_part = np.pi / 2 + 1j * np.arange(beta_max, 0. - d_beta, -d_beta)
        re_part = np.arange(np.pi / 2, np.pi + d_beta, d_beta) + 0j
        return np.concatenate([im_part, re_part])

    def f_integrand(beta):
        return compute_reflection_integrand_angled(beta, k, d_rho, dz, ds, m, n, mu, nu)

    beta_contour = angle_contour()
    integrand = [f_integrand(beta) for beta in beta_contour]
    integral = si.trapz(integrand, beta_contour)

    coef = 4 * np.pi * 1j ** (nu - n + m - mu)
    return coef * np.exp(1j * (m - mu) * d_phi) * integral
