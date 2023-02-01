import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import scipy.special
import acsmuthi.utility.wavefunctions as wvfs
import acsmuthi.postprocessing.rendering as rendering


def hat_leg(m, n, x):
    coefficient = np.sqrt((2 * n + 1) / 4 / np.pi * scipy.special.factorial(n - m) / scipy.special.factorial(n + m))
    return coefficient * scipy.special.clpmn(m, n, x, type=2)[0][-1][-1]


def integrand(alpha, k_rho, m, n, x, y, z, k):
    k_z = np.emath.sqrt(k ** 2 - k_rho ** 2)
    exp = np.exp(1j * (k_rho * np.cos(alpha) * x + k_rho * np.sin(alpha) * y + k_z * np.abs(z)))
    if z >= 0:
        legendre = hat_leg(m, n, k_z / k)
    else:
        legendre = hat_leg(m, n, - k_z / k)
    return k_rho / k_z * (-1j) ** n * legendre * np.exp(1j * m * alpha) * exp / 2 / k / np.pi


def y_integrated3(m, n, x, y, z, k):
    alpha_start, alpha_end, k_rho_start, k_rho_end = 0, 2 * np.pi, 0.05, 199

    alphas = np.linspace(alpha_start, alpha_end, 2000)
    k_rhos = np.linspace(k_rho_start, k_rho_end, 2000)

    k_rho_integrand = np.zeros(len(k_rhos), dtype=complex)
    for i, k_rho in enumerate(k_rhos):
        alpha_integrand = integrand(alphas, k_rho, m, n, x, y, z, k)
        k_rho_integrand[i] = scipy.integrate.simpson(alpha_integrand, alphas)
    return scipy.integrate.simpson(k_rho_integrand, k_rhos)


def y_integrated2(m, n, x, y, z, k):
    alpha_start, alpha_end, k_rho_start, k_rho_end = 0, 2 * np.pi, 0.06, np.inf

    def real_int(alpha, k_rho):
        return np.real(integrand(alpha, k_rho, m, n, x, y, z, k))

    def imag_int(alpha, k_rho):
        return np.imag(integrand(alpha, k_rho, m, n, x, y, z, k))

    y_int_real = scipy.integrate.dblquad(lambda alpha, k_rho: real_int(alpha, k_rho), k_rho_start, k_rho_end,
                                      lambda alpha: alpha_start, lambda alpha: alpha_end, epsabs=1e-3)[0]
    y_int_imag = scipy.integrate.dblquad(lambda alpha, k_rho: imag_int(alpha, k_rho), k_rho_start, k_rho_end,
                                         lambda alpha: alpha_start, lambda alpha: alpha_end, epsabs=1e-3)[0]
    return y_int_real + 1j * y_int_imag


def y_integrated(m, n, x, y, z, k):
    alpha_start, alpha_end, k_rho_start, k_rho_end = 0, 2 * np.pi, 0.1, np.inf

    def real_integrand(alpha, k_rho):
        return np.real(integrand(alpha, k_rho, m, n, x, y, z, k))

    def imag_integrand(alpha, k_rho):
        return np.imag(integrand(alpha, k_rho, m, n, x, y, z, k))

    def real_k_integrand(k_rho):
        return scipy.integrate.quad(real_integrand, alpha_start, alpha_end, args=(k_rho, ), epsabs=1e-3)[0]

    def imag_k_integrand(k_rho):
        return scipy.integrate.quad(imag_integrand, alpha_start, alpha_end, args=(k_rho, ), epsabs=1e-3)[0]

    y_int = scipy.integrate.quad(lambda k_rho: real_k_integrand(k_rho), k_rho_start, k_rho_end)[0] + \
            1j * scipy.integrate.quad(lambda k_rho: imag_k_integrand(k_rho), k_rho_start, k_rho_end)[0]
    return y_int


def compare(m, n, k):
    bound, number_of_points = 4, 21
    span = rendering.build_discretized_span(bound, number_of_points)
    plane_number = int(number_of_points / 2) + 1
    x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane='xz')
    y_int = np.zeros(len(x_p), dtype=complex)
    y_ex = wvfs.outgoing_wvf(m, n, x_p, y_p, z_p, k)
    i = 0
    for x, y, z in zip(x_p, y_p, z_p):
        print(f'№{i} of {len(x_p)}')
        t = time.process_time()
        y_int[i] = y_integrated2(m, n, x, y, z, k)
        print("time:", time.process_time()-t)
        i += 1
    rendering.plots_for_tests(np.real(y_int), np.real(y_ex), span_v, span_h)
    rendering.plots_for_tests(np.imag(y_int), np.imag(y_ex), span_v, span_h)
    rendering.plots_for_tests(np.abs(np.real(y_int) - np.real(y_ex)),
                              np.abs(np.imag(y_int) - np.imag(y_ex)), span_v, span_h)
    np.testing.assert_allclose(np.real(y_int), np.real(y_ex), rtol=5e-2)
    # t = time.process_time()
    # print(wvfs.outgoing_wvf(m, n, -1, 3, -3, k), y_integrated2(m, n, -1, 3, -3, k))
    # print("time:", time.process_time() - t)


if __name__ == '__main__':
    compare(0, 2, 1)

