import matplotlib.pyplot as plt
import numpy as np
import scipy.special as ss
import scipy.integrate
from acsmuthi.utility import mathematics as mths, wavefunctions as wvfs
from acsmuthi.postprocessing import rendering


def incident_field_test(m, n, x, y, z, k):
    r, phi, theta = mths.dec_to_sph(x, y, z)
    zeta = r * np.sin(theta)
    desired_field = wvfs.regular_wvf(m, n, x, y, z, k)  # ss.lpmn(m, n, np.cos(theta))[0][-1][-1] * ss.spherical_jn(n, k * r)
    a_mn = np.sqrt((2 * n + 1) / 4 / np.pi * ss.factorial(n - m) / ss.factorial(n + m))
    coefficient = 1j ** (m - n) / 2 * np.exp(1j * m * phi) * a_mn
    alpha = np.linspace(0, np.pi, 300)
    legendre = np.zeros(len(alpha), dtype=complex)
    for i in range(len(alpha)):
        legendre[i] = ss.clpmn(m, n, np.cos(alpha[i]), type=2)[0][-1][-1]
    d_integral = legendre * ss.jv(m, k * np.sin(alpha) * zeta) * np.exp(1j * k * np.cos(alpha) * z) * np.sin(alpha)
    actual_field = coefficient * scipy.integrate.simps(d_integral, alpha)
    error = np.abs(np.real(actual_field - desired_field) / np.real(desired_field))
    return error


def complete_inc_test(m, n, k):
    bound, number_of_points = 3, 15
    span = rendering.build_discretized_span(bound, number_of_points)
    plane = 'xz'
    plane_number = int(number_of_points / 2) + 1
    x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane=plane)
    negative_z = np.where(z_p < 0)
    error_plot = np.zeros(len(x_p))
    i, j = 0, 0
    for x, y, z in zip(x_p[negative_z], y_p[negative_z], z_p[negative_z]):
        error = incident_field_test(m, n, x, y, z, k)
        if i % (number_of_points // 2) == 0 and i != 0:
            j += number_of_points // 2 + 1
        error_plot[i + j] = error
        i += 1
    rendering.slice_plot(error_plot, span_v, span_h, plane)


complete_inc_test(-3, 5, 1.5)
