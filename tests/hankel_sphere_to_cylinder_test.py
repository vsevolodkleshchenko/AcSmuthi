import matplotlib.pyplot as plt
import numpy as np
import scipy.special as ss
import scipy.integrate
from acsmuthi.utility import mathematics as mths, wavefunctions as wvfs
from acsmuthi.postprocessing import rendering


def incident_field_test(m, n, x, y, z, k):
    r, phi, theta = mths.dec_to_sph(x, y, z)
    zeta = r * np.sin(theta)
    desired_field = ss.lpmn(m, n, np.cos(theta))[0][-1][-1] * mths.sph_hankel1(n, k * r)  # wvfs.outgoing_wvf(m, n, x, y, z, k)
    a_mn = 1 # np.sqrt((2 * n + 1) / 4 / np.pi * ss.factorial(n - m) / ss.factorial(n + m))
    coefficient = 1j ** (n - m)  # * np.exp(1j * m * phi) * a_mn
    h = np.linspace(0, 100, 3000)
    gamma = np.emath.sqrt(k ** 2 - h ** 2)
    legendre = np.zeros(len(h), dtype=complex)
    for i in range(len(gamma)):
        legendre[i] = ss.clpmn(m, n, gamma[i] / k, type=2)[0][-1][-1]
    d_integral = legendre * ss.jv(m, h * zeta) * np.exp(-1j * gamma * z) * h / gamma / k
    actual_field = coefficient * scipy.integrate.simps(d_integral, h)
    error = np.abs(np.real(actual_field - desired_field) / np.real(desired_field))
    return error


def reflected_field_test(m, n, x, y, z, k):
    x_s, y_s, z_s = 0, 0, 0.1
    x_i, y_i, z_i = 0, 0, -0.1
    delta = np.abs(z_s - 0)
    a, alpha = 0.997, 0
    r_s, phi_s, theta_s = mths.dec_to_sph(x - x_s, y - y_s, z - z_s)
    zeta_s = r_s * np.sin(theta_s)
    h = np.linspace(0, 5, 3000)
    gamma = np.emath.sqrt(k ** 2 - h ** 2)
    a_mn = np.sqrt((2 * n + 1) / 4 / np.pi * ss.factorial(n - m) / ss.factorial(n + m))
    coefficient = 1j ** (n - m) * (-1) ** (n + m) * np.exp(1j * m * phi_s) * a_mn / k * a
    legendre = np.zeros(len(h), dtype=complex)
    for i in range(len(gamma)):
        legendre[i] = ss.clpmn(m, n, - gamma[i] / k, type=2)[0][-1][-1]
    d_integral = legendre * ss.jv(m, h * zeta_s) * np.exp(1j * gamma * (z - z_s + 2 * delta - 1j * alpha)) * h / gamma
    actual_field = coefficient * scipy.integrate.simps(d_integral, h)
    desired_field = (-1) ** (n + m) * wvfs.outgoing_wvf(m, n, x - x_i, y - y_i, z - z_i, k) * a
    error = np.abs((actual_field - desired_field) / desired_field)
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


def complete_ref_test(m, n, k):
    bound, number_of_points = 5, 17
    span = rendering.build_discretized_span(bound, number_of_points)
    plane = 'xz'
    plane_number = int(number_of_points / 2) + 1
    x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane=plane)
    negative_z = np.where(z_p > 0)
    error_plot = np.zeros(len(x_p))
    i, j = 0, 0
    for x, y, z in zip(x_p[negative_z], y_p[negative_z], z_p[negative_z]):
        error = reflected_field_test(m, n, x, y, z, k)
        if i % (number_of_points // 2) == 0:
            j += number_of_points // 2 + 1
        error_plot[i + j] = error
        i += 1
    rendering.slice_plot(error_plot, span_v, span_h, plane)


complete_inc_test(-3, 5, 1.5)
# complete_ref_test(-3, 5, 1.5)
