import numpy as np
import matplotlib.pyplot as plt
from utility import wavefunctions as wvfs
from utility import mathematics as mths


def reflection_dir(incident_dir, normal):
    r"""Direction of reflected from plane (normal) wave vector"""
    return incident_dir - 2 * normal * incident_dir.dot(normal)


def ref_coef(alpha_inc, freq, c_inc, c_t, rho_inc, rho_t):
    r"""Reflection coefficient between two gases/fluids depending on incident angle"""
    w = 2 * np.pi * freq
    h = np.sin(alpha_inc)
    kv_inc = np.emath.sqrt(w ** 2 / c_inc ** 2 - h ** 2)
    kv_t = np.emath.sqrt(w ** 2 / c_t ** 2 - h ** 2)
    return (rho_t * kv_inc - rho_inc * kv_t) / (rho_t * kv_inc - rho_inc * kv_t)


def prony(sample, order_approx):
    r"""Prony (exponential) approximation of sample"""
    matrix1 = np.zeros((order_approx, order_approx), dtype=complex)
    for j in range(order_approx):
        matrix1[j] = sample[j:j+order_approx]
    rhs1 = - sample[order_approx:]
    c_coefficients = np.linalg.solve(matrix1, rhs1)

    p_coefficients = np.flip(np.append(c_coefficients, 1.))
    p = np.roots(p_coefficients)
    alpha_coefficients = np.emath.log(p)

    matrix2 = np.zeros((order_approx, order_approx), dtype=complex)
    for j in range(order_approx):
        matrix2[j] = np.emath.power(p, j)
    rhs2 = sample[:order_approx]
    a_coefficients = np.linalg.solve(matrix2, rhs2)

    return a_coefficients, alpha_coefficients


def ref_coef_h(h, w, c_inc, c_t, rho_inc, rho_t):
    r"""Reflection coefficient depending on wave vector's horizontal component"""
    v_inc = np.emath.sqrt(w ** 2 / c_inc ** 2 - h ** 2)
    v_t = np.emath.sqrt(w ** 2 / c_t ** 2 - h ** 2)
    return (rho_t * v_inc - rho_inc * v_t) / (rho_t * v_inc + rho_inc * v_t)


def ref_coef_approx(w, c_inc, c_t, rho_inc, rho_t, order_approx, t_0):
    r"""Approximation of reflection coefficient with Prony method"""
    r_0 = (rho_t - rho_inc) / (rho_t + rho_inc)
    a, alpha = np.array([r_0], dtype=complex), np.array([0], dtype=complex)

    if order_approx > 1:
        t = np.linspace(0, t_0, 2 * (order_approx - 1))
        v = w / c_inc * (1j * t + (1 - t / t_0))
        h = np.emath.sqrt((w / c_inc) ** 2 - np.emath.power(v, 2))
        r_sample = np.zeros_like(v)
        for i in range(len(v)):
            r_sample[i] = ref_coef_h(h[i], w, c_inc, c_t, rho_inc, rho_t) - r_0
        a_approx, alpha_approx = prony(r_sample, order_approx - 1)
        alpha_approx = alpha_approx * (2 * order_approx - 3) / t_0
        a_approx = a_approx * np.emath.power(np.e, alpha_approx * t_0 / (1 - 1j * t_0))
        alpha_approx = alpha_approx * (- t_0) / (w / c_inc * (1 - 1j * t_0))
        a, alpha = np.append(a, a_approx), np.append(alpha, alpha_approx)
    return a, alpha


def ref_test(w, c_inc, c_t, rho_inc, rho_t, order_approx, t_0):
    r"""Test for correct approximation reflection coefficient"""
    angles = np.linspace(0, 0.5 * np.pi / 2, 100)
    h = w / c_inc * np.sin(angles)
    v = w / c_inc * np.cos(angles)

    r = np.zeros_like(angles)
    for i in range(len(angles)):
        r_h = ref_coef_h(h[i], w, c_inc, c_t, rho_inc, rho_t)
        r[i] = np.abs(r_h)

    a, alpha = ref_coef_approx(w, c_inc, c_t, rho_inc, rho_t, order_approx, t_0)

    r_approx = np.zeros_like(angles)
    for i in range(len(angles)):
        r_app = mths.complex_fsum(a * np.exp(alpha * v[i]))
        r_approx[i] = np.abs(r_app)

    fig, ax = plt.subplots()
    ax.plot(angles * 180 / np.pi, r)
    ax.plot(angles * 180 / np.pi, r_approx)
    plt.show()


# ref_test(2 * np.pi * 80, 344, 548.671 - 492.321 * 1j, 1.293, 0.063 + 1j * 4.688, 11, 19)
# ref_test(2 * np.pi * 82, 331, 1403, 1.225, 997, 1, 1)
# ref_test(2 * np.pi * 82, 331, 1403, 1.225, 997, 2, 0.07)
# ref_test(2 * np.pi * 82, 331, 1403, 1.225, 997, 3, 0.01)
ref_test(2 * np.pi * 82, 331, 1403, 1.225, 997, 6, 0.3)
# ref_test(2 * np.pi * 82, 331, 1403, 1.225, 997, 10, 0.6)
