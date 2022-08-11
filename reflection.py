import numpy as np
import matplotlib.pyplot as plt
import wavefunctions as wvfs
import mathematics as mths


def reflection_dir(incident_dir, normal):
    return incident_dir - 2 * normal * incident_dir.dot(normal)


def image_poses(sphere, interface, alpha):
    image_positions = np.zeros((len(alpha), 3), dtype=complex)

    for q in range(len(alpha)):
        distance = interface.int_dist(sphere.pos)
        image_positions[q] = sphere.pos - (2 * distance + 1j * alpha[q]) * interface.normal
    return image_positions


def image_contribution(m, n, mu, nu, k_fluid, image_positions, a):
    image_contrib = np.zeros(len(a), dtype=complex)
    for q in range(len(a)):
        image_contrib[q] = a[q] * wvfs.regular_separation_coefficient(mu, m, nu, n, k_fluid, - image_positions[q])
    return mths.complex_fsum(image_contrib)


def ref_coef(alpha_inc, freq, c_inc, c_t, rho_inc, rho_t):
    w = 2 * np.pi * freq
    h = np.sin(alpha_inc)
    kv_inc = np.emath.sqrt(w ** 2 / c_inc ** 2 - h ** 2)
    kv_t = np.emath.sqrt(w ** 2 / c_t ** 2 - h ** 2)
    return (rho_t * kv_inc - rho_inc * kv_t) / (rho_t * kv_inc - rho_inc * kv_t)


def prony(sample, order_approx):
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
    v_inc = np.emath.sqrt(w ** 2 / c_inc ** 2 - h ** 2)
    v_t = np.emath.sqrt(w ** 2 / c_t ** 2 - h ** 2)
    return (rho_t * v_inc - rho_inc * v_t) / (rho_t * v_inc + rho_inc * v_t)


def ref_coef_approx(w, c_inc, c_t, rho_inc, rho_t, order_approx, t_0):
    t = np.linspace(0, t_0, 2 * order_approx)
    v = w / c_inc * (1j * t + (1 - t / t_0))
    # h = np.linspace(0, w / c * np.sqrt(1 + t_0 ** 2), 2 * order_approx)
    h = np.emath.sqrt((w / c_inc) ** 2 - np.emath.power(v, 2))

    r_sample = np.zeros_like(v)
    for i in range(len(v)):
        r_sample[i] = ref_coef_h(h[i], w, c_inc, c_t, rho_inc, rho_t)
    a, alpha = prony(r_sample, order_approx)

    final_a = a * np.emath.power(np.e, alpha * t_0 / (1 - 1j * t_0))
    final_alpha = alpha * (- t_0) / (w / c_inc * (1 - 1j * t_0))
    return final_a, final_alpha


def ref_test(w, c_inc, c_t, rho_inc, rho_t, order_approx, t_0):
    angles = np.linspace(0, np.pi / 2, 100)
    h = w / c_inc * np.sin(angles)
    v = w / c_inc * np.cos(angles)

    r = np.zeros_like(angles)
    for i in range(len(angles)):
        r_h = ref_coef_h(h[i], w, c_inc, c_t, rho_inc, rho_t)
        r[i] = np.abs(r_h)

    a, alpha = ref_coef_approx(w, c_inc, c_t, rho_inc, rho_t, order_approx, t_0)

    r_approx = np.zeros_like(angles)
    for i in range(len(angles)):
        r_app = np.sum(a * np.exp(alpha * v[i]))
        r_approx[i] = np.abs(r_app)

    fig, ax = plt.subplots()
    ax.plot(angles * 180 / np.pi, r)
    ax.plot(angles * 180 / np.pi, r_approx)
    plt.show()


# ref_test(2 * np.pi * 80, 344, 548.7 - 492.321 * 1j, 1.293, 0.063 + 1j * 4.688, 11, 20)
# ref_test(2 * np.pi * 80, 344, 1400, 1.293, 1000, 10, 1)

