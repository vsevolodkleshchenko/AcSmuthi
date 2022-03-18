import numpy as np
import numpy.linalg
import scipy
import scipy.special


def neuman1(n, z):
    return -1j * scipy.special.hankel1(n, z) + 1j * scipy.special.jv(n, z)


def sph_neuman(n, z):
    return (-1) ** (n + 1) * np.sqrt(np.pi / 2 / z) * scipy.special.jv(-n - 0.5, z)


def sph_neuman_der(n, z):
    return (-1) ** n * np.sqrt(np.pi / (8 * z ** 3)) * scipy.special.jv(-n - 0.5, z) + \
           (-1) ** (n + 1) * np.sqrt(np.pi / 2 / z) * scipy.special.jvp(-n - 0.5, z)


def sph_hankel1(n, z):
    return scipy.special.spherical_jn(n, z) + 1j * sph_neuman(n, z)


def sph_hankel1_der(n, z):
    return scipy.special.spherical_jn(n, z, derivative=True) + 1j * sph_neuman_der(n, z)


def coef_plane_wave(m, n, alpha_x, alpha_y, alpha_z):
    r"""
    d_mn eq(4.40)
    """
    alpha = np.sqrt(alpha_x * alpha_x + alpha_y * alpha_y + alpha_z * alpha_z)
    alpha_phi = np.arccos(alpha_x / alpha)
    alpha_theta = np.arccos(alpha_z / alpha)
    return 4 * np.pi * 1j ** n * scipy.special.sph_harm(m, n, alpha_theta, alpha_phi)


def regular_wvfs(m, n, x, y, z, k):
    r"""
    regular wavefunctions
    """
    r = np.sqrt(x * x + y * y + z * z)
    phi = np.arccos(x / r)
    theta = np.arccos(z / r)
    return scipy.special.spherical_jn(n, k * r) * \
           scipy.special.sph_harm(m, n, theta, phi)


def outgoing_wvfs(m, n, x, y, z, k):
    r"""
    outgoing wavefunctions
    """
    r = np.sqrt(x * x + y * y + z * z)
    phi = np.arccos(x / r)
    theta = np.arccos(z / r)
    return sph_hankel1(n, k * r) * \
           scipy.special.sph_harm(m, n, theta, phi)


def coef_f(n, k, r_sph, l_sph):
    return scipy.special.spherical_jn(n, k * r_sph, derivative=True) + \
           l_sph * scipy.special.spherical_jn(n, k * r_sph)


def coef_g(n, k, r_sph, l_sph):
    return sph_hankel1_der(n, k * r_sph) + l_sph * sph_hankel1(n, k * r_sph)


def coef_q(n, k, r_sph, l_sph):
    return coef_f(n, k, r_sph, l_sph) / coef_g(n, k, r_sph, l_sph)


def matrix_q(m, k, r_sph, l_sph, order):
    q = np.eye(order - np.abs(m) + 1, dtype=complex)
    for i in range(0, order - np.abs(m) + 1):
        q[i, i] = coef_q(i, k, r_sph, l_sph)


# code below does not tested and should not work

def coef_a(s, nu, m):
    return np.sqrt(scipy.special.factorial(nu + m) / scipy.special.factorial(nu - m)) * \
           scipy.special.factorial(nu + s) / scipy.special.factorial(s + m) / \
           scipy.special.factorial(s) * scipy.special.factorial(nu - s) * np.sqrt(2 * nu + 1)


def coef_sz(m, n, nu, k, dist):
    w = 1j / (2 * k * dist)
    sum = 0
    for j in range(np.abs(m), n + nu + 1):
        s_0 = np.max(0, j - n)
        s_1 = np.min(nu, j - np.abs(m))
        sum1 = 0
        for s in range(s_0, s_1 + 1):
            sum1 += coef_a(s, nu, np.abs(m)) * coef_a(j - s, n, -np.abs(m))
        sum += scipy.special.factorial(j) * w ** j
    return (-1) ** m * 1j ** (n + nu) * np.exp(1j * k * dist) / (1j * k * dist) * sum


def matrix_sz1(m, k, dist, order):
    sz1 = np.empty((order - np.abs(m) + 1, order - np.abs(m) + 1))
    for n in range(0, order - np.abs(m) + 1):
        for nu in range(0, order - np.abs(m) + 1):
            sz1[n, nu] = coef_sz(m, nu, n, k, dist)
    return sz1


def matrix_sz2(m, k, dist, order):
    sz2 = np.empty((order - np.abs(m) + 1, order - np.abs(m) + 1))
    for n in range(order - np.abs(m) + 1):
        for nu in range(order - np.abs(m) + 1):
            sz2[n, nu] = (-1) ** n * coef_sz(m, nu, n, k, dist)
    return g


def solve_system(q1, q2, sz1, sz2, d1, d2):
    c2 = np.linalg.solve(q2 - np.dot(sz2, np.linalg.inv(q1), sz1),
                         d2 - np.dot(sz2, np.linalg.inv(q1), d1))
    c1 = np.dot(np.linalg.inv(q1), d1) - np.dot(np.linalg.inv(q1), sz1, c2)
    return c1, c2


def field_near_sphere1(x, y, z, k, alpha_x, alpha_y, alpha_z, r_sph1, l_sph1, r_sph2,
                       l_sph2, dist, order):
    u = 0
    for m in range(-order, order + 1):
        q1 = matrix_q(m, k, r_sph1, l_sph1, order)
        q2 = matrix_q(m, k, r_sph2, l_sph2, order)
        sz1 = matrix_sz1(m, k, dist, order)
        sz2 = matrix_sz2(m, k, dist, order)
        d1 = [coef_plane_wave(m, n, alpha_x, alpha_y, alpha_z) for n in
              range(np.abs(m), order + 1)]
        d2 = [coef_plane_wave(m, n, alpha_x, alpha_y, alpha_z) for n in
              range(np.abs(m), order + 1)]  # not right
        c1, c2 = solve_system(q1, q2, sz1, sz2, d1, d2)
        for n in range(np.abs(m), order + 1):
            transl = 0
            for nu in range(n, order):
                transl += 1
            u += d1 * regular_wvfs(m, n, x, y, z, k) + c1[n - np.abs(m)] * outgoing_wvfs() * \
                 outgoing_wvfs(m, n, x, y, z, k) * c2[n - np.abs(m)]

