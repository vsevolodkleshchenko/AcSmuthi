import numpy as np
import scipy
import scipy.special
from sympy.physics.wigner import clebsch_gordan
from sympy.physics.quantum.matrixutils import scipy

import plane_one_sphere

"Everything from 'Acoustic scattering by a pair of spheres' - G.C.Gaunaurd 1995"


def coef_Apq(p, q, alpha):
    r"""
    (Eq 3a)
    """
    return 1j ** q * (2 * q + 1) * scipy.special.factorial(q - p) / \
           scipy.special.factorial(q + p) * scipy.special.lpmv(p, q, np.cos(alpha))


def incident_Upq(p, q, k, x, y, z):
    r"""
    u_pq^(1) (Eq 3b)
    """
    r = np.sqrt(x * x + y * y + z * z)
    phi = np.arccos(x / r)
    return scipy.special.spherical_jn(q, k * r) * np.exp(1j * p * phi) * \
           scipy.special.lpmv(p, q, z / r)


def scattered_Upq(p, q, k, x, y, z):
    r"""
    u_pq^(2) (Eq 8)
    """
    r = np.sqrt(x * x + y * y + z * z)
    phi = np.arccos(x / r)
    return scipy.special.hankel1(q, k * r) * np.exp(1j * p * phi) * \
           scipy.special.lpmv(p, q, z / r)


def coef_Bqpnps(q, p, n, s):
    r"""
    (Eq 16)
    """
    coef_sqr = scipy.special.factorial(q + abs(p)) * \
               scipy.special.factorial(n + abs(p)) / \
               scipy.special.factorial(q - abs(p)) / \
               scipy.special.factorial(n - abs(p))
    mul3jsym = (2 * s + 1) * clebsch_gordan(q, n, s, 0, 0, 0) * \
             clebsch_gordan(q, n, s, p, -p, 0)
    return (-1) ** (-p) * (2 * s + 1) * np.sqrt(coef_sqr) * mul3jsym


def coef_Qpqpn2(p, q, n, x, y, z, k, d):
    r"""
    (Eq 15)
    """
    r = np.sqrt(x * x + y * y + z * z)
    const = 1j ** (n - q) * (2 * q + 1) * scipy.special.factorial(q - p) / \
           scipy.special.factorial(q + p)
    sum = np.empty_like(r, dtype=complex)
    s_min = abs(q - n)
    s_max = q + n
    for i in range(len(r)):
        if r[i] >= d:
            for s in range(s_min, s_max + 1):
                sum[i] += (-1j) ** s * coef_Bqpnps(q, p, n, s) * \
                       scipy.special.spherical_jn(s, k * d)
        else:
            for s in range(s_min, s_max + 1):
                sum[i] += (-1j) ** s * coef_Bqpnps(q, p, n, s) * \
                       scipy.special.hankel1(s, k * d)
    return const * sum


def solve_system(order, k, a):
     b = np.zeros((order, 2))
     c = np.zeros((order, 2))



def coef_Bpq(p, q, alpha, c_t, c_l, w, k, a, ro, ro_s):
    return 1
#    return coef_Apq(p, q, alpha) * \
#           plane_one_sphere.coef_cn(q, c_t, c_l, w, k, a, ro, ro_s)


def coef_Cpq(p, q, alpha, c_t, c_l, w, k, a, ro, ro_s, d):
    return 1
#    return coef_Bpq(p, q, alpha, c_t, c_l, w, k, a, ro, ro_s) * np.exp(1j * k * d)


def total_Field(x, y, z, alpha, c_t, c_l, w, k, a, ro, ro_s, d, order):
    field = 0
    for q in range(order + 1):
        for p in range(-q, q + 1):
            Qu = 0
            for n in range(p, order + 1):
                Qu += coef_Qpqpn2(p, q, n, x, y, z, k, d) * \
                      incident_Upq(p, n, k, x, y, z)
            field += coef_Apq(p, q, alpha) * incident_Upq(p, q, k, x, y, z) + \
                     coef_Bpq(p, q, alpha, c_t, c_l, w, k, a, ro, ro_s) * scattered_Upq(p, q, k, x, y, z) + \
                     coef_Cpq(p, q, alpha, c_t, c_l, w, k, a, ro, ro_s, d) * Qu
    return field


def spheres_simulation():
    # coordinates
    span_x = np.linspace(16, 19, 3)
    span_y = np.linspace(1, 10, 3)
    span_z = np.linspace(2, 8, 3)
    grid = np.vstack(np.meshgrid(span_x, span_y, span_z)).reshape(3, -1).T
    x = grid[:, 0]
    y = grid[:, 1]
    z = grid[:, 2]

    # parameters of the spheres
    a = 10
    ro_s = 20
    c_l = 2.5
    c_t = 3.5
    d = 3.

    # parameters of fluid
    ro = 1.1

    # parameters of the field
    p0 = 4
    k = 1.2
    c = 30
    w = k * c
    alpha = 0.8

    # order of decomposition
    order = 2

    field = total_Field(x, y, z, alpha, c_t, c_l, w, k, a, ro, ro_s, d, order)
    print(field)

spheres_simulation()
