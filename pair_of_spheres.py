import numpy as np
import scipy
import scipy.special
from sympy.physics.wigner import clebsch_gordan
from sympy.physics.quantum.matrixutils import scipy

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


def scatered_Upq(p, q, k, x, y, z):
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
    coef_sqr = scipy.special.factorial(q + np.abs(p)) * \
               scipy.special.factorial(n + np.abs(p)) / \
               scipy.special.factorial(q - np.abs(p)) / \
               scipy.special.factorial(n - np.abs(p))
    mul3jsym = (2 * s + 1) * clebsch_gordan(q, n, s, 0, 0, 0) * \
             clebsch_gordan(q, n, s, p, -p, 0)
    return (-1) ** (-p) * (2 * s + 1) * np.sqrt(coef_sqr) * mul3jsym


def coef_Qpqpn2(q, p, n, x, y, z, k, d):
    r"""
    (Eq 15)
    """
    r = np.sqrt(x * x + y * y + z * z)
    const = 1j ** (n - q) * (2 * q + 1) * scipy.special.factorial(q - p) / \
           scipy.special.factorial(q + p)
    sum = 0
    s_min = np.abs(q - n)
    s_max = q + n
    if r >= d:
        for s in range(s_min, s_max):
            sum += (-1j) ** s * coef_Bqpnps(q, p, n, p, d) * \
                   scipy.special.spherical_jn(s, k * d)
    else:
        for s in range(s_min, s_max):
            sum += (-1j) ** s * coef_Bqpnps(q, p, n, p, d) * \
                   scipy.special.hankel1(s, k * d)
    return const * sum

