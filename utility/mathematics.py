import numpy as np
import math
import pywigxjpf as wig
import scipy
import scipy.special as ss


def sph_neyman(n, z):
    r"""Spherical Neyman function"""
    return (-1) ** (n + 1) * np.sqrt(np.pi / 2 / z) * scipy.special.jv(-n - 0.5, z)


def sph_neyman_der(n, z):
    r"""First derivative of spherical Neyman function"""
    return (-1) ** n * np.sqrt(np.pi / (8 * z ** 3)) * scipy.special.jv(-n - 0.5, z) + \
           (-1) ** (n + 1) * np.sqrt(np.pi / 2 / z) * scipy.special.jvp(-n - 0.5, z)


def sph_hankel1(n, z):
    r"""Spherical Hankel function of the first kind"""
    return scipy.special.spherical_jn(n, z) + 1j * sph_neyman(n, z)


def sph_hankel1_der(n, z):
    r"""First derivative of spherical Hankel function of the first kind"""
    return scipy.special.spherical_jn(n, z, derivative=True) + 1j * sph_neyman_der(n, z)


def sph_bessel_der2(n, z):
    r"""Second derivative of spherical Bessel function"""
    if n == 0:
        return - ss.spherical_jn(1, z, derivative=True)
    else:
        return ss.spherical_jn(n - 1, z, derivative=True) + (n + 1) / z**2 * ss.spherical_jn(n, z) - \
               (n + 1) / z * ss.spherical_jn(n, z, derivative=True)


def csph_harm(m, n, phi, theta):
    r"""Spherical harmonic of complex argument"""
    coefficient = np.sqrt((2 * n + 1) / 4 / np.pi * scipy.special.factorial(n - m) / scipy.special.factorial(n + m))
    return coefficient * np.exp(1j * m * phi) * scipy.special.clpmn(m, n, np.cos(theta), type=2)[0][-1][-1]


def dec_to_sph(x, y, z):
    """Transition from cartesian coordinates to spherical coordinates"""
    e = 1e-16
    r = np.sqrt(x * x + y * y + z * z)
    phi = np.zeros(np.size(r))
    theta = np.zeros(np.size(r))
    settings = np.seterr(all='ignore')
    theta = np.where(r >= e, np.arccos(z / r), theta)
    phi = np.where((np.abs(x) <= e) & (y > e), np.pi / 2, phi)
    phi = np.where((np.abs(x) <= e) & (y < -e), 3 * np.pi / 2, phi)
    phi = np.where((np.abs(y) <= e) & (x < -e), np.pi, phi)
    phi = np.where((x > e) & (y > e), np.arctan(y / x), phi)
    phi = np.where((x < -e) & (y > e), np.pi - np.arctan(- y / x), phi)
    phi = np.where((x < -e) & (y < -e), np.pi + np.arctan(y / x), phi)
    phi = np.where((x > e) & (y < -e), 2 * np.pi - np.arctan(- y / x), phi)
    np.seterr(**settings)
    if len(phi) == len(theta) == 1:
        phi, theta = phi[0], theta[0]
    return r, phi, theta


def gaunt_coefficient(n, m, nu, mu, q):
    r"""Gaunt coefficient: G(n,m;nu,mu;q)"""
    s = np.sqrt((2 * n + 1) * (2 * nu + 1) * (2 * q + 1) / 4 / np.pi)
    wig.wig_table_init(60, 3)  # this needs revision
    wig.wig_temp_init(60)  # this needs revision
    return (-1.) ** (m + mu) * s * wig.wig3jj(2*n, 2*nu, 2*q, 0, 0, 0) * \
           wig.wig3jj(2*n, 2*nu, 2*q, 2*m, 2*mu, - 2*m - 2*mu)


def complex_fsum(array):
    r"""Accurate sum of numpy.array with complex values"""
    return math.fsum(np.real(array)) + 1j * math.fsum(np.imag(array))


def spheres_multipoles_fsum(field_array, length):
    r"""Accurate sum by spheres and multipoles;
    the shape of field array: 0 axis - spheres, 1 axis - multipoles, 2 axis - coordinates
    return: numpy.array with values of field in all coordinates """
    field = np.zeros(length, dtype=complex)
    for i in range(length):
        field[i] = complex_fsum(np.concatenate(field_array[:, :, i]))
    return field


def multipoles_fsum(field_array, length):
    r"""Accurate sum by multipoles; the shape of field array: 0 axis - multipoles, 1 axis - coordinates
    return: np.array with values of field in all coordinates """
    field = np.zeros(length, dtype=complex)
    for i in range(length):
        field[i] = complex_fsum(field_array[:, i])
    return field


def spheres_fsum(field_array, length):
    r"""Accurate sum by multipoles; the shape of field array: 0 axis - spheres, 1 axis - coordinates
    return: np.array with values of field in all coordinates """
    field = np.zeros(length, dtype=complex)
    for i in range(length):
        field[i] = complex_fsum(field_array[:, i])
    return field


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
