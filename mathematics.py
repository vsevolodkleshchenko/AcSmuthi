import numpy as np
import math
import pywigxjpf as wig
import scipy
import scipy.special


def sph_neyman(n, z):
    r""" spherical Neyman function """
    return (-1) ** (n + 1) * np.sqrt(np.pi / 2 / z) * scipy.special.jv(-n - 0.5, z)


def sph_neyman_der(n, z):
    r""" first derivative of spherical Neyman function """
    return (-1) ** n * np.sqrt(np.pi / (8 * z ** 3)) * scipy.special.jv(-n - 0.5, z) + \
           (-1) ** (n + 1) * np.sqrt(np.pi / 2 / z) * scipy.special.jvp(-n - 0.5, z)


def sph_hankel1(n, z):
    r""" Spherical Hankel function of the first kind """
    return scipy.special.spherical_jn(n, z) + 1j * sph_neyman(n, z)


def sph_hankel1_der(n, z):
    r""" First derivative of spherical Hankel function of the first kind """
    return scipy.special.spherical_jn(n, z, derivative=True) + 1j * sph_neyman_der(n, z)


def dec_to_sph(x, y, z):
    """ Transition from cartesian cs to spherical cs """
    e = 1e-16
    r = np.sqrt(x * x + y * y + z * z)
    phi = np.zeros(np.size(r))
    theta = np.zeros(np.size(r))
    theta = np.where(r >= e, np.arccos(z / r), theta)
    phi = np.where((x > e) & (y > e), np.arctan(y / x), phi)
    phi = np.where((x < -e) & (y > e), np.pi - np.arctan(- y / x), phi)
    phi = np.where((x < -e) & (y < -e), np.pi + np.arctan(y / x), phi)
    phi = np.where((x > e) & (y < -e), 2 * np.pi - np.arctan(- y / x), phi)
    phi = np.where((np.abs(x) <= e) & (y > e), np.pi / 2, phi)
    phi = np.where((np.abs(x) <= e) & (y < -e), 3 * np.pi / 2, phi)
    phi = np.where((np.abs(y) <= e) & (x < -e), np.pi, phi)
    return r, phi, theta


def gaunt_coefficient(n, m, nu, mu, q):
    r""" Gaunt coefficient: G(n,m;nu,mu;q)
    eq(3.71) in 'Encyclopedia' """
    s = np.sqrt((2 * n + 1) * (2 * nu + 1) * (2 * q + 1) / 4 / np.pi)
    wig.wig_table_init(60, 3)  # this needs revision
    wig.wig_temp_init(60)  # this needs revision
    return (-1.) ** (m + mu) * s * wig.wig3jj(2*n, 2*nu, 2*q, 0, 0, 0) * \
           wig.wig3jj(2*n, 2*nu, 2*q, 2*m, 2*mu, - 2*m - 2*mu)


def complex_fsum(array):
    return math.fsum(np.real(array)) + 1j * math.fsum(np.imag(array))


def spheres_multipoles_fsum(field_array, length):
    r""" do accurate sum by spheres and multipoles
    the shape of field array: 0 axis - spheres, 1 axis - multipoles, 2 axis - coordinates
    return: np.array with values of field in all coordinates """
    field = np.zeros(length, dtype=complex)
    for i in range(length):
        field[i] = complex_fsum(np.concatenate(field_array[:, :, i]))
    return field


def multipoles_fsum(field_array, length):
    r""" do accurate sum by multipoles
    the shape of field array: 0 axis - multipoles, 1 axis - coordinates
    return: np.array with values of field in all coordinates """
    field = np.zeros(length, dtype=complex)
    for i in range(length):
        field[i] = complex_fsum(field_array[:, i])
    return field
