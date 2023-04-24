import numpy as np
import scipy
import scipy.special as ss
from acsmuthi.utility import mathematics as mths


def n_idx(order):
    r"""Build np.array of numbers 0,1,1,1,2,2,2,2,2,...,n,..,n"""
    return np.repeat(np.arange(order + 1), np.arange(order + 1) * 2 + 1)


def m_idx(order):
    r"""Build np.array of numbers 0,-1,0,1,-2,-1,0,1,2,...,-n,..,n"""
    return np.concatenate([np.arange(-i, i + 1) for i in range(order + 1)])


def multipoles(n):
    r"""Build zip of multipoles indexes"""
    return zip(m_idx(n), n_idx(n))


def incident_coefficient(m, n, direction):
    r"""Coefficient in decomposition of plane wave"""
    dir_abs, dir_phi, dir_theta = mths.dec_to_sph(direction[0], direction[1], direction[2])
    return 4 * np.pi * 1j ** n * np.conj(ss.sph_harm(m, n, dir_phi, dir_theta))


def incident_coefficients(direction, order):
    r"""All coefficients in decomposition of plane wave (n <= order)"""
    inc_coef = np.zeros((order + 1) ** 2, dtype=complex)
    for m, n in multipoles(order):
        inc_coef[n ** 2 + n + m] = incident_coefficient(m, n, direction)
    return inc_coef


def incident_coefficients_array(direction, length, order):
    r"""Repeated incident coefficients to multiply it with basis functions counted in all points"""
    c_array = np.zeros(((order + 1) ** 2), dtype=complex)
    i = 0
    for mn in multipoles(order):
        c_array[i] = incident_coefficient(mn[0], mn[1], direction)
        i += 1
    return np.split(np.repeat(c_array, length), (order + 1) ** 2)


def regular_wvf(m, n, x, y, z, k):
    r"""Regular basis spherical wave function"""
    r, phi, theta = mths.dec_to_sph(x, y, z)
    return ss.spherical_jn(n, k * r) * scipy.special.sph_harm(m, n, phi, theta)


def regular_wvfs_array(order, x, y, z, k):
    r"""Builds np.array of all regular wave functions with n <= order"""
    rw_array = np.zeros(((order + 1) ** 2, *x.shape), dtype=complex)
    i = 0
    for mn in multipoles(order):
        rw_array[i] = regular_wvf(mn[0], mn[1], x, y, z, k)
        i += 1
    return rw_array


def outgoing_wvf(m, n, x, y, z, k):
    r"""Outgoing basis spherical wave function"""
    r, phi, theta = mths.dec_to_sph(x, y, z)
    return mths.spherical_h1n(n, k * r) * ss.sph_harm(m, n, phi, theta)


def outgoing_wvfs_array(order, x, y, z, k):
    r"""Builds np.array of all outgoing wave functions with n <= order"""
    ow_array = np.zeros(((order + 1) ** 2, *x.shape), dtype=complex)
    i = 0
    for mn in multipoles(order):
        ow_array[i] = outgoing_wvf(mn[0], mn[1], x, y, z, k)
        i += 1
    return ow_array
