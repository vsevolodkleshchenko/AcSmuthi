import numpy as np
import scipy
import scipy.special as ss
import mathematics as mths


def n_idx(order):
    r""" build np.array of numbers 0,1,1,1,2,2,2,2,2,...,n,..,n """
    return np.repeat(np.arange(order + 1), np.arange(order + 1) * 2 + 1)


def m_idx(order):
    r""" build np.array of numbers 0,-1,0,1,-2,-1,0,1,2,...,-n,..,n """
    return np.concatenate([np.arange(-i, i + 1) for i in range(order + 1)])


def multipoles(n):
    r""" build zip of multipoles indexes """
    return zip(m_idx(n), n_idx(n))


def incident_coefficient(m, n, direction):
    r""" Coefficient in decomposition of plane wave
    d^m_n - eq(4.40) of 'Encyclopedia' """
    dir_abs, dir_phi, dir_theta = mths.dec_to_sph(direction[0], direction[1], direction[2])
    return 4 * np.pi * 1j ** n * np.conj(ss.sph_harm(m, n, dir_phi, dir_theta))


def incident_coefficients(direction, order):
    r""" Coefficients in decomposition of plane wave
    d^m_n - eq(4.40) of 'Encyclopedia' """
    inc_coef = np.zeros((order + 1) ** 2, dtype=complex)
    for m, n in multipoles(order):
        inc_coef[n ** 2 + n + m] = incident_coefficient(m, n, direction)
    return inc_coef


def incident_coefficients_array(direction, length, order):
    c_array = np.zeros(((order + 1) ** 2), dtype=complex)
    i = 0
    for mn in multipoles(order):
        c_array[i] = incident_coefficient(mn[0], mn[1], direction)
        i += 1
    return np.split(np.repeat(c_array, length), (order + 1) ** 2)


def regular_wvf(m, n, x, y, z, k):
    r""" Regular basis spherical wave function
    ^psi^m_n - eq(between 4.37 and 4.38) of 'Encyclopedia' """
    r, phi, theta = mths.dec_to_sph(x, y, z)
    return ss.spherical_jn(n, k * r) * scipy.special.sph_harm(m, n, phi, theta)


def cregular_wvf(m, n, x, y, z, k):
    r""" Regular basis spherical wave function of complex argument
    ^psi^m_n - eq(between 4.37 and 4.38) of 'Encyclopedia' """
    r, phi, theta = mths.dec_to_sph(x, y, z)
    return ss.spherical_jn(n, k * r) * mths.csph_harm(m, n, phi, theta)


def regular_wvfs_array(order, x, y, z, k):
    r""" builds np.array of all regular wave functions with n <= order"""
    rw_array = np.zeros(((order + 1) ** 2, len(x)), dtype=complex)
    i = 0
    for mn in multipoles(order):
        rw_array[i] = regular_wvf(mn[0], mn[1], x, y, z, k)
        i += 1
    return rw_array


def outgoing_wvf(m, n, x, y, z, k):
    r""" Outgoing basis spherical wave function
    psi^m_n - eq(between 4.37 and 4.38) in 'Encyclopedia' """
    r, phi, theta = mths.dec_to_sph(x, y, z)
    return mths.sph_hankel1(n, k * r) * ss.sph_harm(m, n, phi, theta)


def outgoing_wvfs_array(order, x, y, z, k):
    r""" builds np.array of all outgoing wave functions with n <= order"""
    ow_array = np.zeros(((order + 1) ** 2, len(x)), dtype=complex)
    i = 0
    for mn in multipoles(order):
        ow_array[i] = outgoing_wvf(mn[0], mn[1], x, y, z, k)
        i += 1
    return ow_array


def local_incident_coefficient(m, n, k, direction, position, order):
    r""" Counts local incident coefficients
    d^m_nj - eq(42) in 'Multiple scattering and scattering cross sections P. A. Martin' """
    incident_coefficient_array = np.zeros((order+1) ** 2, dtype=complex)
    i = 0
    for munu in multipoles(order):
        incident_coefficient_array[i] = incident_coefficient(munu[0], munu[1], direction) * \
                                        regular_separation_coefficient(munu[0], m, munu[1], n, k, position)
        i += 1
    return mths.complex_fsum(incident_coefficient_array)


def axisymmetric_outgoing_wvf(n, x, y, z, k):
    r""" Outgoing axisymmetric basis spherical wave function """
    r, phi, theta = mths.dec_to_sph(x, y, z)
    return mths.sph_hankel1(n, k * r) * ss.lpmv(0, n, np.cos(theta))


def axisymmetric_outgoing_wvfs_array(x, y, z, k, length, order):
    r""" builds np.array of all axisymmetric outgoing wave functions with n <= order"""
    as_ow_array = np.zeros((order + 1, length), dtype=complex)
    for n in range(order + 1):
        as_ow_array[n] = axisymmetric_outgoing_wvf(n, x, y, z, k)
    return as_ow_array


def regular_separation_coefficient(m, mu, n, nu, k, dist):
    r""" Coefficient ^S^mmu_nnu(b) of separation matrix
    eq(3.92) and eq(3.74) in 'Encyclopedia' """
    if abs(n - nu) >= abs(m - mu):
        q0 = abs(n - nu)
    if (abs(n - nu) < abs(m - mu)) and ((n + nu + abs(m - mu)) % 2 == 0):
        q0 = abs(m - mu)
    if (abs(n - nu) < abs(m - mu)) and ((n + nu + abs(m - mu)) % 2 != 0):
        q0 = abs(m - mu) + 1
    q_lim = (n + nu - q0) // 2
    sum_array = np.zeros(q_lim + 1, dtype=complex)
    i = 0
    if dist.dtype == complex:
        reg_wvf = cregular_wvf
    else:
        reg_wvf = regular_wvf
    for q in range(0, q_lim + 1):
        sum_array[i] = (-1) ** q * reg_wvf(m - mu, q0 + 2 * q, dist[0], dist[1], dist[2], k) * \
                       mths.gaunt_coefficient(n, m, nu, -mu, q0 + 2 * q)
        i += 1
    return 4 * np.pi * (-1) ** (mu + nu + q_lim) * mths.complex_fsum(sum_array)


def outgoing_separation_coefficient(m, mu, n, nu, k, dist):
    r""" Coefficient S^mmu_nnu(b) of separation matrix
    eq(3.97) and eq(3.74) in 'Encyclopedia' """
    if abs(n - nu) >= abs(m - mu):
        q0 = abs(n - nu)
    if (abs(n - nu) < abs(m - mu)) and ((n + nu + abs(m - mu)) % 2 == 0):
        q0 = abs(m - mu)
    if (abs(n - nu) < abs(m - mu)) and ((n + nu + abs(m - mu)) % 2 != 0):
        q0 = abs(m - mu) + 1
    q_lim = (n + nu - q0) // 2
    sum_array = np.zeros(q_lim + 1, dtype=complex)
    i = 0
    for q in range(0, q_lim + 1):
        sum_array[i] = (-1) ** q * outgoing_wvf(m - mu, q0 + 2 * q, dist[0], dist[1], dist[2], k) * \
                       mths.gaunt_coefficient(n, m, nu, -mu, q0 + 2 * q)
        i += 1
    return 4 * np.pi * (-1) ** (mu + nu + q_lim) * mths.complex_fsum(sum_array)
