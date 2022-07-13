import numpy as np
import scipy
import scipy.special
import mathematics as mths


def n_idx(n):
    r""" build np.array of numbers 0,1,1,1,2,2,2,2,2,...,n,..,n """
    return np.repeat(np.arange(n+1), np.arange(n+1) * 2 + 1)


def m_idx(n):
    r""" build np.array of numbers 0,-1,0,1,-2,-1,0,1,2,...,-n,..,n """
    return np.concatenate([np.arange(-i, i + 1) for i in range(n + 1)])


def multipoles(n):
    return zip(m_idx(n), n_idx(n))


def incident_coefficient(m, n, k):
    r""" Coefficients in decomposition of plane wave
    d^m_n - eq(4.40) of 'Encyclopedia' """
    k_abs, k_phi, k_theta = mths.dec_to_sph(k[0], k[1], k[2])
    return 4 * np.pi * 1j ** n * np.conj(scipy.special.sph_harm(m, n, k_phi, k_theta))


def local_incident_coefficient(m, n, k, sph_pos, order):
    r""" Counts local incident coefficients
    d^m_nj - eq(42) in 'Multiple scattering and scattering cross sections P. A. Martin' """
    incident_coefficient_array = np.zeros((order+1) ** 2, dtype=complex)
    i = 0
    for munu in multipoles(order):
        incident_coefficient_array[i] = incident_coefficient(munu[0], munu[1], k) * \
                                        regular_separation_coefficient(munu[0], m, munu[1], n, k, sph_pos)
        i += 1
    return mths.complex_fsum(incident_coefficient_array)


def coefficient_array(n, k, coef, length):
    c_array = np.zeros(((n+1) ** 2), dtype=complex)
    i = 0
    for mn in multipoles(n):
        c_array[i] = coef(mn[0], mn[1], k)
        i += 1
    return np.split(np.repeat(c_array, length), (n + 1) ** 2)


def regular_wave_function(m, n, x, y, z, k):
    r""" Regular basis spherical wave function
    ^psi^m_n - eq(between 4.37 and 4.38) of 'Encyclopedia' """
    k_abs, k_phi, k_theta = mths.dec_to_sph(k[0], k[1], k[2])
    r, phi, theta = mths.dec_to_sph(x, y, z)
    return scipy.special.spherical_jn(n, k_abs * r) * scipy.special.sph_harm(m, n, phi, theta)


def regular_wave_functions_array(n, x, y, z, k):
    r""" builds np.array of all regular wave functions with order <= n"""
    rw_array = np.zeros(((n+1) ** 2, len(x)), dtype=complex)
    i = 0
    for mn in multipoles(n):
        rw_array[i] = regular_wave_function(mn[0], mn[1], x, y, z, k)
        i += 1
    return rw_array


def outgoing_wave_function(m, n, x, y, z, k):
    r""" Outgoing basis spherical wave function
    psi^m_n - eq(between 4.37 and 4.38) in 'Encyclopedia' """
    k_abs, k_phi, k_theta = mths.dec_to_sph(k[0], k[1], k[2])
    r, phi, theta = mths.dec_to_sph(x, y, z)
    return mths.sph_hankel1(n, k_abs * r) * scipy.special.sph_harm(m, n, phi, theta)


def outgoing_wave_functions_array(n, x, y, z, k):
    r""" builds np.array of all outgoing wave functions with order less n"""
    ow_array = np.zeros(((n+1) ** 2, len(x)), dtype=complex)
    i = 0
    for mn in multipoles(n):
        ow_array[i] = outgoing_wave_function(mn[0], mn[1], x, y, z, k)
        i += 1
    return ow_array


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
    for q in range(0, q_lim + 1):
        sum_array[i] = (-1) ** q * regular_wave_function(m - mu, q0 + 2 * q, dist[0], dist[1], dist[2], k) * \
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
        sum_array[i] = (-1) ** q * outgoing_wave_function(m - mu, q0 + 2 * q, dist[0], dist[1], dist[2], k) * \
                       mths.gaunt_coefficient(n, m, nu, -mu, q0 + 2 * q)
        i += 1
    return 4 * np.pi * (-1) ** (mu + nu + q_lim) * mths.complex_fsum(sum_array)
