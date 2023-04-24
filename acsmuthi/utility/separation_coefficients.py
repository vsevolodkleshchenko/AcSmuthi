import numpy as np
import pywigxjpf as wig

from acsmuthi.utility import wavefunctions as wvfs

wig.wig_table_init(60, 3)
wig.wig_temp_init(60)


def gaunt_coefficient(n, m, nu, mu, q):
    r"""Gaunt coefficient: G(n,m;nu,mu;q)"""
    s = np.sqrt((2 * n + 1) * (2 * nu + 1) * (2 * q + 1) / 4 / np.pi)
    return (-1.) ** (m + mu) * s * wig.wig3jj(2*n, 2*nu, 2*q, 0, 0, 0) * \
        wig.wig3jj(2*n, 2*nu, 2*q, 2*m, 2*mu, - 2*m - 2*mu)


def regular_separation_coefficient(m, mu, n, nu, k, dist):
    r"""Coefficient ^S^mmu_nnu(b) of separation matrix"""
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
        sum_array[i] = (-1) ** q * wvfs.regular_wvf(m - mu, q0 + 2 * q, dist[0], dist[1], dist[2], k) * \
                       gaunt_coefficient(n, m, nu, -mu, q0 + 2 * q)
        i += 1
    return 4 * np.pi * (-1) ** (mu + nu + q_lim) * np.sum(sum_array)


def outgoing_separation_coefficient(m, mu, n, nu, k, dist):
    r"""Coefficient S^mmu_nnu(b) of separation matrix"""
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
        sum_array[i] = (-1) ** q * wvfs.outgoing_wvf(m - mu, q0 + 2 * q, dist[0], dist[1], dist[2], k) * \
                       gaunt_coefficient(n, m, nu, -mu, q0 + 2 * q)
        i += 1
    return 4 * np.pi * (-1) ** (mu + nu + q_lim) * np.sum(sum_array)
