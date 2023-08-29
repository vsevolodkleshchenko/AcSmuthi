#cython: language_level=3
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import scipy.special
import numpy as np
import scipy.special.cython_special as ss

import pywigxjpf
pywigxjpf.wig_table_init(60, 3)
pywigxjpf.wig_temp_init(60)
wig3jj = pywigxjpf.wig3jj

cimport numpy as np
cimport scipy.special.cython_special as ss
cimport cython


cdef double complex sph_hankel1(long n, double z):
    return ss.spherical_jn(n, z) + 1j * ss.spherical_yn(n, z)


cdef double complex regular_wvf(long m, long n, double x, double y, double z, double k):
    cdef double r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    cdef double theta = np.arccos(z / r)
    cdef double phi = np.arctan2(y, x)
    return ss.spherical_jn(n, k * r) * ss.sph_harm(m, n, phi, theta)


cdef double complex outgoing_wvf(long m, long n, double x, double y, double z, double k):
    cdef double r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    cdef double theta = np.arccos(z / r)
    cdef double phi = np.arctan2(y, x)
    return sph_hankel1(n, k * r) * ss.sph_harm(m, n, phi, theta)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double gaunt_coefficient(long n, long m, long nu, long mu, long q):
    r"""Gaunt coefficient: G(n,m;nu,mu;q)"""
    cdef double s = np.sqrt((2 * n + 1) * (2 * nu + 1) * (2 * q + 1) / 4 / np.pi)
    return (-1.) ** (m + mu) * s * wig3jj(2*n, 2*nu, 2*q, 0, 0, 0) * \
           wig3jj(2*n, 2*nu, 2*q, 2*m, 2*mu, - 2*m - 2*mu)


###########################
### Free space coupling ###
###########################


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex regular_separation_coefficient(long m, long mu, long n, long nu, double k, double[:] dist):
    cdef long q0, q_lim, q, i
    if abs(n - nu) >= abs(m - mu):
        q0 = abs(n - nu)
    if (abs(n - nu) < abs(m - mu)) and ((n + nu + abs(m - mu)) % 2 == 0):
        q0 = abs(m - mu)
    if (abs(n - nu) < abs(m - mu)) and ((n + nu + abs(m - mu)) % 2 != 0):
        q0 = abs(m - mu) + 1
    q_lim = (n + nu - q0) // 2
    cdef double complex[:] sum_array = np.zeros(q_lim + 1, dtype=np.complex128)
    i = 0
    for q in range(0, q_lim + 1):
        sum_array[i] = (-1) ** q * regular_wvf(m - mu, q0 + 2 * q, dist[0], dist[1], dist[2], k) * \
                       gaunt_coefficient(n, m, nu, -mu, q0 + 2 * q)
        i += 1
    return 4 * np.pi * (-1) ** (mu + nu + q_lim) * np.sum(sum_array)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex outgoing_separation_coefficient(long m, long mu, long n, long nu, double k, double[:] dist):
    cdef long q0, q_lim, q, i
    if abs(n - nu) >= abs(m - mu):
        q0 = abs(n - nu)
    if (abs(n - nu) < abs(m - mu)) and ((n + nu + abs(m - mu)) % 2 == 0):
        q0 = abs(m - mu)
    if (abs(n - nu) < abs(m - mu)) and ((n + nu + abs(m - mu)) % 2 != 0):
        q0 = abs(m - mu) + 1
    q_lim = (n + nu - q0) // 2
    cdef double complex[:] sum_array = np.zeros(q_lim + 1, dtype=np.complex128)
    i = 0
    for q in range(0, q_lim + 1):
        sum_array[i] = (-1) ** q * outgoing_wvf(m - mu, q0 + 2 * q, dist[0], dist[1], dist[2], k) * \
                       gaunt_coefficient(n, m, nu, -mu, q0 + 2 * q)
        i += 1
    return 4 * np.pi * (-1) ** (mu + nu + q_lim) * np.sum(sum_array)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex[:, :] _coupling_block(double[:] particle_pos, double[:] other_particle_pos, double k_medium, long order):
    cdef double complex[:, :] block = np.zeros(((order + 1) ** 2, (order + 1) ** 2), dtype=complex)
    cdef long m, n, mu, nu, imn, imunu
    cdef double[:] distance = np.asarray(particle_pos) - np.asarray(other_particle_pos)
    for n in range(order + 1):
        for m in range(-n, n + 1):
            imn = n ** 2 + n + m
            for nu in range(order + 1):
                for mu in range(-nu, nu + 1):
                    imunu = nu ** 2 + nu + mu
                    block[imn, imunu] = outgoing_separation_coefficient(mu, m, nu, n, k_medium, distance)
    return block


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex[:, :] _translation_block(long order, double k_medium, double[:] distance):
    cdef double complex[:, :] d = np.zeros(((order + 1) ** 2, (order + 1) ** 2), dtype=complex)
    cdef long m, n, mu, nu, imn, imunu
    for n in range(order + 1):
        for m in range(-n, n + 1):
            imn = n ** 2 + n + m
            for nu in range(order + 1):
                for mu in range(-nu, nu + 1):
                    imunu = nu ** 2 + nu + mu
                    d[imn, imunu] = regular_separation_coefficient(mu, m, nu, n, k_medium, distance)
    return d


def coupling_block(double[:] particle_pos, double[:] other_particle_pos, double k_medium, long order):
    return _coupling_block(particle_pos, other_particle_pos, k_medium, order)


def translation_block(long order, double k_medium, double[:] distance):
    return _translation_block(order, k_medium, distance)


###########################
### Substrate coupling ####
###########################


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex substrate_coupling_element(long m, long n, long mu, long nu, double k, double[:] emitter_pos, double[:] receiver_pos):
    cdef long q0, q_lim, q, i
    cdef double ds, dx, dy, dz

    ds = np.abs(emitter_pos[2])
    dx, dy, dz = receiver_pos[0] - emitter_pos[0], receiver_pos[1] - emitter_pos[1], receiver_pos[2] - emitter_pos[2] + 2 * ds

    if abs(n - nu) >= abs(m - mu):
        q0 = abs(n - nu)
    if (abs(n - nu) < abs(m - mu)) and ((n + nu + abs(m - mu)) % 2 == 0):
        q0 = abs(m - mu)
    if (abs(n - nu) < abs(m - mu)) and ((n + nu + abs(m - mu)) % 2 != 0):
        q0 = abs(m - mu) + 1
    q_lim = (n + nu - q0) // 2

    cdef double complex[:] sum_array = np.zeros(q_lim + 1, dtype=np.complex128)

    i = 0
    for q in range(0, q_lim + 1):
        sum_array[i] = 1j ** (q0 + 2 * q) * outgoing_wvf(m - mu, q0 + 2 * q, dx, dy, dz, k) * \
                       gaunt_coefficient(n, m, nu, -mu, q0 + 2 * q)
        i += 1

    return 4 * np.pi * 1j ** (nu - n) * (-1.) ** (n + m + mu) * np.sum(sum_array)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex[:, :] _substrate_coupling_block(double[:] receiver_pos, double[:] emitter_pos, double k, long order):
    cdef double complex[:, :] block = np.zeros(((order + 1) ** 2, (order + 1) ** 2), dtype=complex)
    cdef long m, n, mu, nu, imn, imunu

    for n in range(order + 1):
        for m in range(-n, n + 1):
            imn = n ** 2 + n + m
            for nu in range(order + 1):
                for mu in range(-nu, nu + 1):
                    imunu = nu ** 2 + nu + mu
                    block[imn, imunu] = substrate_coupling_element(mu, nu, m, n, k, emitter_pos, receiver_pos)
    return block


def substrate_coupling_block(double[:] receiver_pos, double[:] emitter_pos, double k, long order):
    return _substrate_coupling_block(receiver_pos, emitter_pos, k, order)
