import numpy as np
import numpy.linalg
import scipy
import scipy.special
from sympy.physics.wigner import clebsch_gordan
import matplotlib.pyplot as plt
import time



def neyman1(n, z):
    r""" Neyman function of the first kind
    :param n: array_like - order (float)
    :param z: array_like - argument (float or complex)
    :return: array_like """
    return -1j * scipy.special.hankel1(n, z) + 1j * scipy.special.jv(n, z)


def sph_neyman(n, z):
    r""" spherical Neyman function
    :param n: array_like - order (float)
    :param z: array_like - argument (float or complex)
    :return: array_like """
    return (-1) ** (n + 1) * np.sqrt(np.pi / 2 / z) * scipy.special.jv(-n - 0.5, z)


def sph_neyman_der(n, z):
    r""" first derivative of spherical Neyman function
    :param n: array_like - order (float)
    :param z: array_like - argument (float or complex)
    :return: array_like """
    return (-1) ** n * np.sqrt(np.pi / (8 * z ** 3)) * scipy.special.jv(-n - 0.5, z) + \
           (-1) ** (n + 1) * np.sqrt(np.pi / 2 / z) * scipy.special.jvp(-n - 0.5, z)


def sph_hankel1(n, z):
    r""" spherical Hankel function of the first kind
    :param n: array_like - order (float)
    :param z: array_like - argument (float or complex)
    :return: array_like """
    return scipy.special.spherical_jn(n, z) + 1j * sph_neyman(n, z)


def sph_hankel1_der(n, z):
    """ first derivative of spherical Hankel function of the first kind
    :param n: array_like - order (float)
    :param z: array_like - argument (float or complex)
    :return: array_like"""
    return scipy.special.spherical_jn(n, z, derivative=True) + 1j * sph_neyman_der(n, z)

def inc_coef(m, n, k_x, k_y, k_z):
    r""" coefficients in decomposition of plane wave
    d^m_n - eq(4.40) of Encyclopedia
    :param m, n: array_like - order and degree of the harmonic (int)
    :param k_x, k_y, k_z: array_like coordinates of incident wave vector
    :return: array_like (complex float) """
    k = np.sqrt(k_x * k_x + k_y * k_y + k_z * k_z)
    k_phi = np.arctan(k_y / k_x)
    k_theta = np.arccos(k_z / k)
    return 4 * np.pi * 1j ** n * scipy.special.sph_harm(m, n, k_phi, k_theta)


def regular_wvfs(m, n, x, y, z, k):
    """ regular basis spherical wave functions
    ^psi^m_n - eq(between 4.37 and 4.38) of Encyclopedia
    :param m, n: array_like - order and degree of the wave function(int)
    :param x, y, z: array_like - coordinates
    :param k: array_like - absolute value of incident wave vector
    :return: array_like (complex float) """
    r = np.sqrt(x * x + y * y + z * z)
    phi = np.arctan(x / y)
    theta = np.arccos(z / r)
    return scipy.special.spherical_jn(n, k * r) * \
           scipy.special.sph_harm(m, n, theta, phi)


def outgoing_wvfs(m, n, x, y, z, k):
    """outgoing basis spherical wave functions
    psi^m_n - eq(between 4.37 and 4.38) of Encyclopedia
    :param m, n: array_like - order and degree of the wave function(int)
    :param x, y, z: array_like - coordinates
    :param k: array_like - absolute value of incident wave vector
    :return: array_like (complex float) """
    r = np.sqrt(x * x + y * y + z * z)
    phi = np.arctan(x / y)
    theta = np.arccos(z / r)
    return sph_hankel1(n, k * r) * \
           scipy.special.sph_harm(m, n, theta, phi)


def gaunt_coef(n, m, nu, mu, q):
    r"""Gaunt coefficient: G(n,m;nu,mu;q)
    p.329 in Encyclopedia"""
    return np.sqrt((2 * n + 1) * (2 * nu + 1) / (4 * np.pi ) / (2 * q + 1)) * \
           clebsch_gordan(n, 0, nu, 0, q, 0) * clebsch_gordan(n, m, nu, mu, q, m + mu)


def sep_matr_coef(m, mu, n, nu, k, dist_x, dist_y, dist_z, order):
    """coefficient S^mmu_nnu(b) of separation matrix
    eq(3.86) in Encyclopedia"""
    dist = np.sqrt(dist_x * dist_x + dist_y * dist_y + dist_z * dist_z)
    dist_phi = np.arctan(dist_x / dist_y)
    dist_theta = np.arccos(dist_z . dist)
    sum = 0
    for q in range(order):
        sum += 1j ** q * sph_hankel1(q, k * dist) * \
               np.conj(scipy.special.sph_harm(mu - m, q, dist_theta, dist_phi)) * \
               gaunt_coef(n, m, q, mu - m, nu)
    return 4 * np.pi * 1j ** (nu - n) * sum


def syst_matr(k, ro, k_sph, r_sph, ro_sph, order):
    r""" build T matrix from spheres.pdf"""
    num_of_coef = (order + 1) ** 2
    width = num_of_coef * 2
    height = num_of_coef * 2
    t_matrix = np.zeros((height, width), dtype=complex)
    for n in range(order + 1):
        col_idx_1 = np.arange(n ** 2, (n + 1) ** 2)
        col_idx_2 = col_idx_1 + num_of_coef
        row_idx_1 = np.arange(2 * n ** 2, 2 * (n + 1) ** 2 - 1, 2)
        row_idx_2 = np.arange(2 * n ** 2 + 1, 2 * (n + 1) ** 2, 2)
        t_matrix[row_idx_1, col_idx_1] = - sph_hankel1(n, k * r_sph)
        t_matrix[row_idx_2, col_idx_1] = - sph_hankel1_der(n, k * r_sph)
        t_matrix[row_idx_1, col_idx_2] = scipy.special.spherical_jn(n, k_sph * r_sph)
        t_matrix[row_idx_2, col_idx_2] = ro / ro_sph * scipy.special.spherical_jn(n, k_sph * r_sph, derivative=True)
    return t_matrix


def syst_rhs(k_x, k_y, k_z, r_sph, order):
    r""" build right hand side of system from spheres.pdf"""
    k = np.sqrt(k_x * k_x + k_y * k_y + k_z * k_z)
    num_of_coef = (order + 1) ** 2
    rhs = np.zeros(num_of_coef * 2, dtype=complex)
    for n in range(order + 1):
        idx_1 = np.arange(2 * n ** 2, 2 * (n + 1) ** 2 - 1, 2)
        idx_2 = np.arange(2 * n ** 2 + 1, 2 * (n + 1) ** 2, 2)
        m = np.arange(-n, n + 1)
        # m2 = np.concatenate(np.array([m, m]).T)
        rhs[idx_1] = inc_coef(m, n, k_x, k_y, k_z) * \
                   scipy.special.spherical_jn(n, k * r_sph)
        rhs[idx_2] = inc_coef(m, n, k_x, k_y, k_z) * \
                   scipy.special.spherical_jn(n, k * r_sph, derivative=True)
    return rhs


def syst_solve(k_x, k_y, k_z, ro, k_sph, r_sph, ro_sph, order):
    r""" solve T matrix system and counts a coefficients in decomposition
    of scattered field and field inside the sphere """
    num_of_coef = (order + 1) ** 2
    k = np.sqrt(k_x * k_x + k_y * k_y + k_z * k_z)
    t_matrix = syst_matr(k, ro, k_sph, r_sph, ro_sph, order)
    rhs = syst_rhs(k_x, k_y, k_z, r_sph, order)
    coef = np.linalg.solve(t_matrix, rhs)
    sc_coef = coef[:num_of_coef]
    in_coef = coef[num_of_coef:]
    return sc_coef, in_coef


# def sc_coef(m, n, k_x, k_y, k_z, ro, k_sph, r_sph, ro_sph, order):
#     sc_coef = syst_solve(k_x, k_y, k_z, ro, k_sph, r_sph, ro_sph, order)[0]
#     return sc_coef[n ** 2 + n + m]


def total_field(x, y, z, k_x, k_y, k_z, ro, k_sph, r_sph, ro_sph, order):
    """ counts field outside the sphere"""
    k = np.sqrt(k_x * k_x + k_y * k_y + k_z * k_z)
    sc_coef = syst_solve(k_x, k_y, k_z, ro, k_sph, r_sph, ro_sph, order)[0]
    tot_field = 0
    for n in range(order + 1):
        for m in range(-n, n + 1):
            tot_field += inc_coef(m, n, k_x, k_y, k_z) * regular_wvfs(m, n, x, y, z, k) + \
                         sc_coef[n ** 2 + n + m] * outgoing_wvfs(m, n, x, y, z, k)
    return tot_field


def xz_plot(span_x, span_y, span_z, plane_number, k_x, k_y, k_z, ro, k_sph, r_sph, ro_sph, order):
    r"""
    Count field and build a 2D heat-plot in XZ plane for span_y[plane_number]
    --->z """
    grid = np.vstack(np.meshgrid(span_y, span_x, span_z, indexing='ij')).reshape(3, -1).T
    y = grid[:, 0]
    x = grid[:, 1]
    z = grid[:, 2]

    tot_field = np.abs(total_field(x, y, z, k_x, k_y, k_z, ro, k_sph, r_sph, ro_sph, order))

    xz = np.flip(np.asarray(tot_field[(plane_number - 1) * len(span_x) * len(span_z):
                                (plane_number - 1) * len(span_x) * len(span_z) +
                                len(span_x) * len(span_z)]).reshape(len(span_x), len(span_z)), axis=0)
    fig, ax = plt.subplots()
    ax.imshow(xz, cmap='viridis')
    plt.show()

    # print(grid, x, y, z, total_field, xz, sep='\n')


def yz_plot(span_x, span_y, span_z, plane_number, k_x, k_y, k_z, ro, k_sph, r_sph, ro_sph, order):
    r"""
    Count field and build a 2D heat-plot in YZ plane for x[plane_number]
    --->z """
    grid = np.vstack(np.meshgrid(span_x, span_y, span_z, indexing='ij')).reshape(3, -1).T
    x = grid[:, 0]
    y = grid[:, 1]
    z = grid[:, 2]

    tot_field = np.abs(total_field(x, y, z, k_x, k_y, k_z, ro, k_sph, r_sph, ro_sph, order))

    yz = np.flip(np.asarray(tot_field[(plane_number - 1) * len(span_y) * len(span_z):
                                (plane_number - 1) * len(span_y) * len(span_z) +
                                len(span_y) * len(span_z)]).reshape(len(span_y), len(span_z)), axis=0)

    fig, ax = plt.subplots()
    ax.imshow(yz, cmap='viridis')
    plt.show()

    # print(grid, x, y, z, total_field, yz, sep='\n')


def xy_plot(span_x, span_y, span_z, plane_number, k_x, k_y, k_z, ro, k_sph, r_sph, ro_sph, order):
    r"""
    Count field and build a 2D heat-plot in XZ plane for z[plane_number]
    --->y """
    grid = np.vstack(np.meshgrid(span_z, span_x, span_y, indexing='ij')).reshape(3, -1).T
    z = grid[:, 0]
    x = grid[:, 1]
    y = grid[:, 2]

    tot_field = np.abs(total_field(x, y, z, k_x, k_y, k_z, ro, k_sph, r_sph, ro_sph, order))

    xy = np.flip(np.asarray(tot_field[(plane_number-1)*len(span_x)*len(span_y):
                                (plane_number-1)*len(span_x)*len(span_y)+
                                len(span_x)*len(span_y)]).reshape(len(span_x), len(span_y)), axis=0)
    fig, ax = plt.subplots()
    ax.imshow(xy, cmap='viridis')
    plt.show()

    # print(grid, x, y, z, total_field, xy, sep='\n')


def simulation():
    # coordinates
    number_of_points = 20
    span_x = np.linspace(-20, 20, number_of_points)
    span_y = np.linspace(-20, 20, number_of_points)
    span_z = np.linspace(-20, 20, number_of_points)

    # parameters of fluid
    ro = 4

    # parameters of the sphere
    k_sph = 1
    r_sph = 3
    ro_sph = 3

    # parameters of the field
    k_x = 0.2
    k_y = 0.0
    k_z = 2.3

    # order of decomposition
    order = 3

    plane_number = int(number_of_points / 2) + 1
    yz_plot(span_x, span_y, span_z, plane_number, k_x, k_y, k_z, ro, k_sph, r_sph, ro_sph, order)


simulation()

# print(total_field(1,2,3,4,2,3,5,1,3,5,6))
# solve_syst(2, 3, 1, 2, 3, 4, 5, 2)
# print(t_matrix(3,1,2,6,4,2))
# print(rhs_for_tmatr_syst(3,1,2,6,2))
# print(syst_solve(2, 3, 1, 2, 3, 4, 5, 2))