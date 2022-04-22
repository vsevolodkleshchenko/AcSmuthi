import numpy as np
import numpy.linalg
import scipy
import scipy.special
from sympy.physics.wigner import wigner_3j
import matplotlib.pyplot as plt


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


def inc_coef(m, n, k):
    r""" coefficients in decomposition of plane wave
    d^m_n - eq(4.40) of Encyclopedia
    :param m, n: array_like - order and degree of the harmonic (int)
    :return: array_like (complex float) """
    k_x = k[0]
    k_y = k[1]
    k_z = k[2]
    k_abs = np.sqrt(k_x * k_x + k_y * k_y + k_z * k_z)
    k_phi = np.arctan(k_y / k_x)
    k_theta = np.arccos(k_z / k_abs)
    return 4 * np.pi * 1j ** n * scipy.special.sph_harm(m, n, k_phi, k_theta)


def regular_wvfs(m, n, x, y, z, k):
    """ regular basis spherical wave functions
    ^psi^m_n - eq(between 4.37 and 4.38) of Encyclopedia
    :param m, n: array_like - order and degree of the wave function(int)
    :param x, y, z: array_like - coordinates
    :return: array_like (complex float) """
    k_x = k[0]
    k_y = k[1]
    k_z = k[2]
    k_abs = np.sqrt(k_x * k_x + k_y * k_y + k_z * k_z)
    r = np.sqrt(x * x + y * y + z * z)
    phi = np.arctan(y / x)
    theta = np.arccos(z / r)
    return scipy.special.spherical_jn(n, k_abs * r) * \
           scipy.special.sph_harm(m, n, phi, theta)


def outgoing_wvfs(m, n, x, y, z, k):
    """outgoing basis spherical wave functions
    psi^m_n - eq(between 4.37 and 4.38) of Encyclopedia
    :param m, n: array_like - order and degree of the wave function(int)
    :param x, y, z: array_like - coordinates
    :param k: array_like - absolute value of incident wave vector
    :return: array_like (complex float) """
    k_x = k[0]
    k_y = k[1]
    k_z = k[2]
    k_abs = np.sqrt(k_x * k_x + k_y * k_y + k_z * k_z)
    r = np.sqrt(x * x + y * y + z * z)
    phi = np.arctan(y / x)
    theta = np.arccos(z / r)
    return sph_hankel1(n, k_abs * r) * \
           scipy.special.sph_harm(m, n, phi, theta)


def gaunt_coef(n, m, nu, mu, q):
    r"""Gaunt coefficient: G(n,m;nu,mu;q)
    eq(3.71) in Encyclopedia"""
    s = np.sqrt((2 * n + 1) * (2 * nu + 1) * (2 * q + 1) / 4 / np.pi)
    return (-1) ** (m + mu) * s * complex(wigner_3j(n, 0, nu, 0, q, 0)) * \
           complex(wigner_3j(n, m, nu, mu, q, - m - mu))


def sep_matr_coef(m, mu, n, nu, k, dist):
    """coefficient S^mmu_nnu(b) of separation matrix
    eq(3.97) and eq(3.74) in Encyclopedia"""
    if abs(n - nu) >= abs(m - mu):
        q0 = abs(n - nu)
    if (abs(n - nu) < abs(m - mu)) and ((n + nu + abs(m - mu)) % 2 == 0):
        q0 = abs(m - mu)
    if (abs(n - nu) < abs(m - mu)) and ((n + nu + abs(m - mu)) % 2 != 0):
        q0 = abs(m - mu) + 1
    q_lim = (n + nu - q0) // 2
    sum = 0
    for q in range(0, q_lim + 1, 2):
        sum += (-1) ** q * outgoing_wvfs(m - mu, q0 + 2 * q, dist[0], dist[1], dist[2], k) * \
               gaunt_coef(n, m, nu, -mu, q0 + 2 * q)
    return 4 * np.pi * (-1) ** (mu + nu + q_lim) * sum


def syst_matr(k, ro, spheres, order):
    r""" build T matrix from spheres.pdf"""
    k_x = k[0]
    k_y = k[1]
    k_z = k[2]
    k_abs = np.sqrt(k_x * k_x + k_y * k_y + k_z * k_z)

    k_sph = spheres[0, 0]
    r_sph = spheres[0, 1]
    ro_sph = spheres[0, 2]

    num_of_coef = (order + 1) ** 2
    width = num_of_coef * 2
    height = num_of_coef * 2
    t_matrix = np.zeros((height, width), dtype=complex)
    for n in range(order + 1):
        col_idx_1 = np.arange(n ** 2, (n + 1) ** 2)
        col_idx_2 = col_idx_1 + num_of_coef
        row_idx_1 = np.arange(2 * n ** 2, 2 * (n + 1) ** 2 - 1, 2)
        row_idx_2 = np.arange(2 * n ** 2 + 1, 2 * (n + 1) ** 2, 2)
        t_matrix[row_idx_1, col_idx_1] = - sph_hankel1(n, k_abs * r_sph)
        t_matrix[row_idx_2, col_idx_1] = - sph_hankel1_der(n, k_abs * r_sph)
        t_matrix[row_idx_1, col_idx_2] = scipy.special.spherical_jn(n, k_sph * r_sph)
        t_matrix[row_idx_2, col_idx_2] = ro / ro_sph * scipy.special.spherical_jn(n, k_sph * r_sph, derivative=True)
    return t_matrix


def syst_matr2(k, ro, dist, spheres, order):
    r""" build T matrix for 2 spheres from spheres.pdf"""
    k_x = k[0]
    k_y = k[1]
    k_z = k[2]
    k_abs = np.sqrt(k_x * k_x + k_y * k_y + k_z * k_z)

    k_sph1 = spheres[0,0]
    r_sph1 = spheres[0,1]
    ro_sph1 = spheres[0,2]
    k_sph2 = spheres[1,0]
    r_sph2 = spheres[1,1]
    ro_sph2 = spheres[1,2]

    num_of_coef = (order + 1) ** 2
    num_of_sph = len(spheres)
    width = num_of_coef * 2 * num_of_sph
    height = num_of_coef * 2 * num_of_sph
    t_matrix = np.zeros((height, width), dtype=complex)

    for n in range(order + 1):
        col_idx_1 = np.arange(n ** 2, (n + 1) ** 2)
        col_idx_2 = col_idx_1 + num_of_coef
        col_idx_3 = col_idx_2 + num_of_coef
        col_idx_4 = col_idx_3 + num_of_coef
        row_idx_1 = np.arange(2 * n ** 2, 2 * (n + 1) ** 2 - 1, 2)
        row_idx_2 = np.arange(2 * n ** 2 + 1, 2 * (n + 1) ** 2, 2)
        row_idx_3 = row_idx_1 + num_of_coef * 2
        row_idx_4 = row_idx_2 + num_of_coef * 2

        for m in range(-n, n + 1):
            for nu in range(order + 1):
                for mu in range(-nu, nu + 1):
                    t_matrix[2*(n**2+n+m), num_of_coef*2+nu**2+nu+mu] = - scipy.special.spherical_jn(n, k_abs * r_sph1) * \
                                                                               sep_matr_coef(mu, m, nu, n, k, dist)
                    t_matrix[2*(n**2+n+m)+1, num_of_coef*2+nu**2+nu+mu] = - scipy.special.spherical_jn(n, k_abs * r_sph1, derivative=True) * \
                                                                               sep_matr_coef(mu, m, nu, n, k, dist)
                    t_matrix[num_of_coef*2+2*(n**2+n+m), nu**2+nu+mu] = - scipy.special.spherical_jn(n, k_abs * r_sph2) * \
                                                                               sep_matr_coef(mu, m, nu, n, k, -dist)
                    t_matrix[num_of_coef*2+2*(n**2+n+m)+1, nu**2+nu+mu] = - scipy.special.spherical_jn(n, k_abs * r_sph2, derivative=True) * \
                                                                               sep_matr_coef(mu, m, nu, n, k, -dist)

        t_matrix[row_idx_1, col_idx_1] = - sph_hankel1(n, k_abs * r_sph1)
        t_matrix[row_idx_2, col_idx_1] = - sph_hankel1_der(n, k_abs * r_sph1)

        t_matrix[row_idx_1, col_idx_2] = scipy.special.spherical_jn(n, k_sph1 * r_sph1)
        t_matrix[row_idx_2, col_idx_2] = ro / ro_sph1 * scipy.special.spherical_jn(n, k_sph1 * r_sph1, derivative=True)

        t_matrix[row_idx_3, col_idx_3] = - sph_hankel1(n, k_abs * r_sph1)
        t_matrix[row_idx_4, col_idx_3] = - sph_hankel1_der(n, k_abs * r_sph1)

        t_matrix[row_idx_3, col_idx_4] = scipy.special.spherical_jn(n, k_sph2 * r_sph2)
        t_matrix[row_idx_4, col_idx_4] = ro / ro_sph2 * scipy.special.spherical_jn(n, k_sph2 * r_sph2, derivative=True)
    return t_matrix


def syst_rhs(k, spheres, order):
    r""" build right hand side of system from spheres.pdf"""
    k_x = k[0]
    k_y = k[1]
    k_z = k[2]
    k_abs = np.sqrt(k_x * k_x + k_y * k_y + k_z * k_z)

    num_of_sph = len(spheres)
    num_of_coef = (order + 1) ** 2
    rhs = np.zeros(num_of_coef * 2 * num_of_sph, dtype=complex)
    for n in range(order + 1):
        idx_1 = np.arange(2 * n ** 2, 2 * (n + 1) ** 2 - 1, 2)
        idx_2 = np.arange(2 * n ** 2 + 1, 2 * (n + 1) ** 2, 2)
        m = np.arange(-n, n + 1)
        for sph in range(num_of_sph):
            rhs[idx_1] = inc_coef(m, n, k) * \
                       scipy.special.spherical_jn(n, k_abs * spheres[sph, 1])
            rhs[idx_2] = inc_coef(m, n, k) * \
                       scipy.special.spherical_jn(n, k_abs * spheres[sph, 1], derivative=True)
            idx_1 += num_of_coef * 2
            idx_2 += num_of_coef * 2
    return rhs


def syst_solve(k, ro, dist, spheres, order):
    r""" solve T matrix system and counts a coefficients in decomposition
    of scattered field and field inside the spheres """
    num_of_sph = len(spheres)
    if num_of_sph == 2:
        t_matrix = syst_matr2(k, ro, dist, spheres, order)
    elif num_of_sph == 1:
        t_matrix = syst_matr(k, ro, spheres, order)
    rhs = syst_rhs(k, spheres, order)
    coef = np.linalg.solve(t_matrix, rhs)
    return np.split(coef, 2 * num_of_sph)


def total_field(x, y, z, k, ro, dist, spheres, order):
    """ counts field outside the sphere"""
    num_of_sph = len(spheres)
    coef = syst_solve(k, ro, dist, spheres, order)
    tot_field = 0
    for n in range(order + 1):
        for m in range(-n, n + 1):
            other_sph = 0
            for sph in range(1, num_of_sph):
                sc_coef_sph = coef[2 * sph]
                for nu in range(order + 1):
                    for mu in range(-nu, nu + 1):
                        other_sph += sep_matr_coef(mu, m, nu, n, k, dist) * \
                                           sc_coef_sph[nu ** 2 + nu + mu]
                        # print(sep_matr_coef(mu, m, nu, n, k, dist))
            tot_field += inc_coef(m, n, k) * regular_wvfs(m, n, x, y, z, k) + \
                        coef[0][n ** 2 + n + m] * outgoing_wvfs(m, n, x, y, z, k) + \
                        other_sph * regular_wvfs(m, n, x, y, z, k)
    return tot_field


def xz_plot(span, plane_number, k, ro, dist, spheres, order):
    r"""
    Count field and build a 2D heat-plot in XZ plane for span_y[plane_number]
    --->z """
    span_x = span[0]
    span_y = span[1]
    span_z = span[2]
    grid = np.vstack(np.meshgrid(span_y, span_x, span_z, indexing='ij')).reshape(3, -1).T
    y = grid[:, 0]
    x = grid[:, 1]
    z = grid[:, 2]

    tot_field = np.real(total_field(x, y, z, k, ro, dist, spheres, order))

    xz = np.flip(np.asarray(tot_field[(plane_number - 1) * len(span_x) * len(span_z):
                                (plane_number - 1) * len(span_x) * len(span_z) +
                                len(span_x) * len(span_z)]).reshape(len(span_x), len(span_z)), axis=0)
    fig, ax = plt.subplots()
    ax.imshow(xz, cmap='viridis')
    plt.show()

    # print(tot_field, xz, sep='\n')


def yz_plot(span, plane_number, k, ro, dist, spheres, order):
    r"""
    Count field and build a 2D heat-plot in YZ plane for x[plane_number]
    --->z """
    span_x = span[0]
    span_y = span[1]
    span_z = span[2]
    grid = np.vstack(np.meshgrid(span_x, span_y, span_z, indexing='ij')).reshape(3, -1).T
    x = grid[:, 0]
    y = grid[:, 1]
    z = grid[:, 2]

    tot_field = np.real(total_field(x, y, z, k, ro, dist, spheres, order))

    yz = np.flip(np.asarray(tot_field[(plane_number - 1) * len(span_y) * len(span_z):
                                (plane_number - 1) * len(span_y) * len(span_z) +
                                len(span_y) * len(span_z)]).reshape(len(span_y), len(span_z)), axis=0)

    fig, ax = plt.subplots()
    ax.imshow(yz, cmap='viridis')
    plt.show()

    # print(grid, x, y, z, total_field, yz, sep='\n')


def xy_plot(span, plane_number, k, ro, dist, spheres, order):
    r"""
    Count field and build a 2D heat-plot in XZ plane for z[plane_number]
    --->y """
    span_x = span[0]
    span_y = span[1]
    span_z = span[2]
    grid = np.vstack(np.meshgrid(span_z, span_x, span_y, indexing='ij')).reshape(3, -1).T
    z = grid[:, 0]
    x = grid[:, 1]
    y = grid[:, 2]

    tot_field = np.real(total_field(x, y, z, k, ro, dist, spheres, order))

    xy = np.flip(np.asarray(tot_field[(plane_number-1)*len(span_x)*len(span_y):
                                (plane_number-1)*len(span_x)*len(span_y) +
                                len(span_x)*len(span_y)]).reshape(len(span_x), len(span_y)), axis=0)
    fig, ax = plt.subplots()
    ax.imshow(xy, cmap='viridis')
    plt.show()

    # print(grid, x, y, z, total_field, xy, sep='\n')


def simulation():
    # coordinates
    number_of_points = 30
    span_x = np.linspace(-100, 100, number_of_points)
    span_y = np.linspace(-100, 100, number_of_points)
    span_z = np.linspace(-100, 100, number_of_points)
    span = np.array([span_x, span_y, span_z])

    # parameters of fluid
    ro = 1.225

    # parameters of the spheres
    k_sph1 = 0.015
    r_sph1 = 3
    ro_sph1 = 1011
    sphere1 = np.array([k_sph1, r_sph1, ro_sph1])
    k_sph2 = 0.016
    r_sph2 = 5
    ro_sph2 = 1011
    sphere2 = np.array([k_sph2, r_sph2, ro_sph2])

    # parameters of configuration
    dist_x = 2
    dist_y = 10
    dist_z = 2

    # choose simulation 1 or 2
    # simulation 1
    # spheres = np.array([sphere1])
    # dist = np.array([])
    # simulation 2
    spheres = np.array([sphere1, sphere2])
    dist = np.array([dist_x, dist_y, dist_z])

    # parameters of the field
    k_x = 0.009
    k_y = 0.001
    k_z = 0.2
    k = np.array([k_x, k_y, k_z])

    # order of decomposition
    order = 8

    plane_number = int(number_of_points / 2) + 1
    yz_plot(span, plane_number, k, ro, dist, spheres, order)


simulation()
