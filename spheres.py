import numpy as np
import numpy.linalg
import scipy
import scipy.special
from sympy.physics.wigner import wigner_3j
import matplotlib.pyplot as plt
import multiprocessing as mp
import time


def dec_to_sph(x, y, z):
    # a little slower but true
    e = 1e-3
    r = np.sqrt(x * x + y * y + z * z)
    phi = np.zeros(np.size(r))
    theta = np.zeros(np.size(r))
    theta = np.where(r != 0, np.arccos(z / r), theta)
    phi = np.where((x > e) & (y > e), np.arctan(y / x), phi)
    phi = np.where((x < -e) & (y > e), np.pi - np.arctan(- y / x), phi)
    phi = np.where((x < -e) & (y < -e), np.pi + np.arctan(y / x), phi)
    phi = np.where((x > e) & (y < -e), 2 * np.pi - np.arctan(- y / x), phi)
    phi = np.where((np.abs(x) <= e) & (y > e), np.pi, phi)
    phi = np.where((np.abs(x) <= e) & (y < -e), 3 * np.pi / 2, phi)
    phi = np.where((np.abs(y) <= e) & (x < -e), np.pi, phi)
    # faster but wrong
    # r = np.sqrt(x * x + y * y + z * z)
    # phi = np.arctan(y / x)
    # theta = np.arccos(z / r)
    return r, phi, theta


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
    k_abs, k_phi, k_theta = dec_to_sph(k[0], k[1], k[2])
    return 4 * np.pi * 1j ** n * np.conj(scipy.special.sph_harm(m, n, k_phi, k_theta))


def regular_wvfs(m, n, x, y, z, k):
    """ regular basis spherical wave functions
    ^psi^m_n - eq(between 4.37 and 4.38) of Encyclopedia
    :param m, n: array_like - order and degree of the wave function(int)
    :param x, y, z: array_like - coordinates
    :return: array_like (complex float) """
    k_abs, k_phi, k_theta = dec_to_sph(k[0], k[1], k[2])
    r, phi, theta = dec_to_sph(x, y, z)
    return scipy.special.spherical_jn(n, k_abs * r) * \
           scipy.special.sph_harm(m, n, phi, theta)


def outgoing_wvfs(m, n, x, y, z, k):
    """outgoing basis spherical wave functions
    psi^m_n - eq(between 4.37 and 4.38) of Encyclopedia
    :param m, n: array_like - order and degree of the wave function(int)
    :param x, y, z: array_like - coordinates
    :param k: array_like - absolute value of incident wave vector
    :return: array_like (complex float) """
    k_abs, k_phi, k_theta = dec_to_sph(k[0], k[1], k[2])
    r, phi, theta = dec_to_sph(x, y, z)
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


def syst_matr_L(k, ro, pos, spheres, order):
    k_abs = dec_to_sph(k[0], k[1], k[2])[0]
    num_of_coef = (order + 1) ** 2
    block_width = num_of_coef * 2
    block_height = num_of_coef * 2
    num_of_sph = len(spheres)
    t_matrix = np.zeros((block_height * num_of_sph, block_width * num_of_sph), dtype=complex)
    all_spheres = np.arange(num_of_sph)
    for sph in all_spheres:
        k_sph = spheres[sph, 0]
        r_sph = spheres[sph, 1]
        ro_sph = spheres[sph, 2]
        for n in range(order + 1):
            # diagonal block
            col_idx_1 = np.arange(sph * block_width + n ** 2, sph * block_width + (n + 1) ** 2)
            col_idx_2 = col_idx_1 + num_of_coef
            row_idx_1 = np.arange(sph * block_height + 2 * n ** 2, sph * block_height + 2 * (n + 1) ** 2 - 1, 2)
            row_idx_2 = np.arange(sph * block_height + 2 * n ** 2 + 1, sph * block_height + 2 * (n + 1) ** 2, 2)
            t_matrix[row_idx_1, col_idx_1] = - sph_hankel1(n, k_abs * r_sph)
            t_matrix[row_idx_2, col_idx_1] = - sph_hankel1_der(n, k_abs * r_sph)
            t_matrix[row_idx_1, col_idx_2] = scipy.special.spherical_jn(n, k_sph * r_sph)
            t_matrix[row_idx_2, col_idx_2] = ro / ro_sph * scipy.special.spherical_jn(n, k_sph * r_sph, derivative=True)

            # not diagonal block
            other_sph = np.where(all_spheres != sph)[0]
            for osph in other_sph:
                for m in range(-n, n + 1):
                    for nu in range(order + 1):
                        for mu in range(-nu, nu + 1):
                            t_matrix[sph * block_height + 2 * (n ** 2 + n + m),
                                     osph * block_width + nu ** 2 + nu + mu] = \
                                -scipy.special.spherical_jn(n, k_abs * r_sph) * \
                                sep_matr_coef(mu, m, nu, n, k, pos[osph] - pos[sph])
                            t_matrix[sph * block_height + 2 * (n ** 2 + n + m) + 1,
                                     osph * block_width + nu ** 2 + nu + mu] = \
                                -scipy.special.spherical_jn(n, k_abs * r_sph, derivative=True) * \
                                sep_matr_coef(mu, m, nu, n, k, pos[osph] - pos[sph])
    return t_matrix


def syst_rhs(k, spheres, order):
    r""" build right hand side of system from spheres.pdf"""
    k_abs = dec_to_sph(k[0], k[1], k[2])[0]
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


def syst_solve(k, ro, pos, spheres, order):
    r""" solve T matrix system and counts a coefficients in decomposition
    of scattered field and field inside the spheres """
    num_of_sph = len(spheres)
    t_matrix = syst_matr_L(k, ro, pos, spheres, order)
    rhs = syst_rhs(k, spheres, order)
    coef = np.linalg.solve(t_matrix, rhs)
    return np.split(coef, 2 * num_of_sph)


def total_field(x, y, z, k, ro, pos, spheres, order):
    """ counts field outside the sphere"""
    coef = syst_solve(k, ro, pos, spheres, order)
    tot_field = 0
    scat_field = 0
    for n in range(order + 1):
        for m in range(-n, n + 1):
            for sph in range(len(spheres)):
                scat_field += coef[2 * sph][n ** 2 + n + m] * \
                             outgoing_wvfs(m, n, x - pos[sph][0], y - pos[sph][1], z - pos[sph][2], k)
    tot_field += np.exp(1j * (k[0] * x + k[1] * y + k[2] * z)) + scat_field
    return tot_field


def yz_old(span, plane_number, k, ro, pos, spheres, order):
    r"""
    OLD 2D heat-plot in YZ plane for x[plane_number]
    --->z """
    span_x, span_y, span_z = span[0], span[1], span[2]
    grid = np.vstack(np.meshgrid(span_x, span_y, span_z, indexing='ij')).reshape(3, -1).T
    x, y, z = grid[:, 0], grid[:, 1], grid[:, 2]

    tot_field = np.real(total_field(x, y, z, k, ro, pos, spheres, order))

    for sph in range(len(spheres)):
        x_min = pos[sph, 0] - spheres[sph, 1]
        y_min = pos[sph, 1] - spheres[sph, 1]
        z_min = pos[sph, 2] - spheres[sph, 1]
        x_max = pos[sph, 0] + spheres[sph, 1]
        y_max = pos[sph, 1] + spheres[sph, 1]
        z_max = pos[sph, 2] + spheres[sph, 1]
        tot_field = np.where((x >= x_min) & (x <= x_max) &
                             (y >= y_min) & (y <= y_max) &
                             (z >= z_min) & (z <= z_max), 0, tot_field)

    yz = np.asarray(tot_field[(plane_number - 1) * len(span_y) * len(span_z):
                              (plane_number - 1) * len(span_y) * len(span_z) +
                              len(span_y) * len(span_z)]).reshape(len(span_y), len(span_z))
    fig, ax = plt.subplots()
    plt.xlabel('z axis')
    plt.ylabel('y axis')
    im = ax.imshow(yz, cmap='viridis', origin='lower',
                   extent=[span_z.min(), span_z.max(), span_y.min(), span_y.max()])
    plt.colorbar(im)
    plt.show()


def xz_plot(span, plane_number, k, ro, pos, spheres, order):
    r"""
    Count field and build a 2D heat-plot in XZ plane for span_y[plane_number]
    --->z """
    span_x, span_y, span_z = span[0], span[1], span[2]
    grid = np.vstack(np.meshgrid(span_y, span_x, span_z, indexing='ij')).reshape(3, -1).T
    y, x, z = grid[:, 0], grid[:, 1], grid[:, 2]

    x_p = x[(plane_number - 1) * len(span_x) * len(span_z):
            (plane_number - 1) * len(span_x) * len(span_z) + len(span_x) * len(span_z)]
    y_p = y[(plane_number - 1) * len(span_x) * len(span_z):
            (plane_number - 1) * len(span_x) * len(span_z) + len(span_x) * len(span_z)]
    z_p = z[(plane_number - 1) * len(span_x) * len(span_z):
            (plane_number - 1) * len(span_x) * len(span_z) + len(span_x) * len(span_z)]

    tot_field = np.real(total_field(x_p, y_p, z_p, k, ro, pos, spheres, order))

    for sph in range(len(spheres)):
        rx, ry, rz = x_p - pos[sph, 0], y_p - pos[sph, 1], z_p - pos[sph, 2]
        r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        tot_field = np.where(r < spheres[sph, 1], 0, tot_field)

    xz = tot_field.reshape(len(span_y), len(span_z))

    fig, ax = plt.subplots()
    plt.xlabel('z axis')
    plt.ylabel('x axis')
    im = ax.imshow(xz, cmap='viridis', origin='lower',
                   extent=[span_z.min(), span_z.max(), span_x.min(), span_x.max()])
    plt.colorbar(im)
    plt.show()


def yz_plot(span, plane_number, k, ro, pos, spheres, order):
    r"""
    Count field and build a 2D heat-plot in YZ plane for x[plane_number]
    --->z """
    span_x, span_y, span_z = span[0], span[1], span[2]
    grid = np.vstack(np.meshgrid(span_x, span_y, span_z, indexing='ij')).reshape(3, -1).T
    x, y, z = grid[:, 0], grid[:, 1], grid[:, 2]

    x_p = x[(plane_number - 1) * len(span_y) * len(span_z):
                              (plane_number - 1) * len(span_y) * len(span_z) + len(span_y) * len(span_z)]
    y_p = y[(plane_number - 1) * len(span_y) * len(span_z):
                              (plane_number - 1) * len(span_y) * len(span_z) + len(span_y) * len(span_z)]
    z_p = z[(plane_number - 1) * len(span_y) * len(span_z):
                              (plane_number - 1) * len(span_y) * len(span_z) + len(span_y) * len(span_z)]

    tot_field = np.real(total_field(x_p, y_p, z_p, k, ro, pos, spheres, order))

    for sph in range(len(spheres)):
        rx, ry, rz = x_p - pos[sph, 0], y_p - pos[sph, 1], z_p - pos[sph, 2]
        r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        tot_field = np.where(r < spheres[sph, 1], 0, tot_field)

    yz = tot_field.reshape(len(span_y), len(span_z))

    fig, ax = plt.subplots()
    plt.xlabel('z axis')
    plt.ylabel('y axis')
    im = ax.imshow(yz, cmap='viridis', origin='lower',
                   extent=[span_z.min(), span_z.max(), span_y.min(), span_y.max()])
    plt.colorbar(im)
    plt.show()


def xy_plot(span, plane_number, k, ro, pos, spheres, order):
    r"""
    Count field and build a 2D heat-plot in XY plane for span_z[plane_number]
    --->y """
    span_x, span_y, span_z = span[0], span[1], span[2]
    grid = np.vstack(np.meshgrid(span_z, span_x, span_y, indexing='ij')).reshape(3, -1).T
    z, x, y = grid[:, 0], grid[:, 1], grid[:, 2]

    x_p = x[(plane_number - 1) * len(span_y) * len(span_x):
            (plane_number - 1) * len(span_y) * len(span_x) + len(span_y) * len(span_x)]
    y_p = y[(plane_number - 1) * len(span_y) * len(span_x):
            (plane_number - 1) * len(span_y) * len(span_x) + len(span_y) * len(span_x)]
    z_p = z[(plane_number - 1) * len(span_y) * len(span_x):
            (plane_number - 1) * len(span_y) * len(span_x) + len(span_y) * len(span_x)]

    tot_field = np.real(total_field(x_p, y_p, z_p, k, ro, pos, spheres, order))

    for sph in range(len(spheres)):
        rx, ry, rz = x_p - pos[sph, 0], y_p - pos[sph, 1], z_p - pos[sph, 2]
        r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        tot_field = np.where(r < spheres[sph, 1], 0, tot_field)

    xy = tot_field.reshape(len(span_x), len(span_y))
    fig, ax = plt.subplots()
    plt.xlabel('y axis')
    plt.ylabel('x axis')
    im = ax.imshow(xy, cmap='viridis', origin='lower',
                   extent=[span_y.min(), span_y.max(), span_x.min(), span_x.max()])
    plt.colorbar(im)
    plt.show()


def simulation():
    # coordinates
    number_of_points = 200
    span_x = np.linspace(-0.03, 0.03, number_of_points)
    span_y = np.linspace(-0.03, 0.03, number_of_points)
    span_z = np.linspace(-0.03, 0.03, number_of_points)
    span = np.array([span_x, span_y, span_z])

    # parameters of fluid
    ro = 1000

    # parameters of the spheres
    k_sph1 = 1000
    r_sph1 = 0.003
    ro_sph1 = 2700
    sphere1 = np.array([k_sph1, r_sph1, ro_sph1])
    k_sph2 = 1000
    r_sph2 = 0.003
    ro_sph2 = 2700
    sphere2 = np.array([k_sph2, r_sph2, ro_sph2])
    k_sph2 = 1000
    r_sph2 = 0.003
    ro_sph2 = 2700
    sphere3 = np.array([k_sph2, r_sph2, ro_sph2])
    spheres = np.array([sphere1, sphere2, sphere3])

    # parameters of configuration
    pos1 = np.array([0.007, 0, 0])
    pos2 = np.array([0, 0.015, 0])
    pos3 = np.array([-0.003, -0.002, 0])
    pos = np.array([pos1, pos2, pos3])

    # parameters of the field
    k_x = -50
    k_y = 50
    k_z = 0
    k = np.array([k_x, k_y, k_z])

    # order of decomposition
    order = 5

    plane_number = int(number_of_points / 2) + 1
    xy_plot(span, plane_number, k, ro, pos, spheres, order)


def timetest(simulation):
    start = time.process_time()
    simulation()
    end = time.process_time()
    print(end-start)


timetest(simulation)