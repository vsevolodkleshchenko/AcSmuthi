import numpy as np
import scipy
import scipy.special
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import math
import pywigxjpf as wig
# import matplotlib.colors as colors
from sympy.physics.wigner import wigner_3j


def n_idx(n):
    r""" build np.array of numbers 0,1,1,1,2,2,2,2,2,...,n,..,n """
    return np.repeat(np.arange(n+1), np.arange(n+1) * 2 + 1)


def m_idx(n):
    r""" build np.array of numbers 0,-1,0,1,-2,-1,0,1,2,...,-n,..,n """
    return np.concatenate([np.arange(-i, i + 1) for i in range(n + 1)])


# print(np.split(np.repeat(m_idx(3), len(np.array([-3, -2, -1, 0, 1, 2, 3]))), 4 **2 ))


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


def inc_coef(m, n, k):
    r""" Coefficients in decomposition of plane wave
    d^m_n - eq(4.40) of 'Encyclopedia' """
    k_abs, k_phi, k_theta = dec_to_sph(k[0], k[1], k[2])
    return 4 * np.pi * 1j ** n * np.conj(scipy.special.sph_harm(m, n, k_phi, k_theta))


def local_inc_coef(m, n, k, sph_pos, order):
    r""" Counts local incident coefficients
    d^m_nj - eq(42) in 'Multiple scattering and scattering cross sections P. A. Martin' """
    inccoef_array = np.zeros((order+1) ** 2, dtype=complex)
    i = 0
    for munu in zip(m_idx(order), n_idx(order)):
        inccoef_array[i] = inc_coef(munu[0], munu[1], k) * sepc_matr_coef(munu[0], m, munu[1], n, k, sph_pos)
        i += 1
    return accurate_csum(inccoef_array)


def coefficient_array(n, k, coef, length):
    c_array = np.zeros(((n+1) ** 2), dtype=complex)
    i = 0
    for mn in zip(m_idx(n), n_idx(n)):
        c_array[i] = coef(mn[0], mn[1], k)
        i += 1
    return np.split(np.repeat(c_array, length), (n + 1) ** 2)


def regular_wvfs(m, n, x, y, z, k):
    r""" Regular basis spherical wave functions
    ^psi^m_n - eq(between 4.37 and 4.38) of 'Encyclopedia' """
    k_abs, k_phi, k_theta = dec_to_sph(k[0], k[1], k[2])
    r, phi, theta = dec_to_sph(x, y, z)
    return scipy.special.spherical_jn(n, k_abs * r) * scipy.special.sph_harm(m, n, phi, theta)


def regular_wvfs_array(n, x, y, z, k):
    r""" builds np.array of all regular wave functions with order <= n"""
    rw_array = np.zeros(((n+1) ** 2, len(x)), dtype=complex)
    i = 0
    for mn in zip(m_idx(n), n_idx(n)):
        rw_array[i] = regular_wvfs(mn[0], mn[1], x, y, z, k)
        i += 1
    return rw_array


def outgoing_wvfs(m, n, x, y, z, k):
    r""" Outgoing basis spherical wave functions
    psi^m_n - eq(between 4.37 and 4.38) in 'Encyclopedia' """
    k_abs, k_phi, k_theta = dec_to_sph(k[0], k[1], k[2])
    r, phi, theta = dec_to_sph(x, y, z)
    return sph_hankel1(n, k_abs * r) * scipy.special.sph_harm(m, n, phi, theta)


def outgoing_wvfs_array(n, x, y, z, k):
    r""" builds np.array of all outgoing wave functions with order less n"""
    ow_array = np.zeros(((n+1) ** 2, len(x)), dtype=complex)
    i = 0
    for mn in zip(m_idx(n), n_idx(n)):
        ow_array[i] = outgoing_wvfs(mn[0], mn[1], x, y, z, k)
        i += 1
    return ow_array


# print(outgoing_wvfs_array(3, np.array([-3, -3, -3]), np.array([-3, -3, -3]), np.array([-3, -3, -3]), np.array([-1, -1, 1])))


def gaunt_coef(n, m, nu, mu, q):
    r""" Gaunt coefficient: G(n,m;nu,mu;q)
    eq(3.71) in 'Encyclopedia' """
    s = np.sqrt((2 * n + 1) * (2 * nu + 1) * (2 * q + 1) / 4 / np.pi)
    wig.wig_table_init(60, 3)  # this needs revision
    wig.wig_temp_init(60)  # this needs revision
    return (-1.) ** (m + mu) * s * wig.wig3jj(2*n, 2*nu, 2*q, 0, 0, 0) * \
           wig.wig3jj(2*n, 2*nu, 2*q, 2*m, 2*mu, - 2*m - 2*mu)


def sepc_matr_coef(m, mu, n, nu, k, dist):
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
        sum_array[i] = (-1) ** q * regular_wvfs(m - mu, q0 + 2 * q, dist[0], dist[1], dist[2], k) * \
               gaunt_coef(n, m, nu, -mu, q0 + 2 * q)
        i += 1
    return 4 * np.pi * (-1) ** (mu + nu + q_lim) * accurate_csum(sum_array)


def sep_matr_coef(m, mu, n, nu, k, dist):
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
        sum_array[i] = (-1) ** q * outgoing_wvfs(m - mu, q0 + 2 * q, dist[0], dist[1], dist[2], k) * \
               gaunt_coef(n, m, nu, -mu, q0 + 2 * q)
        i += 1
    return 4 * np.pi * (-1) ** (mu + nu + q_lim) * accurate_csum(sum_array)


def syst_matr(k, ro, pos, spheres, order):
    r""" Builds T-matrix """
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
            row_idx_1 = np.arange(sph * block_height + 2 * n ** 2, sph * block_height + 2 * (n + 1) ** 2, 2)
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


def syst_rhs(k, pos, spheres, order):
    r""" build right hand side of T-matrix system """
    k_abs = dec_to_sph(k[0], k[1], k[2])[0]
    num_of_sph = len(spheres)
    num_of_coef = (order + 1) ** 2
    rhs = np.zeros(num_of_coef * 2 * num_of_sph, dtype=complex)
    for sph in range(num_of_sph):
        for n in range(order + 1):
            for m in range(-n, n + 1):
                inccoef = local_inc_coef(m, n, k, pos[sph], order)
                rhs[sph * 2 * num_of_coef + 2 * (n ** 2 + n + m)] = inccoef * \
                       scipy.special.spherical_jn(n, k_abs * spheres[sph, 1])
                rhs[sph * 2 * num_of_coef + 2 * (n ** 2 + n + m) + 1] = inccoef * \
                           scipy.special.spherical_jn(n, k_abs * spheres[sph, 1], derivative=True)
    return rhs


def syst_solve(k, ro, pos, spheres, order):
    r""" solve T matrix system and counts a coefficients in decomposition
    of scattered field and field inside the spheres """
    num_of_sph = len(spheres)
    t_matrix = syst_matr(k, ro, pos, spheres, order)
    rhs = syst_rhs(k, pos, spheres, order)
    coef = scipy.linalg.solve(t_matrix, rhs)
    return np.array(np.split(coef, 2 * num_of_sph))


def accurate_csum(array):
    return math.fsum(np.real(array)) + 1j * math.fsum(np.imag(array))


def accurate_sph_mp_sum(field_array, length):
    r""" do accurate sum by spheres and multipoles
    the shape of field array: 0 axis - spheres, 1 axis - multipoles, 2 axis - coordinates
    return: np.array with values of field in all coordinates """
    field = np.zeros(length, dtype=complex)
    for i in range(length):
        field[i] = accurate_csum(np.concatenate(field_array[:, :, i]))
    return field


def accurate_mp_sum(field_array, length):
    r""" do accurate sum by multipoles
    the shape of field array: 0 axis - multipoles, 1 axis - coordinates
    return: np.array with values of field in all coordinates """
    field = np.zeros(length, dtype=complex)
    for i in range(length):
        field[i] = accurate_csum(field_array[:, i])
    return field


def total_field(x, y, z, k, ro, pos, spheres, order):
    r""" counts field outside the spheres """
    coef = syst_solve(k, ro, pos, spheres, order)
    tot_field_array = np.zeros((len(spheres), (order + 1) ** 2, len(x)), dtype=complex)
    for sph in range(len(spheres)):
        coef_array = np.split(np.repeat(coef[2 * sph], len(x)), (order + 1) ** 2)
        tot_field_array[sph] = coef_array * outgoing_wvfs_array(order, x-pos[sph][0], y-pos[sph][1], z-pos[sph][2], k)
    tot_field = np.sum(tot_field_array, axis=(0, 1))
    # tot_field = accurate_sph_mp_sum(tot_field_array, len(x))
    return tot_field


def cross_section(k, ro, pos, spheres, order):
    r""" Counts scattering and extinction cross sections Sigma_sc and Sigma_ex
    eq(46,47) in 'Multiple scattering and scattering cross sections P. A. Martin' """
    coef = syst_solve(k, ro, pos, spheres, order)
    num_sph = len(spheres)
    sigma_ex = np.zeros(num_sph * (order + 1) ** 2)
    sigma_sc1 = np.zeros(num_sph * (order + 1) ** 2)
    sigma_sc2 = np.zeros((num_sph * (order + 1) ** 2) ** 2, dtype=complex)
    jmn, jmnlmunu = 0, 0
    for j in range(num_sph):
        for mn in zip(m_idx(order), n_idx(order)):
            for l in range(num_sph):
                for munu in zip(m_idx(order), n_idx(order)):
                    sigma_sc2[jmnlmunu] = np.conj(coef[2 * j, mn[1] ** 2 + mn[1] + mn[0]]) * \
                               coef[2 * l, munu[1] ** 2 + munu[1] + munu[0]] * \
                               sepc_matr_coef(munu[0], mn[0], munu[1], mn[1], k, pos[j] - pos[l])
                    jmnlmunu += 1
            sigma_sc1[jmn] = np.abs(coef[2 * j, mn[1] ** 2 + mn[1] + mn[0]]) ** 2
            sigma_ex[jmn] = - np.real(coef[2 * j, mn[1] ** 2 + mn[1] + mn[0]] *
                                      np.conj(local_inc_coef(mn[0], mn[1], k, pos[j], order)))
            jmn += 1
    sigma_sc = math.fsum(np.real(sigma_sc1)) + math.fsum(np.real(sigma_sc2))
    sigma_ex = math.fsum(sigma_ex)
    return sigma_sc, sigma_ex


def total_field_m(x, y, z, k, ro, pos, spheres, order, m=-1):
    r""" Counts field outside the spheres for mth harmonic """
    coef = syst_solve(k, ro, pos, spheres, order)
    tot_field = 0
    for n in range(abs(m), order + 1):
        for sph in range(len(spheres)):
            tot_field += coef[2 * sph][n ** 2 + n + m] * \
                         outgoing_wvfs(m, n, x - pos[sph][0], y - pos[sph][1], z - pos[sph][2], k)
        tot_field += inc_coef(m, n, k) * regular_wvfs(m, n, x, y, z, k)
    return tot_field


def draw_spheres(field, pos, spheres, x_p, y_p, z_p):
    for sph in range(len(spheres)):
        rx, ry, rz = x_p - pos[sph, 0], y_p - pos[sph, 1], z_p - pos[sph, 2]
        r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        field = np.where(r < spheres[sph, 1], 0, field)
    return field


def build_slice(span, plane_number, plane='xz'):
    r""" Build np.arrays of points of grid to build a slice plot """
    span_x, span_y, span_z = span[0], span[1], span[2]
    x, y, z, span_v, span_h = None, None, None, None, None
    if plane == 'xz':
        grid = np.vstack(np.meshgrid(span_y, span_x, span_z, indexing='ij')).reshape(3, -1).T
        y, x, z = grid[:, 0], grid[:, 1], grid[:, 2]
        span_v, span_h = span_x, span_z
    if plane == 'yz':
        grid = np.vstack(np.meshgrid(span_x, span_y, span_z, indexing='ij')).reshape(3, -1).T
        x, y, z = grid[:, 0], grid[:, 1], grid[:, 2]
        span_v, span_h = span_y, span_z
    if plane == 'xy':
        grid = np.vstack(np.meshgrid(span_z, span_x, span_y, indexing='ij')).reshape(3, -1).T
        z, x, y = grid[:, 0], grid[:, 1], grid[:, 2]
        span_v, span_h = span_x, span_y

    x_p = x[(plane_number - 1) * len(span_v) * len(span_h):
            (plane_number - 1) * len(span_v) * len(span_h) + len(span_v) * len(span_h)]
    y_p = y[(plane_number - 1) * len(span_v) * len(span_h):
            (plane_number - 1) * len(span_v) * len(span_h) + len(span_v) * len(span_h)]
    z_p = z[(plane_number - 1) * len(span_v) * len(span_h):
            (plane_number - 1) * len(span_v) * len(span_h) + len(span_v) * len(span_h)]
    return x_p, y_p, z_p, span_v, span_h


def slice_plot(span, plane_number, k, ro, pos, spheres, order, plane='xz'):
    r""" Count field and build a 2D heat-plot in:
     XZ plane for span_y[plane_number] : --->z
     YZ plane for span_x[plane_number] : --->z
     XY plane for span_z[plane_number] : --->y """
    x_p, y_p, z_p, span_v, span_h = build_slice(span, plane_number, plane=plane)
    tot_field = np.real(total_field(x_p, y_p, z_p, k, ro, pos, spheres, order))
    tot_field = draw_spheres(tot_field, pos, spheres, x_p, y_p, z_p)
    tot_field_reshaped = tot_field.reshape(len(span_v), len(span_h))
    fig, ax = plt.subplots()
    plt.xlabel(plane[1]+'axis')
    plt.ylabel(plane[0]+'axis')
    im = ax.imshow(tot_field_reshaped, norm=colors.CenteredNorm(), cmap='seismic', origin='lower',
                   extent=[span_h.min(), span_h.max(), span_v.min(), span_v.max()])
    plt.colorbar(im)
    plt.show()


def simulation():
    r""" main simulation function """
    # coordinates
    number_of_points = 200
    l = 10
    span_x = np.linspace(-l, l, number_of_points)
    span_y = np.linspace(-l, l, number_of_points)
    span_z = np.linspace(-l, l, number_of_points)
    span = np.array([span_x, span_y, span_z])

    # parameters of fluid
    freq = 82
    ro = 1.225
    c_f = 331
    k_fluid = 2 * np.pi * freq / c_f

    # parameters of the spheres
    c_sph = 1403
    k_sph = 2 * np.pi * freq / c_sph
    r_sph = 1
    ro_sph = 1050
    sphere = np.array([k_sph, r_sph, ro_sph])
    spheres = np.array([sphere, sphere])

    # parameters of configuration
    pos1 = np.array([0, 0, -2.5])
    pos2 = np.array([0, 0, 2.5])
    poses = np.array([pos1, pos2])

    # parameters of the field
    k_x = 0.70711 * k_fluid
    k_y = 0
    k_z = 0.70711 * k_fluid
    k = np.array([k_x, k_y, k_z])

    # order of decomposition
    order = 8

    print("Scattering and extinction cross section:", *cross_section(k, ro, poses, spheres, order))

    plane_number = int(number_of_points / 2) + 1
    slice_plot(span, plane_number, k, ro, poses, spheres, order, plane='xz')


def timetest(simulation):
    start = time.process_time()
    simulation()
    end = time.process_time()
    print("Time:", end-start)


timetest(simulation)
