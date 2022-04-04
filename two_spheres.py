import numpy as np
import numpy.linalg
import scipy
import scipy.special
import matplotlib.pyplot as plt
import numba as nb
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


def coef_plane_wave(m, n, k_x, k_y, k_z):
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


def coef_f(n, k, r_sph, l_sph):
    """coefficient in linear system for plane wave scattering on
    2 spheres : jn' + lambda * jn
    :param n: array_like - degree of the wave function(int)
    :param k: array_like - absolute value of incident wave vector
    :param r_sph: float - radius of sphere
    :param l_sph: float complex - lambda parameter of sphere
    :return: array_like (complex float)"""
    return scipy.special.spherical_jn(n, k * r_sph, derivative=True) + \
           l_sph * scipy.special.spherical_jn(n, k * r_sph)


def coef_g(n, k, r_sph, l_sph):
    """coefficient in linear system for plane wave scattering on
    2 spheres: hn' + lambda * hn
    :param n: array_like - degree of the wave function(int)
    :param k: array_like - absolute value of incident wave vector
    :param r_sph: float - radius of sphere
    :param l_sph: float complex - lambda parameter of sphere
    :return: array_like (complex float)"""
    return sph_hankel1_der(n, k * r_sph) + l_sph * sph_hankel1(n, k * r_sph)


def coef_q(n, k, r_sph, l_sph):
    """coefficient in modified linear system for plane wave scattering on
    2 spheres: qn = gn/fn
    :param n: array_like - degree of the wave function(int)
    :param k: array_like - absolute value of incident wave vector
    :param r_sph: float - radius of sphere
    :param l_sph: float complex - lambda parameter of sphere
    :return: array_like (complex float)"""
    return coef_f(n, k, r_sph, l_sph) / coef_g(n, k, r_sph, l_sph)


def matrix_q(m, k, r_sph, l_sph, order):
    """matrix of coefficients qn for every m in modified
    linear system for plane wave scattering on 2 spheres
    :param m: array_like - order of the wave function(int)
    :param k: array_like - absolute value of incident wave vector
    :param r_sph: float - radius of sphere
    :param l_sph: float complex - lambda parameter of sphere
    :param order: int - order of decomposition
    :return: np.array (complex float)"""
    q = np.eye(order - np.abs(m) + 1, dtype=complex)
    for i in range(0, order - np.abs(m) + 1):
        q[i, i] = coef_q(i, k, r_sph, l_sph)
    return q


def coef_a(s, nu, m):
    """coefficient A_s(m, n) for S
    eq.(3.124) in Encyclopedia
    :param s: array_like - number of coefficient(int)
    :param nu: array_like - number of coefficient(int)
    :param m: array_like - number of coefficient(int)
    :return: array_like"""
    return np.sqrt(scipy.special.factorial(nu + m) / scipy.special.factorial(nu - m)) * \
           scipy.special.factorial(nu + s) / scipy.special.factorial(s + m) / \
           scipy.special.factorial(s) * scipy.special.factorial(nu - s) * np.sqrt(2 * nu + 1)


def coef_sz(m, n, nu, k, dist):
    """coefficient S^m_nnu(kb) if (b || z) in linear system for
    plane wave scattering on 2 spheres
    eq(3.126) in Encyclopedia
    :param m: array_like - number of coefficient(int)
    :param n: array_like - number of coefficient(int)
    :param nu: array_like - number of coefficient(int)
    :param k: array_like - absolute value of incident wave vector
    :param dist: float - distance between 2 spheres
    :return: array_like"""
    w = 1j / (2 * k * dist)
    sum = 0
    for j in range(np.abs(m), n + nu + 1):
        s_0 = np.max([0, j - n])
        s_1 = np.min([nu, j - np.abs(m)])
        sum1 = 0
        for s in range(s_0, s_1 + 1):
            sum1 += coef_a(s, nu, np.abs(m)) * coef_a(j - s, n, -np.abs(m))
        sum += scipy.special.factorial(j) * w ** j
    return (-1) ** m * 1j ** (n + nu) * np.exp(1j * k * dist) / (1j * k * dist) * sum


def matrix_sz1(m, k, dist, order):
    """matrix of coefficients Sz in first equation in linear
    system for plane wave scattering on 2 spheres
    :param m: array_like - number of coefficient(int)
    :param k: array_like - absolute value of incident wave vector
    :param dist: float - distance between 2 spheres
    :param order: int - order of decomposition
    :return: np.array"""
    sz1 = np.empty((order - np.abs(m) + 1, order - np.abs(m) + 1), dtype=complex)
    for n in range(np.abs(m), order + 1):
        for nu in range(np.abs(m), order + 1):
            sz1[n - np.abs(m), nu - np.abs(m)] = coef_sz(m, nu, n, k, dist)
    return sz1


def matrix_sz2(m, k, dist, order):
    """matrix of coefficients Sz in second equation in linear
    system for plane wave scattering on 2 spheres
    :param m: array_like - number of coefficient(int)
    :param k: array_like - absolute value of incident wave vector
    :param dist: float - distance between 2 spheres
    :param order: int - order of decomposition
    :return np.array"""
    sz2 = np.empty((order - np.abs(m) + 1, order - np.abs(m) + 1), dtype=complex)
    for n in range(np.abs(m), order + 1):
        for nu in range(np.abs(m), order + 1):
            sz2[n - np.abs(m), nu - np.abs(m)] = (-1) ** n * coef_sz(m, nu, n, k, dist)
    return sz2


def solve_system(q1, q2, sz1, sz2, d1, d2):
    r"""
    solve final system
    :param q1: np.array - matrix q1
    :param q2: np.array - matrix q2
    :param sz1: np.array - matrix sz1
    :param sz2: np.array - matrix sz2
    :param d1: np.array - matrix d1
    :param d2: np.arrai - matrix d2
    :return: np.array, np.array
    """
    c2 = np.linalg.solve(q2 - sz2.dot(np.linalg.inv(q1)).dot(sz1),
                         d2 - sz2.dot(np.linalg.inv(q1)).dot(d1))
    c1 = np.linalg.inv(q1).dot(d1) - np.linalg.inv(q1).dot(sz1).dot(c2)
    return c1, c2


def field_near_sphere1(x, y, z, k_x, k_y, k_z, r_sph1, l_sph1, r_sph2,
                       l_sph2, dist, order):
    r"""
    Scattered + incident field for plane wave scattering on 2 spheres (near first)
    eq(between 4.52 and 4.53) in Encyclopedia
    :param x, y, z: array_like - coordinates
    :param k_x, k_y, k_z:  array_like coordinates of incident wave vector
    :param r_sph1: float - radius of sphere 1
    :param l_sph1: float complex - lambda parameter of sphere 1
    :param r_sph2: float - radius of sphere 2
    :param l_sph2: float complex - lambda parameter of sphere 2
    :param dist: float - distance between 2 spheres
    :param order: int - order of decomposition
    :return: np.array (float)"""
    k = np.sqrt(k_x * k_x + k_y * k_y + k_z * k_z)
    u = 0
    for m in range(-order, order + 1):
        q1 = matrix_q(m, k, r_sph1, l_sph1, order)
        q2 = matrix_q(m, k, r_sph2, l_sph2, order)
        sz1 = matrix_sz1(m, k, dist, order)
        sz2 = matrix_sz2(m, k, dist, order)
        d1 = [coef_plane_wave(m, n, k_x, k_y, k_z) for n in
              range(np.abs(m), order + 1)]
        d2 = [coef_plane_wave(m, n, k_x, k_y, k_z) for n in
              range(np.abs(m), order + 1)]
        c1, c2 = solve_system(q1, q2, sz1, sz2, d1, d2)
        for n in range(np.abs(m), order + 1):
            transl = 0
            for nu in range(np.abs(m), order + 1):
                transl += (-1) ** nu * coef_sz(m, nu, n, k, dist) * c2[nu - np.abs(m)]
            u += d1[n - np.abs(m)] * regular_wvfs(m, n, x, y, z, k) + \
                 c1[n - np.abs(m)] * outgoing_wvfs(m, n, x, y, z, k) + \
                 outgoing_wvfs(m, n, x, y, z, k) * transl
    return np.real(u)


def xz_plot(span_x, span_y, span_z, plane_number, k_x, k_y, k_z, r_sph1, l_sph1, r_sph2,
            l_sph2, dist, order):
    r"""
    Count field and build a 2D heat-plot in XZ plane for y[plane_number]
    --->z
    :param span_x, span_y, span_z: np.array (float)
    :param plane_number: int
    :param k_x, k_y, k_z:  array_like coordinates of incident wave vector
    :param r_sph1: float - radius of sphere 1
    :param l_sph1: float complex - lambda parameter of sphere 1
    :param r_sph2: float - radius of sphere 2
    :param l_sph2: float complex - lambda parameter of sphere 2
    :param dist: float - distance between 2 spheres
    :param order: int - order of decomposition
    :return: 2D heat-plot"""
    grid = np.vstack(np.meshgrid(span_y, span_x, span_z, indexing='ij')).reshape(3, -1).T
    y = grid[:, 0]
    x = grid[:, 1]
    z = grid[:, 2]

    total_field = field_near_sphere1(x, y, z, k_x, k_y, k_z, r_sph1,
                                     l_sph1, r_sph2, l_sph2, dist, order)

    xz = np.flip(np.asarray(total_field[(plane_number - 1) * len(span_x) * len(span_z):
                                (plane_number - 1) * len(span_x) * len(span_z) +
                                len(span_x) * len(span_z)]).reshape(len(span_x), len(span_z)), axis=0)
    fig, ax = plt.subplots()
    ax.imshow(xz, cmap='viridis')
    plt.show()

    # print(grid, x, y, z, total_field, xz, sep='\n')


def yz_plot(span_x, span_y, span_z, plane_number, k_x, k_y, k_z, r_sph1, l_sph1, r_sph2,
            l_sph2, dist, order):
    r"""
    Count field and build a 2D heat-plot in YZ plane for x[plane_number]
    --->z
    :param span_x, span_y, span_z: np.array (float)
    :param plane_number: int
    :param k_x, k_y, k_z:  array_like coordinates of incident wave vector
    :param r_sph1: float - radius of sphere 1
    :param l_sph1: float complex - lambda parameter of sphere 1
    :param r_sph2: float - radius of sphere 2
    :param l_sph2: float complex - lambda parameter of sphere 2
    :param dist: float - distance between 2 spheres
    :param order: int - order of decomposition
    :return: 2D heat-plot"""
    grid = np.vstack(np.meshgrid(span_x, span_y, span_z, indexing='ij')).reshape(3, -1).T
    x = grid[:, 0]
    y = grid[:, 1]
    z = grid[:, 2]

    total_field = field_near_sphere1(x, y, z, k_x, k_y, k_z, r_sph1,
                                     l_sph1, r_sph2, l_sph2, dist, order)

    yz = np.flip(np.asarray(total_field[(plane_number - 1) * len(span_y) * len(span_z):
                                (plane_number - 1) * len(span_y) * len(span_z) +
                                len(span_y) * len(span_z)]).reshape(len(span_y), len(span_z)), axis=0)

    fig, ax = plt.subplots()
    ax.imshow(yz, cmap='viridis')
    plt.show()

    # print(grid, x, y, z, total_field, yz, sep='\n')


def xy_plot(span_x, span_y, span_z, plane_number, k_x, k_y, k_z, r_sph1, l_sph1, r_sph2,
            l_sph2, dist, order):
    r"""
    Count field and build a 2D heat-plot in XZ plane for z[plane_number]
    --->y
    :param span_x, span_y, span_z: np.array (float)
    :param plane_number: int
    :param k_x, k_y, k_z:  array_like coordinates of incident wave vector
    :param r_sph1: float - radius of sphere 1
    :param l_sph1: float complex - lambda parameter of sphere 1
    :param r_sph2: float - radius of sphere 2
    :param l_sph2: float complex - lambda parameter of sphere 2
    :param dist: float - distance between 2 spheres
    :param order: int - order of decomposition
    :return: 2D heat-plot"""
    grid = np.vstack(np.meshgrid(span_z, span_x, span_y, indexing='ij')).reshape(3, -1).T
    z = grid[:, 0]
    x = grid[:, 1]
    y = grid[:, 2]

    total_field = field_near_sphere1(x, y, z, k_x, k_y, k_z, r_sph1,
                                     l_sph1, r_sph2, l_sph2, dist, order)

    xy = np.flip(np.asarray(total_field[(plane_number-1)*len(span_x)*len(span_y):
                                (plane_number-1)*len(span_x)*len(span_y)+
                                len(span_x)*len(span_y)]).reshape(len(span_x), len(span_y)), axis=0)
    fig, ax = plt.subplots()
    ax.imshow(xy, cmap='viridis')
    plt.show()

    # print(grid, x, y, z, total_field, xy, sep='\n')


def simulation():
    # coordinates
    number_of_points = 3
    span_x = np.linspace(2, 11.1, number_of_points)
    span_y = np.linspace(20, 30, number_of_points)
    span_z = np.linspace(41, 49, number_of_points)

    # parameters of the sphere 1
    l_sph1 = 1.1
    r_sph1 = 2.0

    # parameters of the sphere 2
    l_sph2 = 1.1
    r_sph2 = 2.0

    # parameters of configuration
    dist = 5.0

    # parameters of the field
    k_x = 0.2
    k_y = 0.0
    k_z = 2.3

    # order of decomposition
    order = 3

    plane_number = int(number_of_points / 2) + 1
    xy_plot(span_x, span_y, span_z, plane_number, k_x, k_y, k_z,
            r_sph1, l_sph1, r_sph2, l_sph2, dist, order)


def timetest(simulation):
    start = time.process_time()
    simulation()
    end = time.process_time()
    print(end-start)


timetest(simulation)
