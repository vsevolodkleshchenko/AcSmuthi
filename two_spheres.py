import numpy as np
import numpy.linalg
import scipy
import scipy.special
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


def coef_plane_wave(m, n, k_x, k_y, k_z):
    r""" coefficients in decomposition of plane wave
    d^m_n - eq(4.40) of Encyclopedia
    :param m, n: array_like - order and degree of the harmonic (int)
    :param k_x, k_y, k_z: coordinates of incident wave vector
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
    :param k: absolute value of incident wave vector
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
    :param k: absolute value of incident wave vector
    :return: array_like (complex float) """
    r = np.sqrt(x * x + y * y + z * z)
    phi = np.arctan(x / y)
    theta = np.arccos(z / r)
    return sph_hankel1(n, k * r) * \
           scipy.special.sph_harm(m, n, theta, phi)


def coef_f(n, k, r_sph, l_sph):
    """coefficient in linear system = jn' + lambda * jn"""
    return scipy.special.spherical_jn(n, k * r_sph, derivative=True) + \
           l_sph * scipy.special.spherical_jn(n, k * r_sph)


def coef_g(n, k, r_sph, l_sph):
    """coefficient in linear system = hn' + lambda * hn"""
    return sph_hankel1_der(n, k * r_sph) + l_sph * sph_hankel1(n, k * r_sph)


def coef_q(n, k, r_sph, l_sph):
    """coefficient qn = gn/fn in modified system"""
    return coef_f(n, k, r_sph, l_sph) / coef_g(n, k, r_sph, l_sph)


def matrix_q(m, k, r_sph, l_sph, order):
    """matrix of coefficients qn for every m"""
    q = np.eye(order - np.abs(m) + 1, dtype=complex)
    for i in range(0, order - np.abs(m) + 1):
        q[i, i] = coef_q(i, k, r_sph, l_sph)
    return q


def coef_a(s, nu, m):
    """coefficient A_s(m, n) for S
    eq.(3.124) in Encyclopedia"""
    return np.sqrt(scipy.special.factorial(nu + m) / scipy.special.factorial(nu - m)) * \
           scipy.special.factorial(nu + s) / scipy.special.factorial(s + m) / \
           scipy.special.factorial(s) * scipy.special.factorial(nu - s) * np.sqrt(2 * nu + 1)


def coef_sz(m, n, nu, k, dist):
    """coefficient S^m_nnu(kb) if (b || z)
    eq(3.126) in Encyclopedia"""
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
    """matrix of coefficients Sz in first equation in linear system"""
    sz1 = np.empty((order - np.abs(m) + 1, order - np.abs(m) + 1), dtype=complex)
    for n in range(np.abs(m), order + 1):
        for nu in range(np.abs(m), order + 1):
            sz1[n - np.abs(m), nu - np.abs(m)] = coef_sz(m, nu, n, k, dist)
    return sz1


def matrix_sz2(m, k, dist, order):
    """matrix of coefficients Sz in second equation in linear system"""
    sz2 = np.empty((order - np.abs(m) + 1, order - np.abs(m) + 1), dtype=complex)
    for n in range(np.abs(m), order + 1):
        for nu in range(np.abs(m), order + 1):
            sz2[n - np.abs(m), nu - np.abs(m)] = (-1) ** n * coef_sz(m, nu, n, k, dist)
    return sz2


def solve_system(q1, q2, sz1, sz2, d1, d2):
    """solve final system"""
    c2 = np.linalg.solve(q2 - sz2.dot(np.linalg.inv(q1)).dot(sz1),
                         d2 - sz2.dot(np.linalg.inv(q1)).dot(d1))
    c1 = np.linalg.inv(q1).dot(d1) - np.linalg.inv(q1).dot(sz1).dot(c2)
    return c1, c2


def field_near_sphere1(x, y, z, k_x, k_y, k_z, r_sph1, l_sph1, r_sph2,
                       l_sph2, dist, order):
    """Scattered and incident field near first sphere
    eq(between 4.52 and 4.53) in Encyclopedia"""
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


def simulation():
    # coordinates
    span_x = np.linspace(-10.2, 11.1, 50)
    span_y = np.linspace(-9.4, 9.3, 50)
    span_z = np.linspace(-10.6, 9.5, 50)
    grid = np.vstack(np.meshgrid(span_x, span_y, span_z)).reshape(3, -1).T
    x = grid[:, 0]
    y = grid[:, 1]
    z = grid[:, 2]

    # parameters of the sphere 1
    l_sph1 = 1.1
    r_sph1 = 2

    # parameters of the sphere 2
    l_sph2 = 1.1
    r_sph2 = 2

    # parameters of configuration
    dist = 5

    # parameters of the field
    k_x = 2.1
    k_y = 0
    k_z = 2.3

    # order of decomposition
    order = 3

    total_field = field_near_sphere1(x, y, z, k_x, k_y, k_z, r_sph1,
                                     l_sph1, r_sph2, l_sph2, dist, order)

    # print values of total field for all coordinates
    print(total_field)

    # draw heat plot of amplitude of total field in Oxz slice for y = span_x[0] (axes are wrong) - it is in progress
    zx = np.asarray(np.abs(total_field[0:2500])).reshape(50, 50)
    plt.imshow(zx)
    plt.show()


simulation()
