import numpy as np
import scipy
import scipy.special
from sympy.physics.quantum.matrixutils import scipy


def second_derivation_spherical_jn(n, z):
    r"""
    The second derivation of nth spherical Bessel function

    :param n: order of spherical Bessel function
    :param z: argument of spherical Bessel function
    :return: function
    """

    if n == 0:
        return - scipy.special.spherical_jn(1, z, derivative=True)
    else:
        return scipy.special.spherical_jn(n - 1, z, derivative=True) - \
               (n + 1) / z * scipy.special.spherical_jn(n, z, derivative=True) + \
               (n + 1) / z / z * scipy.special.spherical_jn(n, z)


def Q_n(p0, n):
    r"""
    Coefficient Q_n of decomposition (plane) wave
    Expression (9) from [http://dx.doi.org/10.1121/1.4773924 - Sapozhnikov]

    :param p0: amplitude of incident wave
    :param n: number of coefficient
    :return: nth coefficient
    """
    return p0 * 1j ** n * (2 * n + 1)


def G_n(n, c_t, c_l, w, a, ro, ro_s):
    r"""
    Coefficient Г_n to express coefficient c_n in composition of scattered wave
    Expression (13) from [http://dx.doi.org/10.1121/1.4773924 - Sapozhnikov]

    :param n: number of coefficient
    :param c_t: longitudinal wave velocity of sphere material
    :param c_l: shear wave velocity of sphere material
    :param w: radial frequency of incident wave
    :param a: radius of the sphere
    :param ro: fluid density
    :param ro_s: sphere density
    :return: nth coefficient
    """

    k_t = w / c_t
    k_l = w / c_l
    sigma = (c_l * c_l / 2 - c_t * c_t) / (c_l * c_l - c_t * c_t)
    a_n = scipy.special.spherical_jn(n, k_l * a) - \
          k_l * a * scipy.special.spherical_jn(n, k_l * a, derivative=True)
    b_n = (n * n + n - 2) * scipy.special.spherical_jn(n, k_t * a) + \
          k_t * k_t * a * a * second_derivation_spherical_jn(n, k_t * a)
    xi_n = k_l * a * scipy.special.spherical_jn(n, k_l * a, derivative=True)
    d_n = 2 * n * (n + 1) * scipy.special.spherical_jn(n, k_t * a)
    eps_n = k_l * k_l * a * a * \
          (scipy.special.spherical_jn(n, k_l * a) * sigma / (1 - 2 * sigma) -
           second_derivation_spherical_jn(n, k_l * a))
    eta_n = 2 * n * (n + 1) * \
            (scipy.special.spherical_jn(n, k_t * a) -
             k_t * a * scipy.special.spherical_jn(n, k_t * a, derivative=True))
    return ro * k_t * k_t * a * a / 2 / ro_s * \
           (a_n * d_n + b_n * xi_n) / (a_n * eta_n + b_n * eps_n)


def c_n(n, c_t, c_l, w, k, a, ro, ro_s):
    r"""
    Coefficient c_n in decomposition of scattered wave
    Expression (12) from [http://dx.doi.org/10.1121/1.4773924 - Sapozhnikov]

    :param n: number of coefficient
    :param c_t: longitudinal wave velocity of sphere material
    :param c_l: shear wave velocity of sphere material
    :param w: radial frequency
    :param k: wave vector
    :param a: radius of sphere
    :param ro: fluid density
    :param ro_s: sphere density
    :return: nth coefficient
    """
    return (G_n(n, c_t, c_l, w, a, ro, ro_s) * scipy.special.spherical_jn(n, k * a) -
          k * a * scipy.special.spherical_jn(n, k * a, derivative=True)) / \
         (G_n(n, c_t, c_l, w, a, ro, ro_s) * scipy.special.hankel1(n, k * a) -
          k * a * scipy.special.h1vp(n, k * a, 1))


def incident_wave_decomposition(x, y, z, k, p0, N):
    r"""
    Decomposition of plane wave
    Expression (7) from [http://dx.doi.org/10.1121/1.4773924 - Sapozhnikov]

    :param x: x - coordinate
    :param y: y - coordinate
    :param z: z - coordinate
    :param k: wave vector
    :param p0: amplitude of wave
    :param N: number of terms in sum (decomposition)
    :return: decomposition of plane wave
    """
    r = x * x + y * y + z * z
    p_i = 0
    for n in range(N):
        p_i += Q_n(p0, n) * scipy.special.lpn(n, z / r)[0][-1] * \
               scipy.special.spherical_jn(n, k * r)
    return p_i


def scattered_field(x, y, z, p0, c_t, c_l, w, k, a, ro, ro_s, N):
    r"""
    Decomposition of scattered field on sphere
    Expression (11) from [http://dx.doi.org/10.1121/1.4773924 - Sapozhnikov]

    :param x: x - coordinate
    :param y: y - coordinate
    :param z: z - coordinate
    :param p0: amplitude of wave
    :param c_t: longitudinal wave velocity of sphere material
    :param c_l: shear wave velocity of sphere material
    :param w: radial frequency
    :param k: wave vector
    :param a: radius of sphere
    :param ro: fluid density
    :param ro_s: sphere density
    :param N: number of terms in sum (decomposition)
    :return: scattered on sphere field decomposition
    """
    r = x * x + y * y + z * z
    p_s = 0
    for n in range(N):
        p_s += Q_n(p0, n) * c_n(n, c_t, c_l, w, k, a, ro, ro_s) * \
               scipy.special.hankel1(n, k * r) * scipy.special.lpn(n, z / r)[0][-1]
    return p_s


def count_field():
    r"""
    Simulatoin function:
    Plane wave scattering on sphere
    Theory in [http://dx.doi.org/10.1121/1.4773924 - Sapozhnikov]
    :return: field in point with coordinates(x, y, z)
    """
    # coordinates
    x = 10
    y = 10
    z = 10

    # parameters of the sphere
    a = 1
    ro_s = 1
    c_l = 1
    c_t = 3

    #parametres of fluid
    ro = 1

    # parameters of the field
    p0 = 1
    k = 1
    c = 300
    w = k / c

    # order of decomposition
    N = 3

    p_i = p0 * np.exp(1j * k * z)
    p_s = scattered_field(x, y, z, p0, c_t, c_l, w, k, a, ro, ro_s, N)
    total_field = p_i + p_s

    print(total_field)

count_field()