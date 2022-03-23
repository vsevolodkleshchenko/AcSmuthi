import numpy as np
import matplotlib.pyplot as plt
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


def coef_Qn(p0, n):
    r"""
    Coefficient Q_n in decomposition of (plane) wave
    Expression (9) from [http://dx.doi.org/10.1121/1.4773924 - Sapozhnikov]

    :param p0: amplitude of incident wave
    :param n: number of coefficient
    :return: nth coefficient
    """
    return p0 * 1j ** n * (2 * n + 1)


def coef_Gn(n, c_t, c_l, w, a, ro, ro_s):
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


def coef_cn(n, c_t, c_l, w, k, a, ro, ro_s):
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
    return (coef_Gn(n, c_t, c_l, w, a, ro, ro_s) * scipy.special.spherical_jn(n, k * a) -
          k * a * scipy.special.spherical_jn(n, k * a, derivative=True)) / \
         (coef_Gn(n, c_t, c_l, w, a, ro, ro_s) * scipy.special.hankel1(n, k * a) -
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
    r = np.sqrt(x * x + y * y + z * z)
    p_i = 0
    for n in range(N):
        p_i += coef_Qn(p0, n) * scipy.special.lpmv(0, n, z / r) * \
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
    r = np.sqrt(x * x + y * y + z * z)
    p_s = 0
    for n in range(N):
        p_s += coef_Qn(p0, n) * coef_cn(n, c_t, c_l, w, k, a, ro, ro_s) * \
               scipy.special.hankel1(n, k * r) * scipy.special.lpmv(0, n, z / r)
    return p_s


def radiation_force(p0, c_t, c_l, w, k, a, ro, ro_s, N):
    f_z = 0
    for n in range(N):
        f_z += (n + 1) * np.real(coef_cn(n, c_t, c_l, w, k, a, ro, ro_s) +
                                 np.conj(coef_cn(n + 1, c_t, c_l, w, k, a, ro, ro_s)) +
                                 2 * coef_cn(n, c_t, c_l, w, k, a, ro, ro_s) *
                                 np.conj(coef_cn(n + 1, c_t, c_l, w, k, a, ro, ro_s)))
    return - 2 * np.pi * p0 * p0 / ro / w / w


def count_field():
    r"""
    Simulatoin function:
    Plane wave scattering on sphere
    Theory in [http://dx.doi.org/10.1121/1.4773924 - Sapozhnikov]
    :return: field in point with coordinates(x, y, z)
    """
    # coordinates
    span_x = np.linspace(-10.2, 11.1, 50)
    span_y = np.linspace(-9.4, 9.3, 50)
    span_z = np.linspace(-10.6, 9.5, 50)
    grid = np.vstack(np.meshgrid(span_x, span_y, span_z)).reshape(3, -1).T
    x = grid[:, 0]
    y = grid[:, 1]
    z = grid[:, 2]

    # parameters of the sphere
    a = 1
    ro_s = 1
    c_l = 1
    c_t = 3

    # parameters of fluid
    ro = 1

    # parameters of the field
    p0 = 1
    k = 1
    c = 300
    w = k * c

    # order of decomposition
    N = 3

    p_i = p0 * np.exp(1j * k * z)
    p_s = scattered_field(x, y, z, p0, c_t, c_l, w, k, a, ro, ro_s, N)
    total_field = p_i + p_s

    # print values of total field for all coordinates
    print(total_field)

    # draw heat plot of amplitude of total field in Oxz slice for y = span_x[0] (axes are wrong) - it is in progress
    zx = np.asarray(np.abs(total_field[0:2500])).reshape(50, 50)
    plt.imshow(zx)
    plt.show()


# count_field()
