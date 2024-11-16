import numpy as np
import numpy.typing as npt
from typing import Any, Iterable
import scipy.special as ss


def spherical_h1n(
        n: int | np.ndarray | Iterable | float,
        z:  complex | float | np.ndarray | Iterable | int,
        derivative: bool | None = False
) -> Any:
    """Spherical Hankel function of the first kind or its derivative.

    Compute :math:`h^1_n(z)` using Bessel functions from scipy.special

    :param n: multipole order (non-negative)
    :param z: function argument
    :param derivative: if True, the value of the derivative (rather than the function itself) is returned
    :return: value of the function or its z-derivative
    """
    if derivative:
        return ss.spherical_jn(n, z, derivative=True) + 1j * ss.spherical_yn(n, z, derivative=True)
    else:
        return ss.spherical_jn(n, z) + 1j * ss.spherical_yn(n, z)


def spherical_jn_der2(
        n: int | np.ndarray | Iterable | float,
        z:  complex | float | np.ndarray | Iterable | int
) -> Any:
    """Second derivative of spherical Bessel function of the first kind.

    See http://dlmf.nist.gov/10.51.i for details.

    :param n: multipole order (non-negative)
    :param z: function argument
    :return: second z-derivative of the bessel function
    """
    if n == 0:  # todo: check is it necessary
        return - ss.spherical_jn(1, z, derivative=True)
    else:
        return ss.spherical_jn(n - 1, z, derivative=True) + (n + 1) / z**2 * ss.spherical_jn(n, z) - \
               (n + 1) / z * ss.spherical_jn(n, z, derivative=True)


def car_to_sph(x, y, z):  # todo: r, THETA, PHI
    """Converts cartesian coordinates to spherical coordinates."""
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    if phi.shape == ():
        if phi < 0:
            phi += 2 * np.pi
    else:
        phi[phi < 0] += 2 * np.pi
    return r, phi, theta


def car_to_cyl(x, y, z):
    """Converts cartesian coordinates to cylindrical coordinates."""
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    if phi.shape == ():
        if phi < 0:
            phi += 2 * np.pi
    else:
        phi[phi < 0] += 2 * np.pi
    return rho, phi, z


def legendres_table(z: npt.NDArray[float | complex], n_max: int):
    """Table of associated Legendre function of the first kind for complex arguments.

    Values :math:`P^m_n(z)` of orders 0...n and degrees -m...0 and 0...m for one-dimensional array of z-values

    :param z: function argument
    :param n_max: maximum multipole order of expansion (non-negative)
    :return: two arrays of shape (len(z), n_max + 1, n_max + 1)
    """
    legs_positive_m = np.array([ss.clpmn(n_max, n_max, zi, type=2)[0] for zi in z])
    legs_negative_m = np.array([ss.clpmn(-n_max, n_max, zi, type=2)[0] for zi in z])
    return np.moveaxis(legs_positive_m, 0, -1), np.moveaxis(legs_negative_m, 0, -1)


def legendre_prefactor(
        m: int | npt.NDArray[int] | Iterable[int],
        n: int | npt.NDArray[int] | Iterable[int]
) -> npt.NDArray[float]:
    """Compute Legendre pre factors.

    :math:`\\sqrt{ (2n+1)/(4\\pi) \\cdot (n-m)!/(n+m)!}`

    :param m: multipole degree
    :param n: multipole order (non-negative)
    :return: Legendre pre factors
    """
    return np.sqrt((2 * n + 1) / 4 / np.pi * ss.factorial(n - m) / ss.factorial(n + m))
