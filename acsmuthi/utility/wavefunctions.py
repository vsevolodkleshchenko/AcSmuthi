import numpy as np
import numpy.typing as npt
import scipy.special as ss
from acsmuthi.utility import mathematics as mths


def n_idx(n_max: int) -> npt.NDArray[int]:
    """Build array of multipole orders (indexes) n from 0 to n_max.

    Each multipole order n is repeated (2n + 1) times in array.

    :param n_max: maximum multipole order of expansion (non-negative)
    :return: array of multipole orders

    >>> n_idx(3)
    10
    array([0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3])
    """
    return np.repeat(np.arange(n_max + 1), np.arange(n_max + 1) * 2 + 1)


def m_idx(n_max: int) -> npt.NDArray[int]:
    """Build array of multipole degrees (indexes) m from -n_max to n_max.

    :param n_max: maximum multipole order of expansion (non-negative)
    :return: array of multipole degrees

    >>> m_idx(3)
    array([ 0, -1,  0,  1, -2, -1,  0,  1,  2, -3, -2, -1,  0,  1,  2,  3])
    """
    return np.concatenate([np.arange(-i, i + 1) for i in range(n_max + 1)])


def mn_idx(n_max: int) -> tuple[tuple[int, int]]:
    """Build zip of multipoles orders n and degrees m (indexes).

    :param n_max: maximum multipole order of expansion (non-negative)
    :return: ordered m and n values
    """
    return zip(m_idx(n_max), n_idx(n_max))


def plane_wave_sfe_cft(m: int, n: int, direction: npt.NDArray[float]) -> complex:
    """Coefficient with multipole indexes m and n of plane wave decomposition in spherical wave expansion (sfe).

    :param m: multipole degree
    :param n: multipole order (non-negative)
    :param direction: unit vector cartesian coordinates of plane wave propagation direction
    :return: expansion coefficient m, n of plane wave decomposition
    """
    dir_abs, dir_phi, dir_theta = mths.dec_to_sph(direction[0], direction[1], direction[2])
    return 4 * np.pi * 1j ** n * np.conj(ss.sph_harm(m, n, dir_phi, dir_theta))


def plane_wave_sfe_cfs(direction: npt.NDArray[float], n_max) -> npt.NDArray[complex]:
    """Coefficients of plane wave decomposition in spherical wave expansion (sfe) up to n_max order.

    Coefficients are ordered by indexes (m, n) = (0, 0), (-1, 1), (0, 1), (1, 1), ..., (n_max, n_max).

    :param direction: unit vector cartesian coordinates of plane wave propagation direction
    :param n_max: maximum multipole order of expansion (non-negative)
    :return: one-dimensional array of ordered expansion coefficients
    """
    coefficients = np.zeros((n_max + 1) ** 2, dtype=complex)
    for m, n in mn_idx(n_max):
        coefficients[n ** 2 + n + m] = plane_wave_sfe_cft(m, n, direction)
    return coefficients


# todo: check and delete
# def incident_coefficients_array(direction, length, order):
#     r"""Repeated incident coefficients to multiply it with basis functions counted in all points"""
#     c_array = np.zeros(((order + 1) ** 2), dtype=complex)
#     i = 0
#     for mn in mn_idx(order):
#         c_array[i] = plane_wave_sfe_cft(mn[0], mn[1], direction)
#         i += 1
#     return np.split(np.repeat(c_array, length), (order + 1) ** 2)


def regular_wvf(
        m: int,
        n: int,
        x: float | npt.NDArray[float],
        y: float | npt.NDArray[float],
        z: float | npt.NDArray[float],
        k: float
) -> complex | npt.NDArray[complex]:
    """Regular basis spherical wave function

    Evaluate :math:`j_n(kr) Y^m_n(\\theta, \\phi)` at given points.

    :param m: multipole degree
    :param n: multipole order (non-negative)
    :param x: x-coordinate (should be of the same shape as y and z)
    :param y: y-coordinate (should be of the same shape as x and z)
    :param z: z-coordinate (should be of the same shape as x and y)
    :param k: wavenumber
    :return: wavefunction values at points
    """
    r, phi, theta = mths.dec_to_sph(x, y, z)
    return ss.spherical_jn(n, k * r) * ss.sph_harm(m, n, phi, theta)


def regular_wvfs_array(
        n_max: int,
        x: float | npt.NDArray[float],
        y: float | npt.NDArray[float],
        z: float | npt.NDArray[float],
        k: float
) -> npt.NDArray[complex]:
    """Builds array of all regular basis spherical wave functions values up to n_max

    :param n_max: maximum multipole order of expansion (non-negative)
    :param x: x-coordinate (should be of the same shape as y and z)
    :param y: x-coordinate (should be of the same shape as x and z)
    :param z: z-coordinate (should be of the same shape as x and y)
    :param k: wavenumber
    :return: values of wavefunctions at coordinate points
    """
    regular_wvfs = np.zeros(((n_max + 1) ** 2, *x.shape), dtype=complex)
    for i, (m, n) in enumerate(mn_idx(n_max)):
        regular_wvfs[i] = regular_wvf(m, n, x, y, z, k)
    return regular_wvfs


def outgoing_wvf(
        m: int,
        n: int,
        x: float | npt.NDArray[float],
        y: float | npt.NDArray[float],
        z: float | npt.NDArray[float],
        k: float
) -> complex | npt.NDArray[complex]:
    """Outgoing basis spherical wave function

    Evaluate :math:`h^1_n(kr) Y^m_n(\\theta, \\phi)` at given points.

    :param m: multipole degree
    :param n: multipole order (non-negative)
    :param x: x-coordinate (should be of the same shape as y and z)
    :param y: y-coordinate (should be of the same shape as x and z)
    :param z: z-coordinate (should be of the same shape as x and y)
    :param k: wavenumber
    :return: wavefunction values at points
    """
    r, phi, theta = mths.dec_to_sph(x, y, z)
    return mths.spherical_h1n(n, k * r) * ss.sph_harm(m, n, phi, theta)


def outgoing_wvfs_array(
        n_max: int,
        x: float | npt.NDArray[float],
        y: float | npt.NDArray[float],
        z: float | npt.NDArray[float],
        k: float
) -> npt.NDArray[complex]:
    """Builds array of all outgoing basis spherical wave functions values up to n_max

    :param n_max: maximum multipole order of expansion (non-negative)
    :param x: x-coordinate (should be of the same shape as y and z)
    :param y: x-coordinate (should be of the same shape as x and z)
    :param z: z-coordinate (should be of the same shape as x and y)
    :param k: wavenumber
    :return: values of wavefunctions at coordinate points
    """
    outgoing_wfs = np.zeros(((n_max + 1) ** 2, *x.shape), dtype=complex)
    for i, (m, n) in enumerate(mn_idx(n_max)):
        outgoing_wfs[i] = outgoing_wvf(m, n, x, y, z, k)
    return outgoing_wfs
