from abc import ABC, abstractmethod
import numpy as np
import copy
from typing import Literal, Self

from numpy import typing as npt

from acsmuthi.utility import wavefunctions as wvfs
from acsmuthi.utility.wavefunctions import mn_idx, outgoing_wvf, regular_wvf


# todo: rewrite


class FieldExpansion(ABC):
    """Abstract class for field expansions."""

    def __init__(self):
        """Field expansion constructor.

        validity_conditions data attribute represent the spatial validity of the field representation.
        """
        self.validity_conditions = []   # todo: delete or use somewhere ???

    def is_valid(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """Test if points are in definition range of the expansion.

        :param x: x-coordinates of query points
        :param y: y-coordinates of query points
        :param z: z-coordinates of query points
        :return: array indicating if points are inside definition domain
        """
        validity = np.ones(x.shape, dtype=bool)
        for check in self.validity_conditions:
            validity = np.logical_and(validity, check(x, y, z))
        return validity

    # @abstractmethod
    # def diverging(self, x, y, z):
    #     pass

    @abstractmethod
    def pressure_field(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """Abstract class for pressure field evaluation.

        :param x: x-coordinates of query points
        :param y: y-coordinates of query points
        :param z: z-coordinates of query points
        :return: pressure scalar field
        """
        pass

    # @abstractmethod
    # def velocity_field(self, x, y, z):
    #     pass


class SphericalWaveExpansion(FieldExpansion):
    """Representation of field in spherical waves (multipole) expansion.

    :math:`\\sum_{n,m} c^n_m z_n(kr) Y^m_n(\\vec{r})`, where :math:`c^m_n` are the expansion coefficients, and
    :math:`z_n(x) = h_n^1(x)` for outgoing (scattered) field and `z_n(x) = j_n(x)` for incoming (incident) field.

    """
    def __init__(
            self,
            amplitude: float,
            k: float,
            reference_point: np.ndarray,
            kind: Literal['regular', 'outgoing'],
            n_max: int,
            inner_r: float = 0,
            outer_r: float = np.inf,
            coefficients: np.ndarray = None
    ):
        """Spherical field expansion constructor.

        :param amplitude: pressure field amplitude  # todo: move from pressure to something else
        :param k: wavenumber in medium where field expansion is valid
        :param reference_point: coordinates of point relative to which the spherical waves are considered.
        :param kind: indicates whether the field is outgoing or incoming (regular)
        :param n_max: maximum multipole order of spherical expansion (non-negative)
        :param inner_r: radius inside which the expansion diverges
        :param outer_r:radius outside which the expansion diverges
        :param coefficients: expansion coefficients ordered by indexes n, m
        """
        FieldExpansion.__init__(self)
        self.ampl = amplitude
        self.k = k
        self.reference_point = reference_point
        self.kind = kind
        self.n_max = n_max
        self.inner_r = inner_r
        self.outer_r = outer_r
        self.coefficients = coefficients

    # def diverging(self, x, y, z):
    #     r = np.sqrt((x - self.reference_point[0]) ** 2 + (y - self.reference_point[1]) ** 2 + (z - self.reference_point[2]) ** 2)
    #     if self.kind == 'regular':
    #         return r >= self.outer_r
    #     if self.kind == 'outgoing':
    #         return r <= self.inner_r
    #     else:
    #         return None

    def pressure_field(self, x, y, z):  # todo: compare with direct realization
        """Pressure field evaluation using spherical basis functions and expansion coefficients."""
        if self.kind == 'regular':
            wvf = _regular_wvfs_array
        elif self.kind == 'outgoing':
            wvf = _outgoing_wvfs_array
        xr, yr, zr = x - self.reference_point[0], y - self.reference_point[1], z - self.reference_point[2]
        r = np.sqrt(xr ** 2 + yr ** 2 + zr ** 2)
        wave_functions_array = wvf(self.n_max, xr, yr, zr, self.k)
        coefficients_array = np.broadcast_to(self.coefficients, wave_functions_array.T.shape).T
        field_array = coefficients_array * wave_functions_array
        field = self.ampl * np.sum(field_array, axis=0)
        return np.where((r >= self.inner_r) & (r < self.outer_r), field, 0)

    # def compatible(self, other: Self):
    #     return (type(other).__name__ == "SphericalWaveExpansion"  # todo: maybe it is possible to do it easier
    #             and self.k == other.k
    #             and self.n_max == other.n_max
    #             and self.kind == other.kind
    #             and np.array_equal(self.reference_point, other.reference_point))
    #
    # def __add__(self, other: Self):
    #     if not self.compatible(other):
    #         raise ValueError('SphericalWaveExpansions are inconsistent.')
    #     swe_sum = SphericalWaveExpansion(amplitude=self.ampl, k=self.k, reference_point=self.reference_point,
    #                                      kind=self.kind, n_max=self.n_max, inner_r=max(self.inner_r, other.inner_r),
    #                                      outer_r=min(self.outer_r, other.outer_r))
    #     swe_sum.coefficients = self.coefficients + other.coefficients
    #     return swe_sum


def _regular_wvfs_array(
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


def _outgoing_wvfs_array(
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
