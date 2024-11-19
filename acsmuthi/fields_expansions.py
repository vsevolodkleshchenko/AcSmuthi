from abc import ABC, abstractmethod
from typing import Literal, Self

import numpy as np

from acsmuthi.utility.wavefunctions import mn_idx, outgoing_wvf, regular_wvf


class FieldExpansion(ABC):
    """Abstract class for field expansions."""

    # def __init__(self):
    #     """Field expansion constructor.
    #
    #     validity_conditions data attribute represent the spatial validity of the field representation.
    #     """
    #     # self.validity_conditions = []   # todo: delete or use somewhere ???

    @abstractmethod
    def valid(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """Test if points are in definition range of the expansion.

        :param x: x-coordinates of query points
        :param y: y-coordinates of query points
        :param z: z-coordinates of query points
        :return: array indicating if points are inside definition domain
        """
        pass
        # validity = np.ones(x.shape, dtype=bool)
        # for check in self.validity_conditions:
        #     validity = np.logical_and(validity, check(x, y, z))
        # return validity

    @abstractmethod
    def pressure_field(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """Evaluate pressure field.

        :param x: x-coordinates of query points
        :param y: y-coordinates of query points
        :param z: z-coordinates of query points
        :return: pressure scalar field
        """
        pass


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
            lower_z: float = - np.inf,
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
        self.lower_z = lower_z
        self.coefficients = coefficients

    def valid(self, x, y, z):
        """Test if points are in definition range of the expansion.

        :param x: x-coordinates of query points
        :param y: y-coordinates of query points
        :param z: z-coordinates of query points
        :return: array indicating if points are inside definition domain
        """
        return np.logical_and(z >= self.lower_z, np.logical_not(self.diverging(x, y, z)))

    def diverging(self, x, y, z):
        """Test if points are in domain where expansion could diverge.

        :param x: x-coordinates of query points
        :param y: y-coordinates of query points
        :param z: z-coordinates of query points
        :return: array indicating if points are outside domain of convergence
        """
        xr, yr, zr = x - self.reference_point[0], y - self.reference_point[1], z - self.reference_point[2]
        r = np.sqrt(xr ** 2 + yr ** 2 + zr ** 2)
        if self.kind == 'regular':
            return r >= self.outer_r    # todo: decide >, >= etc.
        if self.kind == 'outgoing':
            return r < self.inner_r

    def pressure_field(self, x, y, z):
        """Pressure field evaluation using spherical basis functions and expansion coefficients.
        """
        vld = self.valid(x, y, z)
        wave_functions = np.zeros(((self.n_max + 1) ** 2, *x.shape), dtype=complex)
        xr, yr, zr = x - self.reference_point[0], y - self.reference_point[1], z - self.reference_point[2]
        for i, (m, n) in enumerate(mn_idx(self.n_max)):
            if self.kind == 'regular':
                wave_functions[i, vld] = regular_wvf(m, n, xr[vld], yr[vld], zr[vld], self.k)
            elif self.kind == 'outgoing':
                wave_functions[i, vld] = outgoing_wvf(m, n, xr[vld], yr[vld], zr[vld], self.k)

        coefficients = np.broadcast_to(self.coefficients, wave_functions.T.shape).T

        multipole_fields = coefficients * wave_functions
        return self.ampl * np.sum(multipole_fields, axis=0)

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
