import numpy as np
import copy
from typing import Literal
from acsmuthi.utility import wavefunctions as wvfs


class FieldExpansion:
    def __init__(self):
        self.validity_conditions = []   # todo ???

    def valid(self, x, y, z):
        ret = np.ones(x.shape, dtype=bool)
        for check in self.validity_conditions:
            ret = np.logical_and(ret, check(x, y, z))
        return ret

    def diverging(self, x, y, z):
        pass

    def compute_pressure_field(self, x, y, z):
        pass

    def velocity_field(self, x, y, z):
        pass


class SphericalWaveExpansion(FieldExpansion):
    def __init__(self, amplitude, k, origin, kind: Literal['regular', 'outgoing'], order, inner_r=0, outer_r=np.inf, coefficients=None):
        FieldExpansion.__init__(self)
        self.ampl = amplitude
        self.k = k
        self.origin = origin
        self.kind = kind
        self.order = order
        self.inner_r = inner_r
        self.outer_r = outer_r
        self.coefficients = coefficients

    def diverging(self, x, y, z):
        r = np.sqrt((x - self.origin[0]) ** 2 + (y - self.origin[1]) ** 2 + (z - self.origin[2]) ** 2)
        if self.kind == 'regular':
            return r >= self.outer_r
        if self.kind == 'outgoing':
            return r <= self.inner_r
        else:
            return None

    def compute_pressure_field(self, x, y, z):
        if self.kind == 'regular':
            wvf = wvfs.regular_wvfs_array
        elif self.kind == 'outgoing':
            wvf = wvfs.outgoing_wvfs_array
        xr, yr, zr = x - self.origin[0], y - self.origin[1], z - self.origin[2]
        r = np.sqrt(xr ** 2 + yr ** 2 + zr ** 2)
        wave_functions_array = wvf(self.order, xr, yr, zr, self.k)
        coefficients_array = np.broadcast_to(self.coefficients, wave_functions_array.T.shape).T
        field_array = coefficients_array * wave_functions_array
        field = self.ampl * np.sum(field_array, axis=0)
        return np.where((r >= self.inner_r) & (r < self.outer_r), field, 0)

    def compatible(self, other):
        return (type(other).__name__ == "SphericalWaveExpansion"
                and self.k == other.k
                and self.order == other.order
                and self.kind == other.kind
                and np.array_equal(self.origin, other.origin))

    def __add__(self, other):
        if not self.compatible(other):
            raise ValueError('SphericalWaveExpansions are inconsistent.')
        swe_sum = SphericalWaveExpansion(
            amplitude=self.ampl,
            k=self.k,
            origin=self.origin,
            kind=self.kind, order=self.order,
            inner_r=max(self.inner_r, other.inner_r),
            outer_r=min(self.outer_r, other.outer_r)
        )
        swe_sum.coefficients = self.coefficients + other.coefficients
        return swe_sum
