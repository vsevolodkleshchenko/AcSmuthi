import numpy as np
import copy
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
    def __init__(self, amplitude, k, origin, kind, order, inner_r=0, outer_r=np.inf, coefficients=None):
        FieldExpansion.__init__(self)
        self.ampl = amplitude
        self.k = k
        self.origin = origin
        self.kind = kind  # 'regular' or 'outgoing'
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


class PiecewiseFieldExpansion(FieldExpansion):
    def __init__(self):
        FieldExpansion.__init__(self)
        self.expansion_list = []

    def valid(self, x, y, z):
        vld = np.zeros(x.shape, dtype=bool)
        for fex in self.expansion_list:
            vld = np.logical_or(vld, fex.valid(x, y, z))
        vld = np.logical_and(vld, FieldExpansion.valid(self, x, y, z))
        return vld

    def diverging(self, x, y, z):
        dvg = np.zeros(x.shape, dtype=bool)
        for fex in self.expansion_list:
            dvg = np.logical_and(dvg, fex.diverging(x, y, z))
        return dvg

    def compute_pressure_field(self, x, y, z):
        # x, y, z = np.array(x, ndmin=1), np.array(y, ndmin=1), np.array(z, ndmin=1)
        p = np.zeros(x.shape, dtype=complex)
        vld = self.valid(x, y, z)
        for fex in self.expansion_list:
            dp = fex.compute_pressure_field(x, y, z)
            p[vld] = p[vld] + dp[vld]
        return p

    def compatible(self):
        return True

    def __add__(self, other):
        # todo: testing
        pfe_sum = PiecewiseFieldExpansion()

        if type(other).__name__ == "PiecewiseFieldExpansion":
            added = [False for other_fex in other.expansion_list]
            for self_fex in self.expansion_list:
                fex = copy.deepcopy(self_fex)
                for i, other_fex in enumerate(other.expansion_list):
                    if (not added[i]) and self_fex.compatible(other_fex):
                        fex = fex + other_fex
                        added[i] = True
                pfe_sum.expansion_list.append(fex)
            for i, other_fex in enumerate(other.expansion_list):
                if not added[i]:
                    pfe_sum.expansion_list.append(other_fex)
        else:
            added = False
            for self_fex in self.expansion_list:
                fex = copy.deepcopy(self_fex)
                if (not added) and fex.compatible(other):
                    pfe_sum.expansion_list.append(fex + other)
                    added = True
                else:
                    pfe_sum.expansion_list.append(fex)
            if not added:
                pfe_sum.expansion_list.append(other)

        return pfe_sum
