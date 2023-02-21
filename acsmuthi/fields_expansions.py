import numpy as np
import copy
from acsmuthi.utility import mathematics as mths, wavefunctions as wvfs


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


class SphericalWaveExpansion(FieldExpansion):
    def __init__(self, amplitude, k_l, origin, kind, order,
                 inner_r=0,
                 outer_r=np.inf,
                 coefficients=None,
                 k_t=None):
        FieldExpansion.__init__(self)
        self.ampl = amplitude
        self.k_l = k_l
        self.k_t = k_t
        self.origin = origin
        self.field = None  # todo - delete this field
        self.kind = kind  # 'regular' or 'outgoing'
        self.order = order
        self.inner_r = inner_r
        self.outer_r = outer_r
        self.coefficients = coefficients

    def diverging(self, x, y, z):
        r = np.sqrt((x - self.origin[0]) ** 2 + (y - self.origin[1]) ** 2
                    + (z - self.origin[2]) ** 2)
        if self.kind == 'regular':
            return r >= self.outer_r
        if self.kind == 'outgoing':
            return r <= self.inner_r
        else:
            return None

    def compute_pressure_field(self, x, y, z):
        coefficients_array = np.split(np.repeat(self.coefficients, len(x)), (self.order + 1) ** 2)
        if self.kind == 'regular':
            wvf = wvfs.regular_wvfs_array
        elif self.kind == 'outgoing':
            wvf = wvfs.outgoing_wvfs_array
        # xr = x[self.valid(x, y, z)] - self.origin[0]              # todo: it should be like this
        # yr = y[self.valid(x, y, z)] - self.origin[1]
        # zr = z[self.valid(x, y, z)] - self.origin[2]
        # wave_functions_array = wvf(self.order, xr, yr, zr, self.k_l)
        wave_functions_array = wvf(self.order, x - self.origin[0], y - self.origin[1], z - self.origin[2], self.k_l)
        field_array = coefficients_array * wave_functions_array
        self.field = self.ampl * mths.multipoles_fsum(field_array, len(x))
        return self.field

    def compatible(self, other):
        return (type(other).__name__ == "SphericalWaveExpansion"
                and self.k_l == other.k_l
                and self.k_t == other.k_t
                and self.order == other.order
                and self.kind == other.kind
                and np.array_equal(self.origin, other.origin))

    def __add__(self, other):
        if not self.compatible(other):
            raise ValueError('SphericalWaveExpansions are inconsistent.')
        swe_sum = SphericalWaveExpansion(amplitude=self.ampl,
                                         k_l=self.k_l,
                                         k_t=self.k_t,
                                         origin=self.origin,
                                         kind=self.kind,
                                         inner_r=max(self.inner_r, other.inner_r),
                                         outer_r=min(self.outer_r, other.outer_r))
        swe_sum.coefficients = self.coefficients + other.coefficients
        return swe_sum


class PlaneWave(SphericalWaveExpansion):
    def __init__(self, amplitude, k_l, origin, kind, order, direction):
        super().__init__(amplitude, k_l, origin, kind, order)
        self.dir = direction
        self.coefficients = wvfs.incident_coefficients(direction, order)
        self.exact_field = None

    def compute_exact_field(self, x, y, z):
        self.exact_field = self.ampl * np.exp(1j * self.k_l * (self.dir[0] * x + self.dir[1] * y + self.dir[2] * z))

    def intensity(self, density, sound_speed):
        return self.ampl ** 2 / (2 * density * sound_speed)


class StandingWave(SphericalWaveExpansion):
    def __init__(self, amplitude, k_l, origin, kind, order, direction):
        super().__init__(amplitude, k_l, origin, kind, order)
        self.dir = direction
        self.coefficients = wvfs.incident_coefficients(direction, order) + wvfs.incident_coefficients(-direction, order)
        self.exact_field = None

    def compute_exact_field(self, x, y, z):
        self.exact_field = self.ampl * np.exp(1j * self.k_l * (self.dir[0] * x + self.dir[1] * y + self.dir[2] * z)) + \
                           self.ampl * np.exp(1j * self.k_l * (-self.dir[0] * x - self.dir[1] * y - self.dir[2] * z))

    def intensity(self, density, sound_speed):
        return self.ampl ** 2 / (2 * density * sound_speed)
