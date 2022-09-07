import numpy as np
from utility import wavefunctions as wvfs, mathematics as mths


class SphericalWaveExpansion:
    def __init__(self, amplitude, k_l, origin, kind, order, coefficients=None, k_t=None):
        self.ampl = amplitude
        self.k_l = k_l
        self.k_t = k_t
        self.origin = origin
        self.field = None
        self.kind = kind  # 'regular' or 'outgoing'
        self.order = order
        self.coefficients = coefficients

    def compute_pressure_field(self, x, y, z):
        coefficients_array = np.split(np.repeat(self.coefficients, len(x)), (self.order + 1) ** 2)
        if self.kind == 'regular':
            wvf = wvfs.regular_wvfs_array
        else:
            wvf = wvfs.outgoing_wvfs_array
        wave_functions_array = wvf(self.order, x - self.origin[0], y - self.origin[1], z - self.origin[2], self.k_l)
        field_array = coefficients_array * wave_functions_array
        self.field = self.ampl * mths.multipoles_fsum(field_array, len(x))


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
