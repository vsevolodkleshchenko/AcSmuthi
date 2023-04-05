import numpy as np

import acsmuthi.fields_expansions as fldsex
import acsmuthi.utility.wavefunctions as wvfs
import acsmuthi.linear_system.coupling_matrix as cmt


class InitialField:
    def __init__(self, k_l):
        self.k_l = k_l
        self.validity_conditions = []

    def piecewice_field_expansion(self, particle, medium):
        pass

    def spherical_field_expansion(self, medium, order):
        pass


class PlaneWave(InitialField):
    def __init__(self, k_l, amplitude, direction, origin=None):
        InitialField.__init__(self, k_l=k_l)
        self.ampl = amplitude
        self.dir = direction
        if origin is None:
            self.reference_point = np.array([0, 0, 0])
        self.exact_field = None

    def spherical_wave_expansion(self, origin, order):
        base_coefficients = wvfs.incident_coefficients(self.dir, order)
        if np.array_equal(origin, self.reference_point):
            coefficients = base_coefficients
        else:
            coefficients = cmt.translation_block(order, self.k_l, origin - self.reference_point) @ base_coefficients
        return fldsex.SphericalWaveExpansion(amplitude=self.ampl,
                                             k_l=self.k_l,
                                             origin=origin,
                                             kind='regular',
                                             order=order,
                                             coefficients=coefficients)

    def compute_exact_field(self, x, y, z):
        self.exact_field = self.ampl * np.exp(1j * self.k_l * (self.dir[0] * (x - self.reference_point[0]) +
                                                               self.dir[1] * (y - self.reference_point[1]) +
                                                               self.dir[2] * (z - self.reference_point[2])))
        return self.exact_field

    def intensity(self, density, sound_speed):
        return self.ampl ** 2 / (2 * density * sound_speed)


class StandingWave(InitialField):
    def __init__(self, k_l, amplitude, direction, origin=None):
        InitialField.__init__(self, k_l=k_l)
        self.ampl = amplitude
        self.dir = direction
        if origin is None:
            self.reference_point = np.array([0, 0, 0])
        self.exact_field = None

    def spherical_wave_expansion(self, origin, order):
        base_coefficients = wvfs.incident_coefficients(self.dir, order) + wvfs.incident_coefficients(-self.dir, order)
        if np.array_equal(origin, self.reference_point):
            coefficients = base_coefficients
        else:
            coefficients = cmt.translation_block(order, self.k_l, origin - self.reference_point) @ base_coefficients
        return fldsex.SphericalWaveExpansion(amplitude=self.ampl,
                                             k_l=self.k_l,
                                             origin=origin,
                                             kind='regular',
                                             order=order,
                                             coefficients=coefficients)

    def compute_exact_field(self, x, y, z):
        self.exact_field = self.ampl * np.exp(1j * self.k_l * (self.dir[0] * (x - self.reference_point[0]) +
                                                               self.dir[1] * (y - self.reference_point[1]) +
                                                               self.dir[2] * (z - self.reference_point[2])))
        self.exact_field += self.ampl * np.exp(1j * self.k_l * (-self.dir[0] * (x - self.reference_point[0]) +
                                                                -self.dir[1] * (y - self.reference_point[1]) +
                                                                -self.dir[2] * (z - self.reference_point[2])))
        return self.exact_field

    def intensity(self, density, sound_speed):
        return self.ampl ** 2 / (2 * density * sound_speed)