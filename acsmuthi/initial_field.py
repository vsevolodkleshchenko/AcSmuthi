import numpy as np

import acsmuthi.fields_expansions as fldsex
import acsmuthi.utility.wavefunctions as wvfs
import acsmuthi.linear_system.coupling_matrix as cmt


class InitialField:
    def __init__(self, k, amplitude):
        self.k = k
        self.amplitude = amplitude
        self.validity_conditions = []

    def piecewice_field_expansion(self, particle, medium):
        pass

    def spherical_wave_expansion(self, origin, order):
        pass


class PlaneWave(InitialField):
    def __init__(self, k, amplitude, direction, reference_point=None):
        InitialField.__init__(self, k=k, amplitude=amplitude)
        self.direction = direction
        if reference_point is None:
            self.reference_point = np.array([0, 0, 0])
        else:
            self.reference_point = reference_point

    def spherical_wave_expansion(self, origin, order):
        reference_coefficients = wvfs.incident_coefficients(self.direction, order)
        if np.array_equal(origin, self.reference_point):
            coefficients = reference_coefficients
        else:
            coefficients = np.exp(1j * self.k * self.direction @ (origin - self.reference_point)) * \
                           reference_coefficients
        return fldsex.SphericalWaveExpansion(amplitude=self.amplitude, k=self.k, origin=origin, kind='regular',
                                             order=order, coefficients=coefficients)

    def compute_exact_field(self, x, y, z):
        exact_field = self.amplitude * np.exp(1j * self.k * (
                self.direction[0] * (x - self.reference_point[0]) +
                self.direction[1] * (y - self.reference_point[1]) +
                self.direction[2] * (z - self.reference_point[2])
        ))
        return exact_field

    def intensity(self, density, sound_speed):
        return self.amplitude ** 2 / (2 * density * sound_speed)


class StandingWave(InitialField):
    def __init__(self, k, amplitude, direction, reference_point=None):
        InitialField.__init__(self, k=k, amplitude=amplitude)
        self.direction = direction
        if reference_point is None:
            self.reference_point = np.array([0, 0, 0])
        else:
            self.reference_point = reference_point

    def spherical_wave_expansion(self, origin, order):
        reference_coefficients_forward = wvfs.incident_coefficients(self.direction, order)
        reference_coefficients_backward = wvfs.incident_coefficients(-self.direction, order)
        if np.array_equal(origin, self.reference_point):
            coefficients = reference_coefficients_forward + reference_coefficients_backward
        else:
            phase_forward = np.exp(1j * self.k * self.direction @ (origin - self.reference_point))
            phase_backward = np.exp(-1j * self.k * self.direction @ (origin - self.reference_point))
            coefficients = phase_forward * reference_coefficients_forward + \
                           phase_backward * reference_coefficients_backward
        return fldsex.SphericalWaveExpansion(amplitude=self.amplitude, k=self.k, origin=origin, kind='regular',
                                             order=order, coefficients=coefficients)

    def compute_exact_field(self, x, y, z):
        exact_field = self.amplitude * np.exp(1j * self.k * (
                self.direction[0] * (x - self.reference_point[0]) +
                self.direction[1] * (y - self.reference_point[1]) +
                self.direction[2] * (z - self.reference_point[2])
        ))
        exact_field += self.amplitude * np.exp(1j * self.k * (
                -self.direction[0] * (x - self.reference_point[0]) +
                -self.direction[1] * (y - self.reference_point[1]) +
                -self.direction[2] * (z - self.reference_point[2])
        ))
        return exact_field

    def intensity(self, density, sound_speed):
        return self.amplitude ** 2 / (2 * density * sound_speed)