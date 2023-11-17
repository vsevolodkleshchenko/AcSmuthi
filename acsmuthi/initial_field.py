import numpy as np

import acsmuthi.fields_expansions as fldsex
import acsmuthi.utility.wavefunctions as wvfs
from acsmuthi.linear_system.coupling.coupling_basics import fresnel_r, fresnel_r_hard


class InitialField:
    def __init__(self, k, amplitude):
        self.k = k
        self.amplitude = amplitude
        self.validity_conditions = []

    def piecewice_field_expansion(self, particle, medium):
        pass

    def spherical_wave_expansion(self, origin, medium, order):
        pass


class PlaneWave(InitialField):
    def __init__(self, k, amplitude, direction, reference_point=None):
        InitialField.__init__(self, k=k, amplitude=amplitude)
        self.direction = direction
        if reference_point is None:
            self.reference_point = np.array([0, 0, 0])
        else:
            self.reference_point = reference_point

    def spherical_wave_expansion(self, origin, medium, order):
        reference_coefficients = wvfs.incident_coefficients(self.direction, order)

        if np.array_equal(origin, self.reference_point):
            coefficients = reference_coefficients
        else:
            coefficients = np.exp(1j * self.k * self.direction @ (origin - self.reference_point)) * \
                           reference_coefficients

        if medium.is_substrate and self.direction[2] < 0:
            reflection_phase = np.exp(-2j * self.direction[2] * self.k * self.reference_point[2])
            reflected_direction = np.array([self.direction[0], self.direction[1], -self.direction[2]])
            if medium.hard_substrate:
                r = fresnel_r_hard()
            else:
                k_substrate = self.k * medium.cp / medium.cp_sub
                r = fresnel_r(abs(self.k * self.direction[0]), self.k, k_substrate, medium.density, medium.density_sub)
            reflected_coefficients = r * reflection_phase * wvfs.incident_coefficients(reflected_direction, order)
            if not np.array_equal(origin, self.reference_point):
                reflected_coefficients *= np.exp(1j * self.k * reflected_direction @ (origin - self.reference_point))

            coefficients += reflected_coefficients

        return fldsex.SphericalWaveExpansion(amplitude=self.amplitude, k=self.k, origin=origin, kind='regular',
                                             order=order, coefficients=coefficients)

    def compute_exact_field(self, x, y, z, medium):
        exact_field = self.amplitude * np.exp(1j * self.k * (
                self.direction[0] * (x - self.reference_point[0]) +
                self.direction[1] * (y - self.reference_point[1]) +
                self.direction[2] * (z - self.reference_point[2])
        ))
        if medium.is_substrate:
            if medium.hard_substrate:
                r = fresnel_r_hard()
            else:
                k_substrate = self.k * medium.cp / medium.cp_sub
                r = fresnel_r(abs(self.k * self.direction[0]), self.k, k_substrate, medium.density, medium.density_sub)
            if self.direction[2] < 0:
                exact_field += r * self.amplitude * np.exp(1j * self.k * (
                    self.direction[0] * (x - self.reference_point[0]) +
                    self.direction[1] * (y - self.reference_point[1]) -
                    self.direction[2] * (z - self.reference_point[2])
                )) * np.exp(-2j * self.direction[2] * self.k * self.reference_point[2])
            exact_field = np.where(z >= 0, exact_field, 0)
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

    def spherical_wave_expansion(self, origin, medium, order):
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