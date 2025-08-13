from abc import ABC, abstractmethod
import numpy as np

import acsmuthi.fields_expansions as fldsex
import acsmuthi.utility.wavefunctions as wvfs
from acsmuthi.medium import MediumSystem


class InitialField(ABC):     # todo: frequency, not k; validity conditions
    """Abstract class for initial (incident) fields.
    """
    def __init__(self, frequency: float, amplitude: float):
        """Initial field constructor.

        Args:
            frequency (float): wave frequency in Hz
            amplitude (float): wave amplitude in Pa
        """
        self.freq = frequency
        self.amplitude = amplitude
        self.validity_conditions = []

    # @abstractmethod
    # def piecewice_field_expansion(self, particle, medium):  # todo: delete if it is not used
    #     pass

    @abstractmethod
    def spherical_wave_expansion(
        self,
        reference_point: np.ndarray,
        medium: MediumSystem,
        order: int,
    ) -> fldsex.SphericalWaveExpansion:
        """Compute spherical wave expansion of the initial field.
        """
        pass


class PlaneWave(InitialField):
    """Plane wave pressure initial field.
    """
    def __init__(
        self,
        frequency: float,
        amplitude: float,
        direction: np.ndarray,
        reference_point: np.ndarray | None = None
    ):
        """Plane wave constructor.

        Args:
            frequency (float): frequency of the wave in Hz
            amplitude (float): amplitude of the wave in Pa
            direction (np.ndarray): unit vector indicating the direction of the wave propagation
            reference_point (optional): coordinates of point relative to which the spherical waves are considered
        """
        InitialField.__init__(self, frequency=frequency, amplitude=amplitude)
        self.direction = direction
        if reference_point is None:
            self.reference_point = np.array([0, 0, 0])
        else:
            self.reference_point = reference_point

    def spherical_wave_expansion(   # todo: think about args; transfer pwe to sfe????
        self,
        reference_point: np.ndarray,
        medium: MediumSystem,
        order: int,
    ) -> fldsex.SphericalWaveExpansion:
        """Compute spherical wave expansion of the plane wave pressure field.

        Args:
            medium (MediumSystem): medium system where the field expansion is computed
            order (int): multipole order of spherical expansion
            reference_point (np.ndarray): reference point for the spherical wave expansion of the plane wave
        """
        k = medium.sur_medium.wavenumber(self.freq)
        reference_coefficients = wvfs.plane_wave_sfe_cfs(self.direction, order)

        if np.array_equal(reference_point, self.reference_point):
            coefficients = reference_coefficients
        else:
            kr = k * self.direction @ (reference_point - self.reference_point)
            coefficients = np.exp(1j * kr) * reference_coefficients

        if medium.is_substrate and self.direction[2] < 0:
            reflection_phase = np.exp(-2j * self.direction[2] * k * self.reference_point[2])
            reflected_direction = np.array([self.direction[0], self.direction[1], -self.direction[2]])
            r = medium.fresnel_r(k_parallel=k * np.linalg.norm(self.direction[:-1]), frequency=self.freq)
            reflected_pw_cfs = wvfs.plane_wave_sfe_cfs(direction=reflected_direction, n_max=order)
            reflected_coefficients = r * reflection_phase * reflected_pw_cfs
            if not np.array_equal(reference_point, self.reference_point):
                kr = k * reflected_direction @ (reference_point - self.reference_point)
                reflected_coefficients *= np.exp(1j * kr)

            coefficients += reflected_coefficients

        return fldsex.SphericalWaveExpansion(
            amplitude=self.amplitude,
            k=k,
            reference_point=reference_point,
            kind='regular',
            n_max=order,
            coefficients=coefficients
        )

    def pressure_field(  # todo: through the pfe
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
        z: float | np.ndarray,
        medium: MediumSystem
    ) -> np.ndarray:
        """Compute the exact pressure field of the plane wave in medium in given coordinates.
        """
        k = medium.sur_medium.wavenumber(self.freq)
        exact_field = self.amplitude * np.exp(
            1j * k * (
                self.direction[0] * (x - self.reference_point[0]) +
                self.direction[1] * (y - self.reference_point[1]) +
                self.direction[2] * (z - self.reference_point[2])
            )
        )
        if medium.is_substrate:
            r = medium.fresnel_r(k_parallel=k * np.linalg.norm(self.direction[:-1]), frequency=self.freq)

            if self.direction[2] < 0:
                exact_field += r * self.amplitude * np.exp(
                    1j * k * (
                        self.direction[0] * (x - self.reference_point[0]) +
                        self.direction[1] * (y - self.reference_point[1]) -
                        self.direction[2] * (z - self.reference_point[2])
                    )
                ) * np.exp(-2j * self.direction[2] * k * self.reference_point[2])
            exact_field = np.where(z >= 0, exact_field, 0)
        return exact_field

    def intensity(self, medium: MediumSystem) -> float:
        return self.amplitude ** 2 / (2 * medium.sur_medium.density * medium.sur_medium.c_longitudinal)


# class StandingWave(InitialField):
# #  todo: it doesn't work - delete or change or do something; maybe make summation method
#     def __init__(self, k, amplitude, direction, reference_point=None):
#         InitialField.__init__(self, k=k, amplitude=amplitude)
#         self.direction = direction
#         if reference_point is None:
#             self.reference_point = np.array([0, 0, 0])
#         else:
#             self.reference_point = reference_point

#     def spherical_wave_expansion(self, origin, medium, order):
#         reference_coefficients_forward = wvfs.plane_wave_sfe_cfs(self.direction, order)
#         reference_coefficients_backward = wvfs.plane_wave_sfe_cfs(-self.direction, order)
#         if np.array_equal(origin, self.reference_point):
#             coefficients = reference_coefficients_forward + reference_coefficients_backward
#         else:
#             phase_forward = np.exp(1j * self.k * self.direction @ (origin - self.reference_point))
#             phase_backward = np.exp(-1j * self.k * self.direction @ (origin - self.reference_point))
#             coefficients = phase_forward * reference_coefficients_forward + \
#                            phase_backward * reference_coefficients_backward
#         return fldsex.SphericalWaveExpansion(amplitude=self.amplitude, k=self.k,
#                                               reference_point=origin, kind='regular',
#                                              n_max=order, coefficients=coefficients)

#     def compute_exact_field(self, x, y, z):
#         exact_field = self.amplitude * np.exp(1j * self.k * (
#                 self.direction[0] * (x - self.reference_point[0]) +
#                 self.direction[1] * (y - self.reference_point[1]) +
#                 self.direction[2] * (z - self.reference_point[2])
#         ))
#         exact_field += self.amplitude * np.exp(1j * self.k * (
#                 -self.direction[0] * (x - self.reference_point[0]) +
#                 -self.direction[1] * (y - self.reference_point[1]) +
#                 -self.direction[2] * (z - self.reference_point[2])
#         ))
#         return exact_field

#     def intensity(self, density, sound_speed):
#         return self.amplitude ** 2 / (2 * density * sound_speed)
