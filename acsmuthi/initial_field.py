from abc import ABC, abstractmethod
import numpy as np

import acsmuthi.fields_expansions as fldsex
from acsmuthi.fields_expansions import pwe_to_swe
from acsmuthi.medium import MediumSystem


class InitialField(ABC):
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
            self.reference_point = np.zeros(3)
        else:
            self.reference_point = reference_point

    def plane_wave_expansion(
        self,
        medium: MediumSystem
    ) -> tuple[fldsex.PlaneWaveExpansion]:
        """Compute plane wave expansion of the plane wave pressure field.

        Args:
            medium: medium system where the field expansion is computed
        """
        k = medium.sur_medium.wavenumber(self.freq)
        k_vector = k * self.direction
        incident_pwe = fldsex.PlaneWaveExpansion(
            k_vector=k_vector,
            amplitude=self.amplitude,
            reference_point=self.reference_point,
            lower_z=0 if medium.is_substrate and self.direction[2] < 0 else -np.inf,
            upper_z=np.inf
        )

        if medium.is_substrate and self.direction[2] < 0:
            reflected_pwe = reflect(
                pwe=incident_pwe,
                medium=medium,
                freq=self.freq
            )
            return incident_pwe, reflected_pwe

        else:
            return tuple([incident_pwe])

    def spherical_wave_expansion(   # todo: think about args
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
        sw_expansions = []
        for pwe in self.plane_wave_expansion(medium=medium):
            swe = pwe_to_swe(pwe=pwe, reference_point=reference_point, n_max=order)
            sw_expansions.append(swe)

        # extremely harcoded - todo: fix!!!!!!
        if medium.is_substrate and self.direction[2] < 0:
            swe_inc, swe_ref = sw_expansions
            ampl = swe_inc.ampl
            r_eikz_factor = swe_ref.ampl / ampl
            swe_ref.coefficients *= r_eikz_factor
            swe_ref.ampl = ampl
            return swe_inc + swe_ref
        else:
            return sw_expansions[0]

    def pressure_field(
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
        z: float | np.ndarray,
        medium: MediumSystem
    ) -> np.ndarray:
        """Compute the exact pressure field of the plane wave in medium in given coordinates.
        """
        exact_field = np.zeros_like(x, dtype=complex)
        for pwe in self.plane_wave_expansion(medium=medium):
            exact_field += pwe.pressure_field(x=x, y=y, z=z)
        return exact_field

    def intensity(self, medium: MediumSystem) -> float:
        return self.amplitude ** 2 / (2 * medium.sur_medium.density * medium.sur_medium.c_longitudinal)


def reflect(
    pwe: fldsex.PlaneWaveExpansion,
    medium: MediumSystem,
    freq: float,
) -> fldsex.PlaneWaveExpansion:
    """Reflect downgoing wave defined by plane wave expansion from a substrate.
    """
    assert medium.is_substrate, "Medium must contain substrate"
    assert pwe.kind == 'downgoing', "Only downgoing plane waves can be reflected"

    r = medium.fresnel_r(k_parallel=pwe.k_parallel, frequency=freq)
    kvec_r = np.array([pwe.k_vec[0], pwe.k_vec[1], -pwe.k_vec[2]])
    phase_r = np.exp(-2j * pwe.k_vec[2] * pwe.reference_point[2])
    amplitude_r = r * pwe.ampl * phase_r
    return fldsex.PlaneWaveExpansion(
        k_vector=kvec_r,
        amplitude=amplitude_r,
        reference_point=pwe.reference_point,
        lower_z=0,
        upper_z=np.inf
    )
