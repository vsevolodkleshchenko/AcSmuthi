from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class Medium(ABC):  # todo: add impedance property
    """Abstract class for medium.
    """
    @abstractmethod
    def wavenumber(self, frequency: float):
        """Wavenumber in medium.
        """
        pass


# class MediumOld(Medium):
#     def __init__(
#             self,
#             density: float,
#             sound_speed_longitudinal: float,
#             hard_substrate: bool = False,
#             substrate_density: float | None = None,
#             substrate_velocity: float | None = None,
#             substrate_velocity_shear: float | None = None
#     ):
#         self.density = density
#         self.c_longitudinal = sound_speed_longitudinal
#         if hard_substrate or (substrate_velocity is not None and substrate_density is not None):
#             self.is_substrate = True
#         else:
#             self.is_substrate = False
#         self.hard_substrate = hard_substrate
#         self.density_sub = substrate_density
#         self.cp_sub = substrate_velocity
#         self.cs_sub = substrate_velocity_shear

#     def wavenumber(self, frequency: float):
#         return 2 * np.pi * frequency / self.c_longitudinal

#     def k_substrate(self, k_medium):
#         omega = k_medium * self.c_longitudinal
#         if not self.is_substrate:
#             return None
#         elif self.hard_substrate:
#             return None
#         elif self.cs_sub is None:
#             return np.array([omega / self.cp_sub])
#         else:
#             return np.array([omega / self.cp_sub, omega / self.cs_sub])


class FluidMedium(Medium):
    """Fluid medium where only longitudinal waves are allowed.
    """
    def __init__(self, density: float, sound_speed_longitudinal: float):
        """Fluid medium constructor.

        :param density: density of the medium
        :param sound_speed_longitudinal: speed of sound (longitudinal) in medium
        """
        self.density = density
        self.c_longitudinal = sound_speed_longitudinal

    def wavenumber(self, frequency: float):
        """Wavenumber in fluid medium.
        """
        return 2 * np.pi * frequency / self.c_longitudinal


class ElasticMedium(Medium):
    """Elastic (solid) medium where longitudinal and transversal waves are allowed.
    """
    def __init__(self, density: float, sound_speed_longitudinal: float, sound_speed_transversal: float):
        """Elastic medium constructor

        :param density: density of the medium
        :param sound_speed_longitudinal: speed of sound (longitudinal) in medium
        :param sound_speed_transversal: speed of sound (transversal) in medium
        """
        self.density = density
        self.c_longitudinal = sound_speed_longitudinal
        self.c_transversal = sound_speed_transversal

    def wavenumber(self, frequency: float):
        """Wavenumber of longitudinal wave in elastic medium.
        """
        omega = 2 * np.pi * frequency
        return omega / self.c_longitudinal

    def wavenumber_transversal(self, frequency: float):
        """Wavenumber of transversal wave in elastic medium.
        """
        omega = 2 * np.pi * frequency
        return omega / self.c_transversal


class RigidBoundary(Medium):
    """Represents sound rigid (hard) boundary.
    """

    def wavenumber(self, frequency: float):
        """No waves in this kind of medium"""
        return None


class MediumSystem:
    """Class containing one or two mediums.

    First medium surrounds the particles, second medium (if present) is substrate.
    """
    def __init__(self, mediums: Sequence[Medium]):
        if len(mediums) < 1 or len(mediums) > 2:
            raise ValueError("Only one or two mediums are acceptable")
        self.mediums = mediums
        if len(mediums) == 2:
            self.z_interface = 0.
        else:
            self.z_interface = -np.inf

    @property
    def is_substrate(self) -> bool:
        """Check if substrate is present in medium system"""
        return True if len(self.mediums) == 2 else False

    @property
    def sur_medium(self) -> FluidMedium:
        """Access to first (upper) medium which surrounds the particles."""
        return self.mediums[0]

    @property
    def substrate(self) -> FluidMedium | RigidBoundary | ElasticMedium | None:
        """Access to second (lower) medium."""
        return self.mediums[1] if self.is_substrate else None

    def fresnel_r(self, k_parallel: float | np.ndarray, frequency: float):
        """Fresnel reflection coefficient for two mediums.

        Allowed interfaces are: fluid/fluid, fluid/elastic(solid), hard boundary.

        :param k_parallel: in-plane wavenumber
        :param frequency: frequency
        :return: Fresnel coefficient
        """
        medium1, medium2 = self.sur_medium, self.substrate
        if isinstance(medium1, FluidMedium) and isinstance(medium2, FluidMedium):
            return fresnel_r(
                k_parallel=k_parallel, k_medium=medium1.wavenumber(frequency), c_medium=medium1.c_longitudinal,
                c_substrate=medium2.c_longitudinal, rho_medium=medium1.density, rho_substrate=medium2.density
            )
        elif isinstance(medium1, FluidMedium) and isinstance(medium2, ElasticMedium):
            return fresnel_r_elastic(k_parallel=k_parallel, k_medium=medium1.wavenumber(frequency),
                                     c_medium=medium1.c_longitudinal, cl_substrate=medium2.c_longitudinal,
                                     ct_substrate=medium2.c_transversal, rho_medium=medium1.density,
                                     rho_substrate=medium2.density)
        elif isinstance(medium1, FluidMedium) and isinstance(medium2, RigidBoundary):
            return fresnel_r_hard()
        else:
            raise TypeError(f"Unknown type of interface between {type(medium1)} and {type(medium2)}.")


def fresnel_r_hard():
    """Reflectance from sound rigid boundary.
    """
    return 1


def fresnel_r(k_parallel, k_medium, c_medium, c_substrate, rho_medium, rho_substrate):
    """Reflectance from interface between two fluid mediums.

    :param k_parallel: in-plane wavenumber
    :param k_medium: wavenumber in medium of incidence
    :param c_medium: speed of sound (longitudinal) in medium of incidence
    :param c_substrate: speed of sound (longitudinal) in transmitted medium  # todo: naming
    :param rho_medium: density of medium of incidence
    :param rho_substrate: density of transmitted medium
    :return: Fresnel reflection coefficient
    """
    k_substrate = k_medium * c_medium / c_substrate
    kz_medium = np.emath.sqrt(k_medium ** 2 - k_parallel ** 2)
    kz_substrate = np.emath.sqrt(k_substrate ** 2 - k_parallel ** 2)
    return ((rho_substrate * kz_medium - rho_medium * kz_substrate) /
            (rho_substrate * kz_medium + rho_medium * kz_substrate))


def fresnel_r_elastic(k_parallel, k_medium, c_medium, cl_substrate, ct_substrate, rho_medium, rho_substrate):
    """Reflectance from interface between fluid and elastic (solid) mediums.

    :param k_parallel: in-plane wavenumber
    :param k_medium: wavenumber in medium of incidence
    :param c_medium: speed of sound (longitudinal) in medium of incidence
    :param cl_substrate: speed of sound (longitudinal) in transmitted medium
    :param ct_substrate: speed of sound (transversal) in transmitted medium
    :param rho_medium: density of medium of incidence
    :param rho_substrate: density of transmitted medium
    :return: Fresnel reflection coefficient
    """
    # todo: rewrite it is not correct
    # omega = k_medium * c_medium
    # k_substrate_p = omega / cl_substrate
    # k_substrate_s = omega / ct_substrate

    # ai = np.emath.arcsin(k_parallel / k_medium)
    # al = np.emath.arcsin(k_parallel / k_substrate_p)
    # at = np.emath.arcsin(k_parallel / k_substrate_s)
    # z = rho_medium * c_medium
    # # zt = rho_substrate * c_substrate_s
    # zl = rho_substrate * cl_substrate
    # v = ct_substrate / cl_substrate

    # num = v**2 * np.sin(2 * at) * np.sin(2 * al) + np.cos(2 * at)**2 - z * np.cos(al) / zl / np.cos(ai)
    # den = v**2 * np.sin(2 * at) * np.sin(2 * al) + np.cos(2 * at)**2 + z * np.cos(al) / zl / np.cos(ai)
    # return num / den

    omega = k_medium * c_medium
    k_sub_d = omega / cl_substrate
    k_sub_s = omega / ct_substrate

    kz_med = np.emath.sqrt(k_medium**2 - k_parallel**2)
    gz_d = np.emath.sqrt(k_parallel**2 - k_sub_d**2)
    gz_s = np.emath.sqrt(k_parallel**2 - k_sub_s**2)

    term1 = (2 * k_parallel**2 - k_sub_s**2)**2
    term2 = 4 * k_parallel**2 * gz_s * gz_d
    term3 = 1j * rho_medium / rho_substrate * k_sub_s**4 * gz_d / kz_med
    return (term1 - term2 - term3) / (term1 - term2 + term3)