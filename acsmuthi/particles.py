from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from acsmuthi.medium import Medium, FluidMedium

import acsmuthi.linear_system.t_matrix as tmt


class Particle(ABC):
    """Abstract class for scattering particle."""

    def __init__(self, position: npt.NDArray[float], multipole_order: int):  # todo: change order ?
        """Particle constructor.

        :param position: cartesian coordinates of the particle position
        :param multipole_order: maximum multipole order of spherical expansion (non-negative)
        """
        self.position = position
        self.incident_field = None
        self.scattered_field = None
        self.inner_field = None
        self.t_matrix = None
        self.n_max = multipole_order

    @property
    @abstractmethod
    def circumscribing_sphere_radius(self) -> float:
        """Radius of the sphere that circumscribes the particle.
        """
        pass

    @abstractmethod
    def compute_t_matrix(self, medium: Medium, frequency: float) -> np.ndarray:
        """T-matrix of a particle.

        :param medium: medium surrounding particle
        :param frequency: frequency
        :return: T-matrix
        """
        pass


class SphericalParticle(Particle):
    """Class for spherical homogeneous particle (fluid)."""

    def __init__(
            self,
            position: npt.NDArray[float],
            multipole_order: int,
            radius: float,
            density: float,
            sound_speed_longitudinal: float,
    ):
        """Spherical particle constructor.

        :param position: cartesian coordinates of the particle center
        :param multipole_order: maximum multipole order of spherical expansion (non-negative)
        :param radius: radius of the particle
        :param density: density of the particle
        :param sound_speed_longitudinal: speed of sound (longitudinal) in the particle
        """
        super(SphericalParticle, self).__init__(position=position, multipole_order=multipole_order)
        self.position = position
        self.density = density
        self.c_longitudinal = sound_speed_longitudinal
        self.radius = radius

    @property
    def circumscribing_sphere_radius(self) -> float:
        """Radius of the sphere that circumscribes the particle.
        """
        return self.radius

    def compute_t_matrix(self, medium: Medium, frequency):
        """T-matrix of a spherical particle.
        """
        if not isinstance(medium, FluidMedium):
            raise TypeError("Only fluid medium is supported")
        t = _compute_sphere_t_matrix(
            n_max=self.n_max, c_medium=medium.c_longitudinal, rho_medium=medium.density,
            c_particle=self.c_longitudinal, rho_particle=self.density, radius=self.radius, freq=frequency
        )
        self.t_matrix = t
        return t


# todo: @memo.Memoize
def _compute_sphere_t_matrix(n_max, c_medium, rho_medium, c_particle, rho_particle, radius, freq):
    """Private t-matrix method function.
    """
    return tmt.t_matrix_sphere(n_max, c_medium, rho_medium, c_particle, rho_particle, radius, freq)
