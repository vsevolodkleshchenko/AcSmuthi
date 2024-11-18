from abc import ABC, abstractmethod
import numpy as np

import acsmuthi.linear_system.t_matrix as tmt


class Particle(ABC):
    """Abstract class for scattering particle."""

    def __init__(self, position: np.ndarray[float], multipole_order: int):  # todo: change order ?
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

    @abstractmethod
    def compute_t_matrix(self, c_medium: float, rho_medium: float, frequency: float) -> np.ndarray:  # todo: medium as argument ?
        """T-matrix of a particle.

        :param c_medium: speed of sound (longitudinal) in surrounding medium
        :param rho_medium: density of surrounding medium
        :param frequency: frequency
        :return: T-matrix
        """
        pass


class SphericalParticle(Particle):
    """Class for spherical homogeneous particle (fluid)."""

    def __init__(
            self,
            position: np.ndarray[float],
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
        super(SphericalParticle, self).__init__(position, multipole_order)
        self.position = position
        self.density = density
        self.c_longitudinal = sound_speed_longitudinal
        self.radius = radius

    def compute_t_matrix(self, c_medium, rho_medium, frequency):
        """T-matrix of a spherical particle."""
        t = _compute_sphere_t_matrix(self.n_max, c_medium, rho_medium, self.c_longitudinal, self.density, self.radius,
                                     frequency)
        self.t_matrix = t
        return t


# todo: @memo.Memoize
def _compute_sphere_t_matrix(n_max, c_medium, rho_medium, c_particle, rho_particle, radius, freq):
    """Private t-matrix method function"""
    return tmt.t_matrix_sphere(n_max, c_medium, rho_medium, c_particle, rho_particle, radius, freq)


