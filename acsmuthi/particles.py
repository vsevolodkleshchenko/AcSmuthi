from abc import ABC, abstractmethod
import numpy as np

import acsmuthi.linear_system.t_matrix as tmt


class Particle(ABC):
    def __init__(
            self,
            position: np.ndarray[float],
            multipole_order: int,  # todo: change orders
    ):
        self.position = position
        self.incident_field = None
        self.scattered_field = None
        self.inner_field = None
        self.t_matrix = None
        self.order = multipole_order  # todo: rename to n_max

    @abstractmethod
    def compute_t_matrix(self, c_medium, rho_medium, freq):     # todo: medium as argument
        pass


class SphericalParticle(Particle):  # todo: decide order / l_max
    def __init__(
            self,
            position: np.ndarray[float],
            radius: float,
            density: float,
            sound_speed_longitudinal: float,
            multipole_order: int,
    ):
        super(SphericalParticle, self).__init__(position, multipole_order)
        self.position = position
        self.density = density
        self.c_longitudinal = sound_speed_longitudinal
        self.radius = radius

    def compute_t_matrix(self, c_medium, rho_medium, freq):
        t = _compute_sphere_t_matrix(self.order, c_medium, rho_medium, self.c_longitudinal, self.density, self.radius,
                                     freq)
        self.t_matrix = t
        return t


# todo: @memo.Memoize
def _compute_sphere_t_matrix(n_max, c_medium, rho_medium, c_particle, rho_particle, radius, freq):
    return tmt.t_matrix_sphere(n_max, c_medium, rho_medium, c_particle, rho_particle, radius, freq)


