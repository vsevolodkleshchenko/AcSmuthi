import numpy as np

import acsmuthi.linear_system.t_matrix as tmt
import acsmuthi.utility.memoizing as memo


class Particle:
    def __init__(self,
                 position: np.ndarray[float],
                 density: float,
                 speed_l: float,
                 order: int,
                 speed_t: float = None):
        self.position = position
        self.density = density
        self.speed_l = speed_l
        self.speed_t = speed_t
        self.incident_field = None
        self.scattered_field = None
        self.inner_field = None
        self.t_matrix = None
        self.order = order

    def compute_t_matrix(self, c_medium, rho_medium, freq):
        pass


class SphericalParticle(Particle):
    def __init__(
            self,
            position: np.ndarray[float],
            radius: float,
            density: float,
            speed_l: float,
            order: int,
            speed_t: float = None
    ):
        super(SphericalParticle, self).__init__(position, density, speed_l, order, speed_t)
        self.radius = radius

    def compute_t_matrix(self, c_medium, rho_medium, freq):
        t = _compute_sphere_t_matrix(self.order, c_medium, rho_medium, self.speed_l, self.density, self.radius, freq,
                                     self.speed_t)
        self.t_matrix = t
        return t


# @memo.Memoize
def _compute_sphere_t_matrix(order, c_medium, rho_medium, c_sphere_l, rho_sphere, r_sphere, freq, c_sphere_t=None):
    return tmt.t_matrix_sphere(order, c_medium, rho_medium, c_sphere_l, rho_sphere, r_sphere, freq, c_sphere_t)


