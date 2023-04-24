import numpy as np

import acsmuthi.linear_system.t_matrix as tmt
import acsmuthi.utility.memoizing as memo


class Particle:
    def __init__(self,
                 position: np.ndarray[float],
                 density: float,
                 pressure_velocity: float,
                 order: int,
                 shear_velocity: float = None):
        self.position = position
        self.density = density
        self.cp = pressure_velocity
        self.cs = shear_velocity
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
            pressure_velocity: float,
            order: int,
            shear_velocity: float = None
    ):
        super(SphericalParticle, self).__init__(position, density, pressure_velocity, order, shear_velocity)
        self.radius = radius

    def compute_t_matrix(self, c_medium, rho_medium, freq):
        t = _compute_sphere_t_matrix(self.order, c_medium, rho_medium, self.cp, self.density, self.radius, freq,
                                     self.cs)
        self.t_matrix = t
        return t


# @memo.Memoize
def _compute_sphere_t_matrix(order, c_medium, rho_medium, c_sphere_l, rho_sphere, r_sphere, freq, c_sphere_t=None):
    return tmt.t_matrix_sphere(order, c_medium, rho_medium, c_sphere_l, rho_sphere, r_sphere, freq, c_sphere_t)


