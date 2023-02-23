import acsmuthi.linear_system.t_matrix as tmt
import acsmuthi.utility.memoizing as memo


class Particle:
    def __init__(self, position, radius, density, speed_of_sound, order, speed_t=None):
        self.pos = position
        self.r = radius
        self.rho = density
        self.speed_l = speed_of_sound
        self.speed_t = speed_t
        self.incident_field = None
        self.scattered_field = None
        self.inner_field = None
        self.t_matrix = None
        self.order = order

    def compute_t_matrix(self, c_medium, rho_medium, freq):
        t = _compute_t_matrix(self.order, c_medium, rho_medium, self.speed_l, self.rho, self.r, freq, self.speed_t)
        self.t_matrix = t
        return t


@memo.Memoize
def _compute_t_matrix(order, c_medium, rho_medium, c_sphere_l, rho_sphere, r_sphere, freq, c_sphere_t=None):
    return tmt.t_matrix_sphere(order, c_medium, rho_medium, c_sphere_l, rho_sphere, r_sphere, freq, c_sphere_t)


