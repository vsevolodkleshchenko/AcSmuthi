import numpy as np
import reflection
import wavefunctions as wvfs


class Layer:
    def __init__(self, density, speed_of_sound, a, b, c, d):
        self.rho = density
        self.speed = speed_of_sound
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.reflected_field = None

    def int_dist(self, position):
        r"""Absolute distance between position and interface"""
        return np.abs(self.a * position[0] + self.b * position[1] + self.c * position[2] + self.d) / \
               np.sqrt(self.a ** 2 + self.b ** 2 + self.c ** 2)

    @property
    def int_dist0(self):
        r"""Absolute distance between coordinate's origin and interface"""
        return self.int_dist(np.array([0., 0., 0.]))

    @property
    def normal(self):
        r"""Normal unit vector to interface with direction to coordinate's origin"""
        n = np.array(np.array([self.a, self.b, self.c]) / np.sqrt(self.a ** 2 + self.b ** 2 + self.c ** 2))
        n_dist = n * self.int_dist0
        if n_dist[0] * self.a + n_dist[1] * self.b + n_dist[2] * self.c == -self.d:
            n *= -1
        return n


def r_matrix(ps, order, order_approx=1):
    r"""Build R matrix - reflection matrix"""
    block_matrix = np.zeros((ps.num_sph, (order + 1) ** 2, (order + 1) ** 2), dtype=complex)
    a, alpha = reflection.ref_coef_approx(ps.omega, ps.fluid.speed, ps.interface.speed, ps.fluid.rho,
                                          ps.interface.rho, order_approx, -3)
    for m, n in wvfs.multipoles(order):
        imn = n ** 2 + n + m
        for mu, nu, in wvfs.multipoles(order):
            imunu = nu ** 2 + nu + mu
            for sph in range(ps.num_sph):
                image_poses = reflection.image_poses(ps.spheres[sph], ps.interface, alpha)
                image_contribution = reflection.image_contribution(m, n, mu, nu, ps.k_fluid, image_poses, a)
                block_matrix[sph, imn, imunu] = (-1) ** (nu + mu) * image_contribution
    matrix2d = np.concatenate(block_matrix, axis=1)
    return matrix2d
