import numpy as np
from utility import reflection, wavefunctions as wvfs


class Layer:
    def __init__(self, density, speed_of_sound, a, b, c, d):
        self.rho = density
        self.speed_l = speed_of_sound
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.transmitted_field = None

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


def r_matrix(particles, layer, medium, freq, order, order_approx=2):
    r"""Build R matrix - reflection matrix"""
    block_matrix = np.zeros((len(particles), (order + 1) ** 2, (order + 1) ** 2), dtype=complex)
    omega = 2 * np.pi * freq
    a, alpha = reflection.ref_coef_approx(omega, medium.speed_l, layer.speed_l, medium.rho, layer.rho, order_approx, 0.39)
    for m, n in wvfs.multipoles(order):
        imn = n ** 2 + n + m
        for mu, nu, in wvfs.multipoles(order):
            imunu = nu ** 2 + nu + mu
            for s, particle in enumerate(particles):
                image_poses = reflection.image_poses(particle, layer, alpha)
                k = particle.incident_field.k_l
                image_contribution = reflection.image_contribution(m, n, mu, nu, k, image_poses, a)
                block_matrix[s, imn, imunu] = (-1) ** (nu + mu) * image_contribution
    matrix2d = np.concatenate(block_matrix, axis=1)
    return matrix2d


def reflection_amplitude(medium, layer, freq):
    k_l = medium.incident_field.k_l
    h = k_l * np.sqrt(1 - np.dot(medium.incident_field.dir, layer.normal) ** 2)
    ref_coef = reflection.ref_coef_h(h, 2 * np.pi * freq, medium.speed_l, layer.speed_l, medium.rho, layer.rho)
    return ref_coef


def layer_inc_coef_origin(medium, layer, freq, order):
    r"""Effective incident coefficients - coefficients of decomposition incident field and it's reflection"""
    ref_dir = reflection.reflection_dir(medium.incident_field.dir, layer.normal)
    k_l = medium.incident_field.k_l
    phase = np.exp(1j * k_l * 2 * layer.int_dist0 * np.dot(ref_dir, layer.normal))
    ref_coefs = wvfs.incident_coefficients(ref_dir, order) * phase
    return medium.incident_field.coefficients + reflection_amplitude(medium, layer, freq) * ref_coefs
