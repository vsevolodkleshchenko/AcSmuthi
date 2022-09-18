import numpy as np
from utility import reflection, wavefunctions as wvfs, mathematics as mths
import scipy


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


def r_matrix(particles, layer, medium, freq, order, order_approx=6):
    r_block_matrix = np.zeros((len(particles), (order + 1) ** 2, (order + 1) ** 2), dtype=complex)
    w = 2 * np.pi * freq
    a, alpha = reflection.ref_coef_approx(w, medium.speed_l, layer.speed_l, medium.rho, layer.rho, order_approx, 0.3)
    for s, particle in enumerate(particles):
        r_block_matrix[s] = compute_r_block(particle, layer, a, alpha, order)
    r = np.concatenate(r_block_matrix, axis=1)
    print('r:', scipy.linalg.norm(r, 2), sep='\n')
    return r


def compute_r_block(particle, layer, a, alpha, order):
    r_block = np.zeros(((order + 1) ** 2, (order + 1) ** 2), dtype=complex)
    for m, n in wvfs.multipoles(order):
        imn = n ** 2 + n + m
        for mu, nu in wvfs.multipoles(order):
            imunu = nu ** 2 + nu + mu
            r_block[imn, imunu] = image_contrib(particle, layer, a, alpha, mu, m, nu, n)
    return r_block


def image_contrib(particle, layer, a, alpha, mu, m, nu, n):
    k = particle.incident_field.k_l
    contribution = np.zeros(len(a), dtype=complex)
    for q in range(len(a)):
        dist = particle.pos - (2 * layer.int_dist(particle.pos) - 1j * alpha[q]) * layer.normal
        contribution[q] = a[q] * wvfs.outgoing_separation_coefficient(mu, m, nu, n, k, -dist)
    return (-1) ** (mu + nu) * mths.complex_fsum(contribution)


def new_r_matrix(particles, layer, medium, freq, order, order_approx=6):
    r"""Build R matrix - reflection matrix"""
    r1_block_matrix = np.zeros((len(particles), (order + 1) ** 2, (order + 1) ** 2), dtype=complex)
    r2_block_matrix = np.zeros((len(particles), (order + 1) ** 2, (order + 1) ** 2), dtype=complex)
    r3_block_matrix = np.zeros((len(particles), len(particles), (order + 1) ** 2, (order + 1) ** 2), dtype=complex)

    w = 2 * np.pi * freq
    a, alpha = reflection.ref_coef_approx(w, medium.speed_l, layer.speed_l, medium.rho, layer.rho, order_approx, 0.3)

    r3_block = compute_r3_block(medium.incident_field.k_l, layer.normal, a, alpha, order)
    for s, particle in enumerate(particles):
        r1_block_matrix[s] = compute_r1_block(particle, layer, a[0], order)
        r2_block_matrix[s] = compute_r2_block(particle, layer, order)
        r3_block_matrix[s, s] = r3_block

    r1 = np.concatenate(r1_block_matrix, axis=1)
    r2 = np.concatenate(r2_block_matrix, axis=1)
    r3 = np.concatenate(np.concatenate(r3_block_matrix, axis=1), axis=1)
    r23 = r2 @ r3
    print('r1:', scipy.linalg.norm(r1, 2), sep='\n')
    print('r2:', scipy.linalg.norm(r2, 2), sep='\n')
    print('r3:', scipy.linalg.norm(r3, 2), sep='\n')
    print('r23:', scipy.linalg.norm(r23, 2), sep='\n')
    r = r1 + r23
    return r


def compute_r1_block(particle, layer, a0, order):
    r1_block = np.zeros(((order + 1) ** 2, (order + 1) ** 2), dtype=complex)
    k = particle.incident_field.k_l
    image_pos = particle.pos - 2 * layer.int_dist(particle.pos) * layer.normal
    for m, n in wvfs.multipoles(order):
        imn = n ** 2 + n + m
        for mu, nu in wvfs.multipoles(order):
            imunu = nu ** 2 + nu + mu
            r1_block[imn, imunu] = (-1)**(mu + nu) * wvfs.outgoing_separation_coefficient(mu, m, nu, n, k, -image_pos)
    return a0 * r1_block


def compute_r2_block(particle, layer, order):
    r2_block = np.zeros(((order + 1) ** 2, (order + 1) ** 2), dtype=complex)
    k = particle.incident_field.k_l
    image_pos = particle.pos - 2 * layer.int_dist(particle.pos) * layer.normal
    for m, n in wvfs.multipoles(order):
        imn = n ** 2 + n + m
        for mu, nu in wvfs.multipoles(order):
            imunu = nu ** 2 + nu + mu
            r2_block[imn, imunu] = wvfs.regular_separation_coefficient(mu, m, nu, n, k, -image_pos)
    return r2_block


def compute_r3_block(k, normal, a, alpha, order):
    r3_block = np.zeros(((order + 1) ** 2, (order + 1) ** 2), dtype=complex)
    a, alpha = a[1:], alpha[1:]
    for mu, nu in wvfs.multipoles(order):
        imunu = nu ** 2 + nu + mu
        for f, l in wvfs.multipoles(order):
            ifl = l ** 2 + l + f
            r3_block[imunu, ifl] = new_image_contrib(f, mu, l, nu, a, alpha, k, normal)
    return r3_block


def new_image_contrib(f, mu, l, nu, a, alpha, k, normal):
    contribution = np.zeros(len(a), dtype=complex)
    for q in range(len(a)):
        dist = -1j * alpha[q] * normal
        contribution[q] = a[q] * wvfs.outgoing_separation_coefficient(f, mu, l, nu, k, dist)
    return (-1) ** (f + l) * mths.complex_fsum(contribution)
