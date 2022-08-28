import wavefunctions as wvfs
import numpy as np
import particles
import oop_fields as flds
import scipy.special as ss
import mathematics as mths
import layers


class LinearSystem:
    def __init__(self, particles_array, layer, fluid, incident_field, frequency, order):
        self.order = order
        self.rhs = None
        self.t_matrix = None
        self.d_matrix = None
        self.r_matrix = None
        self.particles = particles_array
        self.layer = layer
        self.incident_field = incident_field
        self.fluid = fluid
        self.freq = frequency

    def compute_t_matrix(self):
        block_matrix = np.zeros((len(self.particles), len(self.particles), (self.order+1)**2, (self.order+1)**2), dtype=complex)
        all_spheres = np.arange(len(self.particles))
        for sph in all_spheres:
            self.particles[sph].compute_t_matrix(self.fluid)
            block_matrix[sph, sph] = self.particles[sph].t_matrix
            other_spheres = np.where(all_spheres != sph)[0]
            for osph in other_spheres:
                block_matrix[sph, osph] = particles.coupling_block(self.particles[sph], self.particles[osph],
                                                                   self.incident_field.k, self.order)
        matrix2d = np.concatenate(np.concatenate(block_matrix, axis=1), axis=1)
        self.t_matrix = np.linalg.inv(matrix2d)

    def compute_d_matrix(self):
        block_matrix = np.zeros((len(self.particles), (self.order + 1) ** 2, (self.order + 1) ** 2), dtype=complex)
        for s, particle in enumerate(self.particles):
            particle.compute_d_matrix()
            block_matrix[s] = particle.d_matrix
        matrix2d = block_matrix.reshape((len(self.particles) * (self.order + 1) ** 2, (self.order + 1) ** 2))
        self.d_matrix = matrix2d

    def compute_right_hand_side(self):
        self.rhs = np.dot(self.d_matrix, self.incident_field.coefficients)

    def compute_r_matrix(self):
        self.r_matrix = layers.r_matrix(self.particles, self.layer, self.fluid, self.freq, self.order)

    def prepare(self):
        for particle in self.particles:
            particle.incident_field = flds.SphericalWaveExpansion(self.incident_field.ampl, self.incident_field.k,
                                                                  'regular', self.order)
            particle.scattered_field = flds.SphericalWaveExpansion(self.incident_field.ampl, self.incident_field.k,
                                                                   'outgoing', self.order)
            k_particle = 2 * np.pi * self.freq / particle.speed
            particle.inner_field = flds.SphericalWaveExpansion(self.incident_field.ampl, k_particle,
                                                               'regular', self.order)
        self.compute_t_matrix()
        self.compute_d_matrix()
        self.compute_right_hand_side()

    def prepare_layer(self):
        for particle in self.particles:
            particle.reflected_field = flds.SphericalWaveExpansion(self.incident_field.ampl, self.incident_field.k,
                                                                   'regular', self.order)
        self.layer.reflected_field = flds.SphericalWaveExpansion(self.incident_field.ampl, self.incident_field.k,
                                                                 'regular', self.order)
        self.compute_r_matrix()

    def solve(self):
        self.prepare()
        if self.layer:
            self.prepare_layer()
            incident_coefs_origin = layers.layer_inc_coef_origin(self.incident_field, self.layer, self.order)
            incident_coefs_array = np.dot(self.d_matrix, incident_coefs_origin)

            m1 = self.t_matrix @ self.d_matrix
            m2 = self.r_matrix @ m1
            m3 = np.linalg.inv(np.eye(m2.shape[0]) - m2)

            scattered_coefs1d = np.dot(m1 @ m3, incident_coefs_origin)
            scattered_coefs = scattered_coefs1d.reshape((len(self.particles), (self.order + 1) ** 2))

            reflected_coefs = np.dot(m3, incident_coefs_origin) - incident_coefs_origin
            local_reflected_coefs_array = np.dot(self.d_matrix, reflected_coefs)

            local_reflected_coefs = local_reflected_coefs_array.reshape((len(self.particles), (self.order + 1) ** 2))
            for s, particle in enumerate(self.particles):
                particle.reflected_field.coefficients = local_reflected_coefs[s]
            self.layer.reflected_field.coefficients = reflected_coefs
        else:
            scattered_coefs1d = np.dot(self.t_matrix, self.rhs)
            scattered_coefs = scattered_coefs1d.reshape((len(self.particles), (self.order + 1) ** 2))
            incident_coefs_array = self.rhs

        incident_coefs = incident_coefs_array.reshape((len(self.particles), (self.order + 1) ** 2))
        for s, particle in enumerate(self.particles):
            particle.incident_field.coefficients = incident_coefs[s]
            particle.scattered_field.coefficients = scattered_coefs[s]
        inner_coefs = inner_coefficients(self.particles, self.fluid, self.order)
        for s, particle in enumerate(self.particles):
            particle.inner_field.coefficients = inner_coefs[s]


def inner_coefficients(particles_array, fluid, order):
    r"""Counts coefficients of decompositions fields inside spheres"""
    in_coef = np.zeros((len(particles_array), (order + 1) ** 2), dtype=complex)
    for s, particle in enumerate(particles_array):
        for m, n in wvfs.multipoles(order):
            imn = n ** 2 + n + m
            k, k_s = particle.incident_field.k, particle.inner_field.k
            sc_coef = particle.scattered_field.coefficients
            in_coef[s, imn] = (ss.spherical_jn(n, k * particle.r) / particles.scaled_coefficient(n, particle, fluid) +
                               mths.sph_hankel1(n, k * particle.r)) * sc_coef[imn]/ss.spherical_jn(n, k_s * particle.r)
    return in_coef
