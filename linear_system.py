import numpy as np
import scipy.linalg

import particles
import fields_expansions as fldsex
import scipy.special as ss
from utility import mathematics as mths, wavefunctions as wvfs
import layers


class LinearSystem:
    def __init__(self, particles_array, layer, medium, frequency, order):
        self.order = order
        self.rhs = None
        self.t_matrix = None
        self.d_matrix = None
        self.r_matrix = None
        self.particles = particles_array
        self.layer = layer
        self.medium = medium
        self.freq = frequency

    def compute_t_matrix(self):
        block_matrix = np.zeros((len(self.particles), len(self.particles), (self.order+1)**2, (self.order+1)**2), dtype=complex)
        all_spheres = np.arange(len(self.particles))
        for sph in all_spheres:
            self.particles[sph].compute_t_matrix(self.medium.speed_l, self.medium.rho, self.freq)
            block_matrix[sph, sph] = self.particles[sph].t_matrix
            other_spheres = np.where(all_spheres != sph)[0]
            for osph in other_spheres:
                block_matrix[sph, osph] = particles.coupling_block(self.particles[sph].pos, self.particles[osph].pos,
                                                                   self.medium.incident_field.k_l, self.order)
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
        self.rhs = np.dot(self.d_matrix, self.medium.incident_field.coefficients)

    def compute_r_matrix(self):
        self.r_matrix = layers.r_matrix(self.particles, self.layer, self.medium, self.freq, self.order)

    def prepare(self):
        for particle in self.particles:
            ampl, k_l = self.medium.incident_field.ampl, self.medium.incident_field.k_l
            particle.incident_field = fldsex.SphericalWaveExpansion(ampl, k_l, particle.pos, 'regular', self.order)
            particle.scattered_field = fldsex.SphericalWaveExpansion(ampl, k_l, particle.pos, 'outgoing', self.order)
            if particle.speed_t:
                kst = 2 * np.pi * self.freq / particle.speed_t
            else:
                kst = None
            ksl = 2 * np.pi * self.freq / particle.speed_l
            particle.inner_field = fldsex.SphericalWaveExpansion(ampl, ksl, particle.pos, 'regular', self.order, k_t=kst)
        self.compute_t_matrix()
        self.compute_d_matrix()
        self.compute_right_hand_side()

    def prepare_layer(self):
        ampl, k_l = self.medium.incident_field.ampl, self.medium.incident_field.k_l
        for particle in self.particles:
            particle.reflected_field = fldsex.SphericalWaveExpansion(ampl, k_l, particle.pos, 'regular', self.order)
        self.medium.reflected_field = fldsex.SphericalWaveExpansion(ampl, k_l, np.array([0, 0, 0]), 'regular', self.order)
        self.compute_r_matrix()

    def solve(self):
        self.prepare()
        if self.layer:
            self.prepare_layer()
            incident_coefs_origin = layers.layer_inc_coef_origin(self.medium, self.layer, self.freq, self.order)
            incident_coefs_array = np.dot(self.d_matrix, incident_coefs_origin)

            print('t:', scipy.linalg.norm(self.t_matrix, 2), sep='\n')
            print('d:', scipy.linalg.norm(self.d_matrix, 2), sep='\n')
            print('r:', scipy.linalg.norm(self.r_matrix, 2), sep='\n')

            m1 = self.t_matrix @ self.d_matrix
            print('td:', scipy.linalg.norm(m1, 2), sep='\n')
            m2 = self.r_matrix @ m1
            print('rtd:', scipy.linalg.norm(m2, 2), sep='\n')

            m3 = np.linalg.inv(np.eye(m2.shape[0]) - m2)

            scattered_coefs1d = np.dot(m1 @ m3, incident_coefs_origin)
            scattered_coefs = scattered_coefs1d.reshape((len(self.particles), (self.order + 1) ** 2))

            reflected_coefs = np.dot(m3, incident_coefs_origin) - incident_coefs_origin
            local_reflected_coefs_array = np.dot(self.d_matrix, reflected_coefs)

            local_reflected_coefs = local_reflected_coefs_array.reshape((len(self.particles), (self.order + 1) ** 2))
            for s, particle in enumerate(self.particles):
                particle.reflected_field.coefficients = local_reflected_coefs[s]
            self.medium.reflected_field.coefficients = reflected_coefs
        else:
            scattered_coefs1d = np.dot(self.t_matrix, self.rhs)
            scattered_coefs = scattered_coefs1d.reshape((len(self.particles), (self.order + 1) ** 2))
            incident_coefs_array = self.rhs

        incident_coefs = incident_coefs_array.reshape((len(self.particles), (self.order + 1) ** 2))
        for s, particle in enumerate(self.particles):
            particle.incident_field.coefficients = incident_coefs[s]
            particle.scattered_field.coefficients = scattered_coefs[s]
        inner_coefs = inner_coefficients(self.particles, self.order)
        for s, particle in enumerate(self.particles):
            particle.inner_field.coefficients = inner_coefs[s]


def inner_coefficients(particles_array, order):
    r"""Counts coefficients of decompositions fields inside spheres"""
    in_coef = np.zeros((len(particles_array), (order + 1) ** 2), dtype=complex)
    for s, particle in enumerate(particles_array):
        for m, n in wvfs.multipoles(order):
            imn = n ** 2 + n + m
            k, k_s = particle.incident_field.k_l, particle.inner_field.k_l
            sc_coef = particle.scattered_field.coefficients
            in_coef[s, imn] = (ss.spherical_jn(n, k * particle.r) * particle.t_matrix[imn, imn] +
                               mths.sph_hankel1(n, k * particle.r)) * sc_coef[imn]/ss.spherical_jn(n, k_s * particle.r)
    return in_coef
