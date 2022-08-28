import tsystem
import wavefunctions as wvfs
import numpy as np
import particles
import oop_fields as flds
import layers


class LinearSystem:
    def __init__(self, physical_system, order):
        self.ps = physical_system
        self.order = order
        self.rhs = None
        self.t_matrix = None
        self.d_matrix = None
        self.r_matrix = None

    def compute_t_matrix(self):
        block_matrix = np.zeros((self.ps.num_sph, self.ps.num_sph, (self.order+1)**2, (self.order+1)**2), dtype=complex)
        all_spheres = np.arange(self.ps.num_sph)
        for sph in all_spheres:
            self.ps.spheres[sph].compute_t_matrix(sph, self.ps)
            block_matrix[sph, sph] = self.ps.spheres[sph].t_matrix
            other_spheres = np.where(all_spheres != sph)[0]
            for osph in other_spheres:
                block_matrix[sph, osph] = particles.coupling_block(self.ps.spheres[sph], self.ps.spheres[osph],
                                                                   self.ps.k_fluid, self.order)
        matrix2d = np.concatenate(np.concatenate(block_matrix, axis=1), axis=1)
        self.t_matrix = np.linalg.inv(matrix2d)

    def compute_d_matrix(self):
        block_matrix = np.zeros((self.ps.num_sph, (self.order + 1) ** 2, (self.order + 1) ** 2), dtype=complex)
        for s, particle in enumerate(self.ps.spheres):
            particle.compute_d_matrix(self.ps)
            block_matrix[s] = particle.d_matrix
        matrix2d = block_matrix.reshape((self.ps.num_sph * (self.order + 1) ** 2, (self.order + 1) ** 2))
        self.d_matrix = matrix2d

    def compute_right_hand_side(self):
        self.rhs = np.dot(self.d_matrix, wvfs.incident_coefficients(self.ps.incident_field.dir, self.order))

    def compute_r_matrix(self):
        self.r_matrix = layers.r_matrix(self.ps, self.order)

    def prepare(self):
        self.compute_t_matrix()
        self.compute_d_matrix()
        self.compute_right_hand_side()

    def prepare_layer(self):
        self.compute_r_matrix()

    def solve(self):
        self.prepare()
        if self.ps.interface:
            self.prepare_layer()
            incident_coefs_origin = tsystem.layer_inc_coef_origin(self.ps, self.order)
            incident_coefs_array = np.dot(self.d_matrix, incident_coefs_origin)

            m1 = self.t_matrix @ self.d_matrix
            m2 = self.r_matrix @ m1
            m3 = np.linalg.inv(np.eye(m2.shape[0]) - m2)

            scattered_coefs1d = np.dot(m1 @ m3, incident_coefs_origin)
            scattered_coefs = scattered_coefs1d.reshape((self.ps.num_sph, (self.order + 1) ** 2))

            reflected_coefs = np.dot(m3, incident_coefs_origin) - incident_coefs_origin
            local_reflected_coefs_array = np.dot(self.d_matrix, reflected_coefs)
            inner_coefs = tsystem.inner_coefficients(scattered_coefs, self.ps, self.order)

            local_reflected_coefs = local_reflected_coefs_array.reshape((self.ps.num_sph, (self.order + 1) ** 2))
            for s in range(self.ps.num_sph):
                self.ps.spheres[s].reflected_field = flds.SphericalWaveExpansion(self.ps.incident_field.ampl, self.ps.k_fluid,
                                                                                 'regular', self.order, local_reflected_coefs[s])
            self.ps.interface.reflected_field = flds.SphericalWaveExpansion(self.ps.incident_field.ampl, self.ps.k_fluid,
                                                                            'regular', self.order, reflected_coefs)
        else:
            scattered_coefs1d = np.dot(self.t_matrix, self.rhs)
            scattered_coefs = scattered_coefs1d.reshape((self.ps.num_sph, (self.order + 1) ** 2))
            inner_coefs = tsystem.inner_coefficients(scattered_coefs, self.ps, self.order)
            incident_coefs_array = self.rhs

        incident_coefs = incident_coefs_array.reshape((self.ps.num_sph, (self.order + 1) ** 2))
        for s in range(self.ps.num_sph):
            self.ps.spheres[s].incident_field = flds.SphericalWaveExpansion(self.ps.incident_field.ampl, self.ps.k_fluid,
                                                                            'regular', self.order, incident_coefs[s])
            self.ps.spheres[s].scattered_field = flds.SphericalWaveExpansion(self.ps.incident_field.ampl, self.ps.k_fluid,
                                                                             'outgoing', self.order, scattered_coefs[s])
            self.ps.spheres[s].inner_field = flds.SphericalWaveExpansion(self.ps.incident_field.ampl, self.ps.k_spheres[s],
                                                                         'regular', self.order, inner_coefs[s])
