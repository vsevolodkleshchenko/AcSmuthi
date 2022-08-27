import tsystem
import numpy as np
import oop_fields as flds


class LinearSystem:
    def __init__(self, physical_system, order):
        self.ps = physical_system
        self.order = order
        self.rhs = None
        self.t_matrix = None
        self.d_matrix = None
        self.r_matrix = None
        self.solution_coefficients = None

    def compute_t_matrix(self):
        self.t_matrix = np.linalg.inv(tsystem.system_matrix(self.ps, self.order))

    def compute_right_hand_side(self):
        self.rhs = tsystem.system_rhs(self.ps, self.order)

    def compute_d_matrix(self):
        self.d_matrix = tsystem.d_matrix(self.ps, self.order)

    def compute_r_matrix(self):
        self.r_matrix = tsystem.r_matrix(self.ps, self.order)

    def prepare(self):
        self.compute_t_matrix()
        self.compute_right_hand_side()

    def prepare_layer(self):
        self.compute_d_matrix()
        self.compute_r_matrix()

    def solve(self):
        self.prepare()
        if self.ps.interface:
            self.prepare_layer()
            inc_coef_origin = tsystem.layer_inc_coef_origin(self.ps, self.order)
            incident_coefs = np.dot(self.d_matrix, inc_coef_origin)

            m1 = self.t_matrix @ self.d_matrix
            m2 = self.r_matrix @ m1
            m3 = np.linalg.inv(np.eye(m2.shape[0]) - m2)

            scattered_coefs1d = np.dot(m1 @ m3, inc_coef_origin)
            scattered_coefs = scattered_coefs1d.reshape((self.ps.num_sph, (self.order + 1) ** 2))

            reflected_coefs = np.dot(m3, inc_coef_origin) - inc_coef_origin
            local_reflected_coefs = np.dot(self.d_matrix, reflected_coefs)
            inner_coefs = tsystem.inner_coefficients(scattered_coefs, self.ps, self.order)

            sep_loc_refl_coefs = local_reflected_coefs.reshape((self.ps.num_sph, (self.order + 1) ** 2))
            for s in range(self.ps.num_sph):
                self.ps.spheres[s].reflected_field = flds.SphericalWaveExpansion(self.ps.incident_field.ampl, self.ps.k_fluid,
                                                                                 sep_loc_refl_coefs[s], 'regular', self.order)
            self.ps.interface.reflected_field = flds.SphericalWaveExpansion(self.ps.incident_field.ampl, self.ps.k_fluid,
                                                                            reflected_coefs, 'regular', self.order)
            self.solution_coefficients = incident_coefs, scattered_coefs, inner_coefs, reflected_coefs, \
                                         local_reflected_coefs
        else:
            scattered_coefs1d = np.dot(self.t_matrix, self.rhs)
            scattered_coefs = scattered_coefs1d.reshape((self.ps.num_sph, (self.order + 1) ** 2))
            inner_coefs = tsystem.inner_coefficients(scattered_coefs, self.ps, self.order)
            incident_coefs = self.rhs
            self.solution_coefficients = incident_coefs, scattered_coefs, inner_coefs

        sep_loc_inc_coefs = incident_coefs.reshape((self.ps.num_sph, (self.order + 1) ** 2))
        for s in range(self.ps.num_sph):
            self.ps.spheres[s].incident_field = flds.SphericalWaveExpansion(self.ps.incident_field.ampl, self.ps.k_fluid,
                                                                            sep_loc_inc_coefs[s], 'regular', self.order)
            self.ps.spheres[s].scattered_field = flds.SphericalWaveExpansion(self.ps.incident_field.ampl, self.ps.k_fluid,
                                                                             scattered_coefs[s], 'outgoing', self.order)
            self.ps.spheres[s].inner_field = flds.SphericalWaveExpansion(self.ps.incident_field.ampl, self.ps.k_spheres[s],
                                                                         inner_coefs[s], 'regular', self.order)
