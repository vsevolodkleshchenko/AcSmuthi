import math

from acsmuthi.utility import mathematics as mths
import numpy as np
import scipy.special as ss


class OneSphericalParticleSimulation:
    def __init__(self, freq, density_particle, cp_particle, radius_particle, density_fluid, cp_fluid, p0, order):
        self.freq = freq
        self.rho_p = density_particle
        self.c_p = cp_particle
        self.r_p = radius_particle
        self.rho_fluid = density_fluid
        self.c_fluid = cp_fluid
        self.order = order
        self.ampl = p0
        self.inc_coefs = None
        self.sc_coefs = None
        self.in_coefs = None

    def count_incident_coefficients(self):
        coefficients = np.zeros(self.order + 1, dtype=complex)
        for n in range(self.order + 1):
            coefficients[n] = 1j ** n * (2 * n + 1)
        self.inc_coefs = coefficients

    def count_scattered_coefficients(self):
        coefficients = np.zeros(self.order + 1, dtype=complex)
        k_p, k_fluid = 2 * np.pi * self.freq / self.c_p, 2 * np.pi * self.freq / self.c_fluid
        gamma = k_p * self.rho_fluid / k_fluid / self.rho_p
        for n in range(self.order + 1):
            term1 = ss.spherical_jn(n, k_p * self.r_p, derivative=True) * ss.spherical_jn(n, k_fluid * self.r_p)
            term2 = ss.spherical_jn(n, k_p * self.r_p) * ss.spherical_jn(n, k_fluid * self.r_p, derivative=True)
            term3 = ss.spherical_jn(n, k_p * self.r_p) * mths.spherical_h1n(n, k_fluid * self.r_p, derivative=True)
            term4 = ss.spherical_jn(n, k_p * self.r_p, derivative=True) * mths.spherical_h1n(n, k_fluid * self.r_p)
            coefficients[n] = (gamma * term1 - term2) / (term3 - gamma * term4)
        self.sc_coefs = coefficients

    def count_inner_coefficients(self):
        coefficients = np.zeros(self.order + 1, dtype=complex)
        k_p, k_fluid = 2 * np.pi * self.freq / self.c_p, 2 * np.pi * self.freq / self.c_fluid
        gamma = k_p * self.rho_fluid / k_fluid / self.rho_p
        for n in range(self.order + 1):
            term1 = ss.spherical_jn(n, k_p * self.r_p) * mths.spherical_h1n(n, k_fluid * self.r_p, derivative=True)
            term2 = ss.spherical_jn(n, k_p * self.r_p, derivative=True) * mths.spherical_h1n(n, k_fluid * self.r_p)
            coefficients[n] = 1j / (k_fluid * self.r_p) ** 2 / (term1 - gamma * term2)
        self.in_coefs = coefficients

    def axisymmetric_outgoing_wvfs_array(self, x, y, z, k):
        as_ow_array = np.zeros((self.order + 1, *x.shape), dtype=complex)
        r, phi, theta = mths.dec_to_sph(x, y, z)
        for n in range(self.order + 1):
            as_ow_array[n] = mths.spherical_h1n(n, k * r) * ss.lpmv(0, n, np.cos(theta))
        return as_ow_array

    def axisymmetric_regular_wvfs_array(self, x, y, z, k):
        as_rw_array = np.zeros((self.order + 1, *x.shape), dtype=complex)
        r, phi, theta = mths.dec_to_sph(x, y, z)
        for n in range(self.order + 1):
            as_rw_array[n] = ss.spherical_jn(n, k * r) * ss.lpmv(0, n, np.cos(theta))
        return as_rw_array

    def evaluate_solution(self):
        self.count_incident_coefficients()
        self.count_scattered_coefficients()
        self.count_inner_coefficients()

    def scattering_cs(self):
        k_p, k_fluid = 2 * np.pi * self.freq / self.c_p, 2 * np.pi * self.freq / self.c_fluid
        prefactor = 4 * np.pi / k_fluid ** 2
        sigma_sc_array = np.zeros(self.order + 1)
        for n in range(self.order + 1):
            sigma_sc_array[n] = (2 * n + 1) * np.abs(self.sc_coefs[n]) ** 2
        sigma_geom = np.pi * self.r_p ** 2
        return prefactor * math.fsum(sigma_sc_array) / sigma_geom

    def extinction_cs(self):
        k_p, k_fluid = 2 * np.pi * self.freq / self.c_p, 2 * np.pi * self.freq / self.c_fluid
        prefactor = - 4 * np.pi / k_fluid ** 2
        sigma_ex_array = np.zeros(self.order + 1)
        for n in range(self.order + 1):
            sigma_ex_array[n] = (2 * n + 1) * np.real(self.sc_coefs[n])
        sigma_geom = np.pi * self.r_p ** 2
        return prefactor * math.fsum(sigma_ex_array) / sigma_geom

    def compute_pressure_field(self, x, y, z):
        outgoing_wvfs = self.axisymmetric_outgoing_wvfs_array(x, y, z, 2 * np.pi * self.freq / self.c_fluid)
        regular_wvfs = self.axisymmetric_regular_wvfs_array(x, y, z, 2 * np.pi * self.freq / self.c_p)
        total_sc_coefs = np.broadcast_to(self.inc_coefs * self.sc_coefs, outgoing_wvfs.T.shape).T
        total_in_coefs = np.broadcast_to(self.inc_coefs * self.in_coefs, regular_wvfs.T.shape).T
        scattered_field = self.ampl * np.sum(np.real(total_sc_coefs * outgoing_wvfs), axis=0)
        inner_field = self.ampl * np.sum(np.real(total_in_coefs * regular_wvfs), axis=0)
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return np.where(r <= self.r_p, inner_field, 0) + np.where(r >= self.r_p, scattered_field, 0)

    def forces(self):
        k, beta = 2 * np.pi * self.freq / self.c_fluid, 1 / (self.rho_fluid * self.c_fluid ** 2)
        prefactor = - 2 * np.pi / k ** 2 * self.ampl ** 2 * beta
        a = self.sc_coefs
        forces_array = np.zeros(self.order + 1)
        for n in range(self.order):
            forces_array[n] = (2 * n + 1) * np.real(a[n]) + 2 * (n + 1) * np.real(np.conj(a[n]) * a[n + 1])
        return prefactor * np.sum(forces_array)
