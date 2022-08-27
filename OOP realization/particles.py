import wavefunctions as wvfs
import tsystem
import numpy as np


class Particle:
    def __init__(self, position, radius, density, speed_of_sound, order):
        self.pos = position
        self.r = radius
        self.rho = density
        self.speed = speed_of_sound
        self.incident_field = None
        self.scattered_field = None
        self.inner_filed = None
        self.reflected_field = None
        self.t_matrix = None
        self.order = order

    def t_matrix(self, sph_number, ps):
        t = np.zeros(((self.order+1)**2, (self.order+1)**2), dtype=complex)
        for i, n, m in enumerate(wvfs.multipoles(self.order)):
            t[i, i] = tsystem.scaled_coefficient(n, sph_number, ps)
        self.t_matrix = t

    def incident_field_decomposition(self, ps):
        d = np.zeros((self.order+1)**2, dtype=complex)
        for i, n, m in enumerate(wvfs.multipoles(self.order)):
            d[i] = wvfs.local_incident_coefficient(m, n, ps.k, ps.incident_field.dir, self.pos, self.order)
        self.incident_field = d
