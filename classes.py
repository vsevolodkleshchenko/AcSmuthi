import numpy as np


class Sphere:
    def __init__(self, position, radius, density, speed_of_sound):
        self.pos = position
        self.r = radius
        self.rho = density
        self.speed = speed_of_sound


class PlaneWave:
    def __init__(self, direction, frequency, amplitude):
        self.dir = direction
        self.freq = frequency
        self.ampl = amplitude

    @property
    def omega(self):
        return 2 * np.pi * self.freq


class Fluid:
    def __init__(self, density, speed_of_sound):
        self.rho = density
        self.speed = speed_of_sound


class Interface:
    def __init__(self, density, speed_of_sound, a, b, c, d):
        self.rho = density
        self.speed = speed_of_sound
        self.a = a
        self.b = b
        self.c = c
        self.d = d

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


class System:
    r"""Class for total physical system"""
    def __init__(self, fluid, incident_field, spheres, interface=None):
        self.fluid = fluid
        self.incident_field = incident_field
        self.spheres = spheres
        self.interface = interface

    @property
    def num_sph(self):
        return len(self.spheres)

    @property
    def omega(self):
        return self.incident_field.omega

    @property
    def freq(self):
        return self.incident_field.freq

    @property
    def k_fluid(self):
        return self.incident_field.omega / self.fluid.speed

    @property
    def k_spheres(self):
        k_spheres_array = np.array([self.incident_field.omega / self.spheres[i].speed for i in range(self.num_sph)])
        return k_spheres_array

    @property
    def intensity_incident_field(self):
        return self.incident_field.ampl ** 2 / (2 * self.fluid.rho * self.fluid.speed)
