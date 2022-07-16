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


class System:
    def __init__(self, fluid, incident_field, spheres):
        self.fluid = fluid
        self.incident_field = incident_field
        self.spheres = spheres

    @property
    def num_sph(self):
        return len(self.spheres)

    # @property
    # def omega(self):
    #     return self.incident_field.omega
    #
    # @property
    # def freq(self):
    #     return self.incident_field.freq

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


def ps_to_param(ps):
    freq = ps.incident_field.freq

    k = ps.k_fluid * ps.incident_field.dir

    k_fluid = ps.k_fluid

    ro_fluid = ps.fluid.rho

    positions = np.zeros((ps.num_sph, 3))
    for sph in range(ps.num_sph):
        positions[sph] = ps.spheres[sph].pos

    spheres = np.zeros((ps.num_sph, 3))
    for sph in range(ps.num_sph):
        spheres[sph] = np.array([ps.k_spheres[sph], ps.spheres[sph].r, ps.spheres[sph].rho])

    p0 = ps.incident_field.ampl

    intensity = ps.intensity_incident_field

    num_sph = ps.num_sph

    return freq, k, k_fluid, ro_fluid, positions, spheres, p0, intensity, num_sph
