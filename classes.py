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


def build_ps():
    # parameters of incident field
    direction = np.array([0.70711, 0, 0.70711])
    freq = 82  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    incident_field = PlaneWave(direction, freq, p0)

    # parameters of fluid
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]
    fluid = Fluid(ro_fluid, c_fluid)

    # parameters of the spheres
    pos1 = np.array([0, 0, -2.5])
    pos2 = np.array([0, 0, 2.5])
    r_sph = 1  # [m]
    ro_sph = 1050  # [kg/m^3]
    c_sph = 1403  # [m/s]

    sphere_cls1 = Sphere(pos1, r_sph, ro_sph, c_sph)
    sphere_cls2 = Sphere(pos2, r_sph, ro_sph, c_sph)
    spheres_cls = np.array([sphere_cls1, sphere_cls2])

    ps = System(fluid, incident_field, spheres_cls)
    return ps


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
