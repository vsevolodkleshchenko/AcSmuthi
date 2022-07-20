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

    sphere1 = Sphere(pos1, r_sph, ro_sph, c_sph)
    sphere2 = Sphere(pos2, r_sph, ro_sph, c_sph)
    spheres = np.array([sphere1, sphere2])

    ps = System(fluid, incident_field, spheres)
    return ps


def build_ps_1s():
    # parameters of incident field
    direction = np.array([0, 0, 0])
    freq = 82  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    incident_field = PlaneWave(direction, freq, p0)

    # parameters of fluid
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]
    fluid = Fluid(ro_fluid, c_fluid)

    # parameters of the spheres
    pos = np.array([0, 0, 0])
    r_sph = 1  # [m]
    ro_sph = 1050  # [kg/m^3]
    c_sph = 1403  # [m/s]

    sphere = Sphere(pos, r_sph, ro_sph, c_sph)
    spheres = np.array([sphere])

    ps = System(fluid, incident_field, spheres)
    return ps


def build_ps_3s():
    # parameters of incident field
    direction = np.array([1, 0, 0])
    freq = 82  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    incident_field = PlaneWave(direction, freq, p0)

    # parameters of fluid
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]
    fluid = Fluid(ro_fluid, c_fluid)

    # parameters of the spheres
    pos1 = np.array([0, 0, -2])
    pos2 = np.array([0, 0, 2])
    pos3 = np.array([3.4, 0, 0])
    r_sph = 1  # [m]
    ro_sph = 1050  # [kg/m^3]
    c_sph = 1403  # [m/s]

    sphere1 = Sphere(pos1, r_sph, ro_sph, c_sph)
    sphere2 = Sphere(pos2, r_sph, ro_sph, c_sph)
    sphere3 = Sphere(pos3, r_sph, ro_sph, c_sph)
    spheres = np.array([sphere1, sphere2, sphere3])

    ps = System(fluid, incident_field, spheres)
    return ps
