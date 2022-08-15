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
        return np.abs(self.a * position[0] + self.b * position[1] + self.c * position[2] + self.d) / \
               np.sqrt(self.a ** 2 + self.b ** 2 + self.c ** 2)

    @property
    def int_dist0(self):
        return self.int_dist(np.array([0., 0., 0.]))

    @property
    def normal(self):
        n = np.array(np.array([self.a, self.b, self.c]) / np.sqrt(self.a ** 2 + self.b ** 2 + self.c ** 2))
        n_dist = n * self.int_dist0
        if n_dist[0] * self.a + n_dist[1] * self.b + n_dist[2] * self.c == -self.d:
            n *= -1
        return n


class System:
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


def build_ps_2s():
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
    direction = np.array([0, 0, 1.])
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


def build_ps_2s_l():
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
    pos1 = np.array([0, 0, -2.5])
    pos2 = np.array([0, 0, 2.5])
    r_sph = 1  # [m]
    ro_sph = 1050  # [kg/m^3]
    c_sph = 1403  # [m/s]

    sphere1 = Sphere(pos1, r_sph, ro_sph, c_sph)
    sphere2 = Sphere(pos2, r_sph, ro_sph, c_sph)
    spheres = np.array([sphere1, sphere2])

    # parameters of interface (substrate)
    a, b, c, d = 1, 0, 0, -2
    ro_interface = ro_sph
    c_interface = c_sph
    interface = Interface(ro_interface, c_interface, a, b, c, d)

    ps = System(fluid, incident_field, spheres, interface)
    return ps