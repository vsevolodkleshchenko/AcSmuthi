import classes as cls
import numpy as np


def build_ps_2s():
    r"""Builds physical system with 2 spheres"""
    # parameters of incident field
    direction = np.array([0.70711, 0, 0.70711])
    freq = 82  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    incident_field = cls.PlaneWave(direction, freq, p0)

    # parameters of fluid
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]
    fluid = cls.Fluid(ro_fluid, c_fluid)

    # parameters of the spheres
    pos1 = np.array([0, 0, -2.5])
    pos2 = np.array([0, 0, 2.5])
    r_sph = 1  # [m]
    ro_sph = 1050  # [kg/m^3]
    c_sph = 1403  # [m/s]

    sphere1 = cls.Sphere(pos1, r_sph, ro_sph, c_sph)
    sphere2 = cls.Sphere(pos2, r_sph, ro_sph, c_sph)
    spheres = np.array([sphere1, sphere2])

    ps = cls.System(fluid, incident_field, spheres)
    return ps


def build_ps_1s():
    r"""Builds physical system with 1 sphere"""
    # parameters of incident field
    direction = np.array([0, 0, 1.])
    freq = 82  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    incident_field = cls.PlaneWave(direction, freq, p0)

    # parameters of fluid
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]
    fluid = cls.Fluid(ro_fluid, c_fluid)

    # parameters of the spheres
    pos = np.array([0, 0, 0])
    r_sph = 1  # [m]
    ro_sph = 1050  # [kg/m^3]
    c_sph = 1403  # [m/s]

    sphere = cls.Sphere(pos, r_sph, ro_sph, c_sph)
    spheres = np.array([sphere])

    ps = cls.System(fluid, incident_field, spheres)
    return ps


def build_ps_3s():
    r"""Builds physical system with 3 spheres"""
    # parameters of incident field
    direction = np.array([1, 0, 0])
    freq = 82  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    incident_field = cls.PlaneWave(direction, freq, p0)

    # parameters of fluid
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]
    fluid = cls.Fluid(ro_fluid, c_fluid)

    # parameters of the spheres
    pos1 = np.array([0, 0, -2])
    pos2 = np.array([0, 0, 2])
    pos3 = np.array([3.4, 0, 0])
    r_sph = 1  # [m]
    ro_sph = 1050  # [kg/m^3]
    c_sph = 1403  # [m/s]

    sphere1 = cls.Sphere(pos1, r_sph, ro_sph, c_sph)
    sphere2 = cls.Sphere(pos2, r_sph, ro_sph, c_sph)
    sphere3 = cls.Sphere(pos3, r_sph, ro_sph, c_sph)
    spheres = np.array([sphere1, sphere2, sphere3])

    ps = cls.System(fluid, incident_field, spheres)
    return ps


def build_ps_2s_i():
    r"""Builds physical system with 2 spheres and 1 interface(layer)"""
    # parameters of incident field
    direction = np.array([-1, 0, 0])
    freq = 82  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    incident_field = cls.PlaneWave(direction, freq, p0)

    # parameters of fluid
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]
    fluid = cls.Fluid(ro_fluid, c_fluid)

    # parameters of the spheres
    pos1 = np.array([0, 0, -2.5])
    pos2 = np.array([0, 0, 2.5])
    r_sph = 1  # [m]
    ro_sph = 1050  # [kg/m^3]
    c_sph = 1403  # [m/s]

    sphere1 = cls.Sphere(pos1, r_sph, ro_sph, c_sph)
    sphere2 = cls.Sphere(pos2, r_sph, ro_sph, c_sph)
    spheres = np.array([sphere1, sphere2])

    # parameters of interface (substrate)
    a, b, c, d = 1, 0, 0, 2
    ro_interface = ro_sph
    c_interface = c_sph
    interface = cls.Interface(ro_interface, c_interface, a, b, c, d)

    ps = cls.System(fluid, incident_field, spheres, interface)
    return ps
