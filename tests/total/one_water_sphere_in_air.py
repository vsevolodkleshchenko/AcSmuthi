from acsmuthi.linear_system.linear_system import LinearSystem
from acsmuthi.particles import Particle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import PlaneWave
from acsmuthi.postprocessing import forces, cross_sections as cs
import numpy as np


def test_one_water_sphere_in_air():
    # parameters of medium
    rho_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    # parameters of incident field
    direction = np.array([0, 0, 1])
    freq = 10_000  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres
    r_sph = 0.01  # [m]
    ro_sph = 997  # [kg/m^3]
    c_sph_l = 1403  # [m/s]

    pos1 = np.array([0, 0, 0])  # [m]

    order = 8

    incident_field = PlaneWave(k_l=k_l,
                               amplitude=p0,
                               direction=direction)

    fluid = Medium(density=rho_fluid, speed_l=c_fluid)

    sphere1 = Particle(position=pos1,
                       radius=r_sph,
                       density=ro_sph,
                       speed_l=c_sph_l,
                       order=order)

    particles = np.array([sphere1])

    linear_system = LinearSystem(particles=particles,
                                 medium=fluid,
                                 initial_field=incident_field,
                                 frequency=freq,
                                 order=order,
                                 store_t_matrix=True)
    linear_system.prepare()
    linear_system.solve()

    scs = cs.extinction_cs(particles=particles,
                           medium=fluid,
                           incident_field=incident_field,
                           freq=freq)

    frcs = forces.all_forces(particles_array=particles,
                             medium=fluid,
                             incident_field=incident_field)

    comsol_scs = 9.5773E-4  # 9.6687E-4
    comsol_frcs = np.array([[3.3725E-14, 3.1652E-14, -3.1027E-14]])

    # np.testing.assert_allclose(frcs, comsol_frcs, rtol=1e-1)
    assert scs == 9.5773E-4  # 9.6687E-4
