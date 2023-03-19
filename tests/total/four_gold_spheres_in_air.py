from acsmuthi.linear_system.linear_system import LinearSystem
from acsmuthi.particles import Particle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import PlaneWave
from acsmuthi.postprocessing import forces, cross_sections as cs
import numpy as np


def test_four_gold_spheres_in_air():
    # parameters of medium
    rho_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    # parameters of incident field
    direction = np.array([0.57735, 0.57735, -0.57735])
    freq = 60_000  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres
    r_sph = 0.01  # [m]
    ro_sph = 19300  # [kg/m^3]
    c_sph_l = 3240  # [m/s]
    c_sph_t = 1200  # [m/s]

    pos1 = np.array([1.7 * r_sph, 1.6 * r_sph, -1.3 * r_sph])  # [m]
    pos2 = np.array([-1.5 * r_sph, 2 * r_sph, -1.4 * r_sph])  # [m]
    pos3 = np.array([-1.6 * r_sph, -1.3 * r_sph, 2.1 * r_sph])  # [m]
    pos4 = np.array([1.2 * r_sph, -1.5 * r_sph, -1.7 * r_sph])  # [m]

    order = 8

    incident_field = PlaneWave(k_l=k_l,
                               amplitude=p0,
                               direction=direction)

    fluid = Medium(density=rho_fluid, speed_l=c_fluid)

    sphere1 = Particle(position=pos1,
                       radius=r_sph,
                       density=ro_sph,
                       speed_l=c_sph_l,
                       order=order,
                       speed_t=c_sph_t)

    sphere2 = Particle(position=pos2,
                       radius=r_sph,
                       density=ro_sph,
                       speed_l=c_sph_l,
                       order=order,
                       speed_t=c_sph_t)

    sphere3 = Particle(position=pos3,
                       radius=r_sph,
                       density=ro_sph,
                       speed_l=c_sph_l,
                       order=order,
                       speed_t=c_sph_t)

    sphere4 = Particle(position=pos4,
                       radius=r_sph,
                       density=ro_sph,
                       speed_l=c_sph_l,
                       order=order,
                       speed_t=c_sph_t)

    particles = np.array([sphere1, sphere2, sphere3, sphere4])

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
    comsol_frcs = np.array([[3.3725E-14, 3.1652E-14, -3.1027E-14], [2.7348E-14, 3.5052E-14, -2.8459E-14],
                            [2.7348E-14, 3.5052E-14, -2.8459E-14], [3.2758E-14, 3.2033E-14, -3.1599E-14]])

    # np.testing.assert_allclose(frcs, comsol_frcs, rtol=1e-1)
    assert scs == 9.5773E-4  # 9.6687E-4