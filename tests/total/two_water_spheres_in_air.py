from acsmuthi.simulation import Simulation
from acsmuthi.particles import SphericalParticle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import PlaneWave
from acsmuthi.postprocessing import forces, cross_sections as cs
import numpy as np


def test_two_water_spheres_in_air():
    # parameters of medium
    rho_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    # parameters of incident field
    direction = np.array([0.70711, 0, 0.70711])
    freq = 82  # [Hz] 10000
    p0 = 1  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres
    r_sph = 1  # [m]
    ro_sph = 997  # [kg/m^3]
    c_sph_l = 1403  # [m/s]

    # pos1 = np.array([0, -1.6 * r_sph, 0])  # [m]
    # pos2 = np.array([0, 1.6 * r_sph, 0])  # [m]
    pos1 = np.array([-2.5, 0, 0])  # [m]
    pos2 = np.array([2.5, 0, 0])  # [m]

    order = 9

    incident_field = PlaneWave(k_l=k_l,
                               amplitude=p0,
                               direction=direction)

    fluid = Medium(density=rho_fluid, speed_l=c_fluid)

    sphere1 = SphericalParticle(position=pos1,
                       radius=r_sph,
                       density=ro_sph,
                       speed_l=c_sph_l,
                       order=order)

    sphere2 = SphericalParticle(position=pos2,
                       radius=r_sph,
                       density=ro_sph,
                       speed_l=c_sph_l,
                       order=order)

    particles = np.array([sphere1, sphere2])
    sim = Simulation(
        particles=particles,
        medium=fluid,
        initial_field=incident_field,
        frequency=freq,
        order=order,
        store_t_matrix=True
    )
    sim.run()

    scs = cs.extinction_cs(particles=particles,
                           medium=fluid,
                           incident_field=incident_field,
                           freq=freq)

    frcs = forces.all_forces(particles_array=particles,
                             medium=fluid,
                             incident_field=incident_field)

    # comsol_frcs = np.array([[6.7778E-6, -9.7296E-9, 5.8066E-6], [5.5507E-6, 9.0569E-9, 6.0051E-6]])
    comsol_scs = 3.9354  # 3.9339
    comsol_frcs = np.array([[6.8436E-6, 0, 5.8481E-6], [5.5495E-6, 0, 5.9831E-6]])

    np.testing.assert_allclose(np.where(np.abs(frcs) <= 1e-14, 0, frcs), comsol_frcs, rtol=1e-2)
    assert np.round(scs, 2) == np.round(comsol_scs, 2)

