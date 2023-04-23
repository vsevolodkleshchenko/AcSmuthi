from acsmuthi.simulation import Simulation
from acsmuthi.particles import SphericalParticle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import PlaneWave
from acsmuthi.postprocessing import forces, cross_sections as cs
import numpy as np


def test_four_gold_spheres_in_water():
    # parameters of medium
    rho_fluid = 997  # [kg/m^3]
    c_fluid = 1403  # [m/s]

    # parameters of incident field
    direction = np.array([0.57735, 0.57735, -0.57735])
    freq = 300  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres
    r_sph = 1  # [m]
    ro_sph = 19300  # [kg/m^3]
    c_sph_l = 3240  # [m/s]
    c_sph_t = 1200  # [m/s]

    pos2 = np.array([1.7 * r_sph, 1.6 * r_sph, 0.1 * r_sph])  # [m]
    pos3 = np.array([-1.5 * r_sph, 2 * r_sph, -1.4 * r_sph])  # [m]
    pos4 = np.array([-1.6 * r_sph, -1.3 * r_sph, 2.1 * r_sph])  # [m]
    pos1 = np.array([1.2 * r_sph, -1.5 * r_sph, -1.7 * r_sph])  # [m]

    order = 8

    incident_field = PlaneWave(k_l=k_l,
                               amplitude=p0,
                               direction=direction)

    fluid = Medium(density=rho_fluid, speed_l=c_fluid)

    sphere1 = SphericalParticle(position=pos1,
                       radius=r_sph,
                       density=ro_sph,
                       speed_l=c_sph_l,
                       order=order,
                       speed_t=c_sph_t)

    sphere2 = SphericalParticle(position=pos2,
                       radius=r_sph,
                       density=ro_sph,
                       speed_l=c_sph_l,
                       order=order,
                       speed_t=c_sph_t)

    sphere3 = SphericalParticle(position=pos3,
                       radius=r_sph,
                       density=ro_sph,
                       speed_l=c_sph_l,
                       order=order,
                       speed_t=c_sph_t)

    sphere4 = SphericalParticle(position=pos4,
                       radius=r_sph,
                       density=ro_sph,
                       speed_l=c_sph_l,
                       order=order,
                       speed_t=c_sph_t)

    particles = np.array([sphere1, sphere2, sphere3, sphere4])

    sim = Simulation(
        particles=particles,
        medium=fluid,
        initial_field=incident_field,
        frequency=freq,
        order=order,
        store_t_matrix=True
    )
    sim.run()

    scs = cs.extinction_cs(
        particles=particles,
        medium=fluid,
        incident_field=incident_field,
        freq=freq
    )

    frcs = forces.all_forces(
        particles_array=particles,
        medium=fluid,
        incident_field=incident_field
    )

    comsol_scs = 5.2806  # 5.2753
    comsol_frcs = np.array([[2.0505E-10, 3.0999E-10, -1.7603E-10], [3.9687E-10, 3.6124E-10,	-3.3042E-10],
                            [1.8437E-10, 1.9482E-10, -1.6927E-10], [2.0023E-10,	2.1854E-10, -1.4755E-10]])
    np.testing.assert_allclose(frcs, comsol_frcs, rtol=5e-2)
    assert np.round(scs, 1) == np.round(comsol_scs, 1)
