from acsmuthi.simulation import Simulation
from acsmuthi.particles import SphericalParticle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import PlaneWave
from acsmuthi.postprocessing import forces, cross_sections as cs
import numpy as np


def test_four_water_spheres_in_air():
    # parameters of medium
    rho_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    # parameters of incident field
    direction = np.array([0.57735, 0.57735, -0.57735])
    freq = 80  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres
    r_sph = 1  # [m]
    ro_sph = 997  # [kg/m^3]
    c_sph_l = 1403  # [m/s]

    pos2 = np.array([1.7 * r_sph, 1.6 * r_sph, 0.1 * r_sph])  # [m]
    pos3 = np.array([-1.5 * r_sph, 2 * r_sph, -1.4 * r_sph])  # [m]
    pos4 = np.array([-1.6 * r_sph, -1.3 * r_sph, 2.1 * r_sph])  # [m]
    pos1 = np.array([1.2 * r_sph, -1.5 * r_sph, -1.7 * r_sph])  # [m]

    order = 9

    incident_field = PlaneWave(k=k_l, amplitude=p0, direction=direction)

    fluid = Medium(density=rho_fluid, pressure_velocity=c_fluid)

    sphere1 = SphericalParticle(position=pos1, radius=r_sph, density=ro_sph, pressure_velocity=c_sph_l, order=order)

    sphere2 = SphericalParticle(position=pos2, radius=r_sph, density=ro_sph, pressure_velocity=c_sph_l, order=order)

    sphere3 = SphericalParticle(position=pos3, radius=r_sph, density=ro_sph, pressure_velocity=c_sph_l, order=order)

    sphere4 = SphericalParticle(position=pos4, radius=r_sph, density=ro_sph, pressure_velocity=c_sph_l, order=order)

    particles = np.array([sphere1, sphere2, sphere3, sphere4])

    sim = Simulation(particles=particles, medium=fluid, initial_field=incident_field, frequency=freq, order=order)
    sim.run()

    scs = cs.extinction_cs(simulation=sim)

    frcs = forces.all_forces(sim)

    comsol_scs = 8.0687	 # 8.0799
    comsol_frcs = np.array([[4.2871E-6,	6.8757E-6, -3.4738E-6],	[6.3812E-6,	5.9692E-6, -4.8634E-6],
                            [4.9670E-6,	3.1631E-6, -2.7768E-6],	[5.8094E-6,	6.0099E-6, -6.1842E-6]])
    np.testing.assert_allclose(frcs, comsol_frcs, rtol=5e-2)
    assert np.round(scs, 1) == np.round(comsol_scs, 1)