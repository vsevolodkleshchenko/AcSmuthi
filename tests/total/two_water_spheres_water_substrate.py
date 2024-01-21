from acsmuthi.simulation import Simulation
from acsmuthi.particles import SphericalParticle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import PlaneWave
from acsmuthi.postprocessing import forces, cross_sections as cs
import numpy as np


def test_two_water_spheres_above_water_substrate():
    # parameters of medium
    rho_fluid = 825  # [kg/m^3]
    c_fluid = 1290  # [m/s]

    # parameters of incident field
    direction = np.array([0.587785, 0., -0.809017])
    freq = 320  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres and the substrate
    r_sph = 1  # [m]
    ro_sph = 1000  # [kg/m^3]
    c_sph = 1480  # [m/s]

    pos1 = np.array([-1.7, 0, 2.3])  # [m]
    pos2 = np.array([1.8, 0., 2.5])  # [m]

    order = 8

    incident_field = PlaneWave(k=k_l, amplitude=p0, direction=direction)

    fluid = Medium(density=rho_fluid, pressure_velocity=c_fluid, substrate_density=ro_sph, substrate_velocity=c_sph)

    sphere1 = SphericalParticle(position=pos1, radius=r_sph, density=ro_sph, pressure_velocity=c_sph, order=order)
    sphere2 = SphericalParticle(position=pos2, radius=r_sph, density=ro_sph, pressure_velocity=c_sph, order=order)
    particles = np.array([sphere1, sphere2])

    sim = Simulation(particles=particles, medium=fluid, initial_field=incident_field, frequency=freq, order=order)
    sim.run()

    scs = cs.extinction_cs(simulation=sim)
    frcs = forces.all_forces(sim)

    comsol_scs = 1.0587
    comsol_frcs = np.array([[1.2520E-10, 0., -1.6968E-10], [9.4881E-11, 0., -1.0201E-10]])

    np.testing.assert_allclose(np.where(np.abs(frcs) > 1e-14, frcs, 0), comsol_frcs, rtol=5e-3)
    assert np.round(scs, 2) == np.round(comsol_scs, 2)