from acsmuthi.simulation import Simulation
from acsmuthi.particles import SphericalParticle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import PlaneWave
from acsmuthi.postprocessing import forces, rendering, cross_sections as cs
import numpy as np


def test_three_hg_spheres_above_hard_substrate():
    # parameters of medium
    rho_fluid = 997  # [kg/m^3]
    c_fluid = 1403  # [m/s]

    # parameters of incident field
    direction = np.array([0.5, -0.5, -0.70711])
    freq = 300  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres
    r_sph = 1  # [m]
    ro_sph = 13870  # [kg/m^3]
    c_sph = 1420  # [m/s]

    pos1 = np.array([-1.8, 2.1, 1.9])  # [m]
    pos2 = np.array([1.7, -1.4, 1.6])  # [m]
    pos3 = np.array([-1.2, -0.8, 1.9])  # [m]

    order = 6

    incident_field = PlaneWave(k=k_l, amplitude=p0, direction=direction)

    fluid = Medium(density=rho_fluid, pressure_velocity=c_fluid, hard_substrate=True)

    sphere1 = SphericalParticle(position=pos1, radius=r_sph, density=ro_sph, pressure_velocity=c_sph, order=order)
    sphere2 = SphericalParticle(position=pos2, radius=r_sph, density=ro_sph, pressure_velocity=c_sph, order=order)
    sphere3 = SphericalParticle(position=pos3, radius=r_sph, density=ro_sph, pressure_velocity=c_sph, order=order)
    particles = np.array([sphere1, sphere2, sphere3])

    sim = Simulation(particles=particles, medium=fluid, initial_field=incident_field, frequency=freq, order=order)
    sim.run()

    scs = cs.extinction_cs(simulation=sim)
    frcs = forces.all_forces(sim)

    comsol_scs = 6.9578
    comsol_frcs = np.array([
        [1.0424e-10, -1.3920e-10, -4.5023e-10],
        [1.2452e-11, -1.8790e-10, -1.5016e-10],
        [1.9203e-10, -4.6026e-11, -5.8631e-10]
    ])

    np.testing.assert_allclose(np.where(np.abs(frcs) > 1e-12, frcs, 0), comsol_frcs, rtol=2e-2)
    assert np.round(scs, 1) == np.round(comsol_scs, 1)